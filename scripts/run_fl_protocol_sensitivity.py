#!/usr/bin/env python3
"""
Federated Scoring-Protocol Sensitivity Benchmark (Unstratified & Class-Stratified S2-session)
---------------------------------------------------------------------------------------------
Evaluates FedAvg, FedProx, FedPer (with post-hoc support-masked FedPer-Guard diagnostic),
warm-started Adam pFedMe (K=3 inner steps per mini-batch by default, configurable K=1),
and Ditto across:
  - UAH-DriveSet (n=6 drivers, 40 trips; RAW physical units with train-only scaling)
  - UCI HAR      (n=30 subjects, 400 sessions)
  - WISDM 1.1    (n=51 subjects, 307 segments; train-only scaling)

Supports both:
  - --split-mode unstratified : original 30% within-client session holdout WITH explicit
    seen-class vs. zero-support-class decomposition + zero-support fraction logging.
  - --split-mode stratified   : per-(client, class) stratified whole-session holdout
    guaranteeing >= 1 training session per held-out test class (0% zero-support deletion).
  - --backbone cnn | gru      : Conv1DFLNet (no BatchNorm, standard for FL) or GRU1DFLNet.
  - --ditto-lam               : configurable Ditto proximal coefficient (default 0.1).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_entity_eval import (  # noqa: E402
    UnitLevelEvaluator,
    exact_paired_sign_flip_test,
    fixed_macro_f1,
    holm_bonferroni,
    support_masked_head_avg,
    train_only_standardize,
)
from run_entity_decoupling import DatasetSpec, load_har, load_uah, load_wisdm  # noqa: E402


DEFAULT_CONFIG = {
    "seeds": [1, 2, 3, 4, 5, 6],
    "rounds": 30,
    "local_epochs": 2,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "fedprox_mu": 0.01,
    "pfedme_lam": 1.0,
    "pfedme_K": 3,
    "pfedme_lr_in": 1e-3,
    "pfedme_outer_step": 0.25,
    "pfedme_beta": 1.0,
    "ditto_lam": 0.1,
    "s2_session_frac": 0.30,
    "s1_window_frac": 0.20,
    "fedper_split": "shared=[conv1,conv2,fc1], personal=[fc2]",
    "fedper_uniform_head": "sample-weighted FedAvg of client fc2 heads",
    "fedper_guard_mode": "support_masked_head_averaging",
    "backbone": "cnn",
    "split_mode": "unstratified",
    "drop_channels": [],
}


class Conv1DFLNet(nn.Module):
    """Two-conv 1D-CNN without BatchNorm1d (avoiding FedAvg BN-stat drift)."""

    def __init__(self, n_channels: int, length: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * (length // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class GRU1DFLNet(nn.Module):
    """Lightweight 1-layer bidirectional GRU backbone for architecture-sensitivity checks."""

    def __init__(self, n_channels: int, length: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = _GRUEncoder(n_channels, hidden=64, out_dim=128)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _GRUEncoder(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.AvgPool1d(kernel_size=4, stride=4)
        self.gru = nn.GRU(n_channels, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, out_dim), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> subsample 4x -> (B, L/4, C)
        out, _ = self.gru(self.stem(x).transpose(1, 2))
        pooled = out.mean(dim=1)
        return self.proj(pooled)


def build_backbone(backbone: str, n_channels: int, length: int, n_classes: int) -> nn.Module:
    if backbone == "cnn":
        return Conv1DFLNet(n_channels, length, n_classes)
    if backbone == "gru":
        return GRU1DFLNet(n_channels, length, n_classes)
    raise ValueError(f"Unknown backbone: {backbone}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trip_vote_accuracy(y_true: np.ndarray, y_pred: np.ndarray, trips: np.ndarray) -> float:
    correct = 0
    total = 0
    for tr in np.unique(trips):
        m = trips == tr
        if not np.any(m):
            continue
        true_lab = int(np.bincount(y_true[m]).argmax())
        pred_lab = int(np.bincount(y_pred[m]).argmax())
        correct += int(true_lab == pred_lab)
        total += 1
    return float(correct / total) if total > 0 else float("nan")


def build_client_splits(
    spec: DatasetSpec,
    seed: int,
    s2_session_frac: float = 0.30,
    s1_window_frac: float = 0.20,
    split_mode: str = "unstratified",
) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    clients: Dict[str, Dict[str, np.ndarray]] = {}

    for cid in np.unique(spec.driver):
        c_idx = np.where(spec.driver == cid)[0]
        c_trips = np.unique(spec.trip[c_idx])
        if len(c_trips) < 2:
            continue

        if split_mode == "stratified":
            by_cls: Dict[int, List[str]] = {}
            for tr in c_trips:
                tm = c_idx[spec.trip[c_idx] == tr]
                maj = int(np.bincount(spec.y[tm]).argmax())
                by_cls.setdefault(maj, []).append(str(tr))
            s2_trips = set()
            for cls_id in sorted(by_cls.keys()):
                tr_list = by_cls[cls_id]
                if len(tr_list) < 2:
                    # Keep singleton-class trips in train to guarantee 0% zero-support deletion
                    continue
                perm = rng.permutation(tr_list)
                n_s2 = min(len(tr_list) - 1, max(1, int(round(len(tr_list) * s2_session_frac))))
                s2_trips.update(perm[:n_s2].tolist())
            if not s2_trips:
                continue
        elif split_mode == "unstratified":
            perm_trips = rng.permutation(c_trips)
            n_s2 = max(1, int(round(len(c_trips) * s2_session_frac)))
            n_s2 = min(n_s2, len(c_trips) - 1)
            s2_trips = set(perm_trips[:n_s2].tolist())
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        s2_mask = np.isin(spec.trip[c_idx], list(s2_trips))
        s2_idx = c_idx[s2_mask]
        seen_idx = c_idx[~s2_mask]
        if len(seen_idx) < 5 or len(s2_idx) < 1:
            continue

        perm_seen = rng.permutation(seen_idx)
        n_s1 = max(1, int(round(len(seen_idx) * s1_window_frac)))
        n_s1 = min(n_s1, len(seen_idx) - 1)
        s1_idx = perm_seen[:n_s1]
        tr_idx = perm_seen[n_s1:]

        clients[str(cid)] = {"train": tr_idx, "s1": s1_idx, "s2": s2_idx}
    return clients


def state_dict_avg(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out: Dict[str, torch.Tensor] = {}
    for k in states[0].keys():
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += st[k].float() * (w / total)
        out[k] = acc.to(states[0][k].dtype)
    return out


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)


def train_local_adam(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    ref_params: Dict[str, torch.Tensor] | None = None,
    prox_coeff: float = 0.0,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if ref_params is not None and prox_coeff > 0.0:
                prox = 0.0
                for name, p in model.named_parameters():
                    prox = prox + torch.sum((p - ref_params[name]) ** 2)
                loss = loss + 0.5 * prox_coeff * prox
            loss.backward()
            opt.step()


def run_fl_method(
    method: str,
    spec: DatasetSpec,
    X_all: np.ndarray,
    clients: Dict[str, Dict[str, np.ndarray]],
    cfg: dict,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    set_seed(seed)
    n_channels, length = X_all.shape[1], X_all.shape[2]
    global_model = build_backbone(cfg["backbone"], n_channels, length, spec.n_classes).to(device)
    client_ids = sorted(clients.keys())
    client_models: Dict[str, nn.Module] = {cid: copy.deepcopy(global_model) for cid in client_ids}
    client_loaders = {
        cid: make_loader(X_all[clients[cid]["train"]], spec.y[clients[cid]["train"]], cfg["batch_size"], shuffle=True)
        for cid in client_ids
    }
    client_weights = [len(clients[cid]["train"]) for cid in client_ids]

    for _rnd in range(cfg["rounds"]):
        g_state = copy.deepcopy(global_model.state_dict())
        g_params = {n: p.detach().clone() for n, p in global_model.named_parameters()}
        round_states = []

        for cid in client_ids:
            loader = client_loaders[cid]
            if method == "FedAvg":
                m = copy.deepcopy(global_model)
                train_local_adam(m, loader, device, cfg["local_epochs"], cfg["lr"], cfg["weight_decay"])
                round_states.append(copy.deepcopy(m.state_dict()))
                client_models[cid] = m

            elif method == "FedProx":
                m = copy.deepcopy(global_model)
                train_local_adam(
                    m, loader, device, cfg["local_epochs"], cfg["lr"], cfg["weight_decay"],
                    ref_params=g_params, prox_coeff=cfg["fedprox_mu"],
                )
                round_states.append(copy.deepcopy(m.state_dict()))
                client_models[cid] = m

            elif method == "FedPer":
                m = client_models[cid]
                m.encoder.load_state_dict(global_model.encoder.state_dict())
                train_local_adam(m, loader, device, cfg["local_epochs"], cfg["lr"], cfg["weight_decay"])
                round_states.append(copy.deepcopy(m.state_dict()))

            elif method == "Ditto":
                m_g = copy.deepcopy(global_model)
                train_local_adam(m_g, loader, device, cfg["local_epochs"], cfg["lr"], cfg["weight_decay"])
                round_states.append(copy.deepcopy(m_g.state_dict()))
                m_p = client_models[cid]
                train_local_adam(
                    m_p, loader, device, cfg["local_epochs"], cfg["lr"], cfg["weight_decay"],
                    ref_params=g_params, prox_coeff=cfg["ditto_lam"],
                )

            elif method == "pFedMe":
                # Faithful Dinh et al. (NeurIPS 2020) Algorithm 1 (warm-started inner Adam):
                # K=3 inner proximal updates per mini-batch + outer Moreau-envelope update.
                w_local = copy.deepcopy(global_model)
                theta = copy.deepcopy(global_model)
                opt_in = torch.optim.Adam(
                    theta.parameters(), lr=float(cfg["pfedme_lr_in"]), weight_decay=cfg["weight_decay"]
                )
                crit = nn.CrossEntropyLoss()
                K = int(cfg["pfedme_K"])
                lam = float(cfg["pfedme_lam"])
                outer_step = float(cfg["pfedme_outer_step"])
                theta.train()
                for _ in range(cfg["local_epochs"]):
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        w_ref = [p.detach().clone() for p in w_local.parameters()]
                        for _k in range(K):
                            opt_in.zero_grad()
                            loss = crit(theta(xb), yb)
                            prox = 0.0
                            for p_t, p_w in zip(theta.parameters(), w_ref):
                                prox = prox + torch.sum((p_t - p_w) ** 2)
                            (loss + 0.5 * lam * prox).backward()
                            opt_in.step()
                        with torch.no_grad():
                            for p_w, p_t in zip(w_local.parameters(), theta.parameters()):
                                p_w.data.sub_(outer_step * (p_w.data - p_t.data))
                round_states.append(copy.deepcopy(w_local.state_dict()))
                client_models[cid] = theta

            else:
                raise ValueError(f"Unknown FL method: {method}")

        new_g = state_dict_avg(round_states, client_weights)
        if method == "pFedMe" and cfg["pfedme_beta"] != 1.0:
            beta = cfg["pfedme_beta"]
            for k in new_g:
                new_g[k] = (1.0 - beta) * g_state[k] + beta * new_g[k]
        global_model.load_state_dict(new_g)

    if method in ("FedAvg", "FedProx"):
        for cid in client_ids:
            client_models[cid].load_state_dict(global_model.state_dict())
    elif method == "FedPer":
        sync_mode = cfg.get("fedper_final_sync", "overwrite")
        if sync_mode == "overwrite":
            # Sync final aggregated shared encoder back to all clients so deployed FedPer
            # and FedPer-Guard evaluate on the aligned round-T shared representation.
            for cid in client_ids:
                client_models[cid].encoder.load_state_dict(global_model.encoder.state_dict())
        elif sync_mode == "overwrite_ft1":
            # Sync final shared encoder then run 1 head-only local alignment epoch.
            for cid in client_ids:
                m = client_models[cid]
                m.encoder.load_state_dict(global_model.encoder.state_dict())
                for p in m.encoder.parameters():
                    p.requires_grad_(False)
                opt_h = torch.optim.Adam(m.head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                crit_h = nn.CrossEntropyLoss()
                m.train()
                for xb, yb in client_loaders[cid]:
                    xb, yb = xb.to(device), yb.to(device)
                    opt_h.zero_grad()
                    crit_h(m(xb), yb).backward()
                    opt_h.step()
                for p in m.encoder.parameters():
                    p.requires_grad_(True)
        elif sync_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown fedper_final_sync: {sync_mode}")

    return global_model, client_models


def build_support_guarded_fedper(
    global_model: nn.Module,
    client_models: Dict[str, nn.Module],
    spec: DatasetSpec,
    clients: Dict[str, Dict[str, np.ndarray]],
    support_masked: bool = True,
) -> Dict[str, nn.Module]:
    """Post-hoc support-masked diagnostic probe for FedPer (delegating to fl_entity_eval.support_masked_head_avg).

    When support_masked=True (default), each fallback class row c is averaged strictly
    over clients that contain class c in their local training set (weighted by local
    class-c sample count), preventing negative-logit pollution from unsupported clients
    whose softmax cross-entropy loss applies persistent positive gradient +p_c > 0.
    When support_masked=False, falls back to the unmasked global_model.head row.
    """
    guarded: Dict[str, nn.Module] = {}
    if support_masked:
        c_weights = {cid: m.head.weight.detach().cpu().numpy() for cid, m in client_models.items()}
        c_biases = {cid: m.head.bias.detach().cpu().numpy() for cid, m in client_models.items()}
        c_counts = {
            cid: np.bincount(spec.y[clients[cid]["train"]], minlength=spec.n_classes).astype(np.float64)
            for cid in client_models.keys()
        }
        _, _, guarded_np = support_masked_head_avg(c_weights, c_biases, c_counts)
        for cid, m in client_models.items():
            gm = copy.deepcopy(m)
            gw, gb = guarded_np[cid]
            with torch.no_grad():
                gm.head.weight.data.copy_(torch.from_numpy(gw).to(gm.head.weight.device))
                gm.head.bias.data.copy_(torch.from_numpy(gb).to(gm.head.bias.device))
            guarded[cid] = gm
        return guarded

    g_w = global_model.head.weight.detach().clone()
    g_b = global_model.head.bias.detach().clone()
    for cid, m in client_models.items():
        gm = copy.deepcopy(m)
        seen_classes = set(np.unique(spec.y[clients[cid]["train"]]).tolist())
        with torch.no_grad():
            for c in range(spec.n_classes):
                if c not in seen_classes:
                    gm.head.weight.data[c].copy_(g_w[c])
                    gm.head.bias.data[c].copy_(g_b[c])
        guarded[cid] = gm
    return guarded


def _parse_uah_road(trip_name: str) -> str:
    s = str(trip_name).upper()
    if "MOTORWAY" in s:
        return "MOTORWAY"
    if "SECONDARY" in s:
        return "SECONDARY"
    return "UNKNOWN"


def evaluate_protocol(
    method: str,
    spec: DatasetSpec,
    X_all: np.ndarray,
    clients: Dict[str, Dict[str, np.ndarray]],
    global_model: nn.Module,
    client_models: Dict[str, nn.Module],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    res: Dict[str, float] = {}
    per_client_s2_dep: Dict[str, float] = {}
    per_client_s2_seen: Dict[str, float] = {}

    for split_key in ("s1", "s2"):
        for mode in ("deployed", "uniform"):
            y_true_all, y_pred_all, trip_all = [], [], []
            seen_correct, seen_total = 0, 0
            unseen_correct, unseen_total = 0, 0
            same_road_corr, same_road_tot = 0, 0
            cross_road_corr, cross_road_tot = 0, 0
            norm_corr, norm_tot = 0, 0
            da_corr, da_tot = 0, 0
            c_accs = []

            for cid in sorted(clients.keys()):
                idx = clients[cid][split_key]
                model = client_models[cid] if mode == "deployed" else global_model
                yp = predict_model(model, X_all[idx], device)
                yt = spec.y[idx]
                tr = spec.trip[idx]
                y_true_all.append(yt)
                y_pred_all.append(yp)
                trip_all.append(tr)
                c_acc = trip_vote_accuracy(yt, yp, tr)
                c_accs.append(c_acc)

                if split_key == "s2" and mode == "deployed":
                    per_client_s2_dep[cid] = c_acc
                    tr_idx = clients[cid]["train"]
                    local_train_classes = set(np.unique(spec.y[tr_idx]).tolist())
                    local_cls_roads = set()
                    if spec.key == "UAH":
                        for tr_u in np.unique(spec.trip[tr_idx]):
                            tm_tr = tr_idx[spec.trip[tr_idx] == tr_u]
                            lab_tr = int(np.bincount(spec.y[tm_tr]).argmax())
                            local_cls_roads.add((lab_tr, _parse_uah_road(tr_u)))

                    c_seen_corr, c_seen_tot = 0, 0
                    for u_tr in np.unique(tr):
                        tm = tr == u_tr
                        t_lab = int(np.bincount(yt[tm]).argmax())
                        p_lab = int(np.bincount(yp[tm]).argmax())
                        ok = int(t_lab == p_lab)
                        if t_lab in local_train_classes:
                            seen_correct += ok
                            seen_total += 1
                            c_seen_corr += ok
                            c_seen_tot += 1
                        else:
                            unseen_correct += ok
                            unseen_total += 1
                        if spec.key == "UAH":
                            rd = _parse_uah_road(u_tr)
                            if (t_lab, rd) in local_cls_roads:
                                same_road_corr += ok
                                same_road_tot += 1
                            else:
                                cross_road_corr += ok
                                cross_road_tot += 1
                            if t_lab == 0:
                                norm_corr += ok
                                norm_tot += 1
                            else:
                                da_corr += ok
                                da_tot += 1
                    per_client_s2_seen[cid] = float(c_seen_corr / c_seen_tot) if c_seen_tot > 0 else float("nan")

            yt_cat = np.concatenate(y_true_all)
            yp_cat = np.concatenate(y_pred_all)
            tr_cat = np.concatenate(trip_all)
            res[f"{split_key}_win_f1_{mode}"] = fixed_macro_f1(yt_cat, yp_cat, spec.n_classes)
            res[f"{split_key}_trip_acc_{mode}"] = trip_vote_accuracy(yt_cat, yp_cat, tr_cat)
            res[f"{split_key}_client_mean_trip_acc_{mode}"] = float(np.mean(c_accs)) if c_accs else float("nan")

            if split_key == "s2" and mode == "deployed":
                tot_trips = seen_total + unseen_total
                res["s2_zero_support_trip_frac"] = float(unseen_total / tot_trips) if tot_trips > 0 else 0.0
                res["s2_seen_cls_trip_count"] = float(seen_total)
                res["s2_total_trip_count"] = float(tot_trips)
                res["s2_trip_acc_dep_seen_cls"] = float(seen_correct / seen_total) if seen_total > 0 else float("nan")
                res["s2_trip_acc_dep_unseen_cls"] = float(unseen_correct / unseen_total) if unseen_total > 0 else float("nan")
                if spec.key == "UAH":
                    res["s2_unseen_cls_road_trip_frac"] = float(cross_road_tot / tot_trips) if tot_trips > 0 else float("nan")
                    res["s2_trip_acc_dep_same_road"] = float(same_road_corr / same_road_tot) if same_road_tot > 0 else float("nan")
                    res["s2_trip_acc_dep_cross_road"] = float(cross_road_corr / cross_road_tot) if cross_road_tot > 0 else float("nan")
                    res["s2_trip_acc_dep_normal"] = float(norm_corr / norm_tot) if norm_tot > 0 else float("nan")
                    res["s2_trip_acc_dep_drowsy_agg"] = float(da_corr / da_tot) if da_tot > 0 else float("nan")

    # Zero-overhead support-guarded FedPer mitigation (support-masked + naive comparison)
    if method == "FedPer":
        for key_name, masked_flag in [("s2_trip_acc_guarded", True), ("s2_trip_acc_guarded_naive", False)]:
            guarded = build_support_guarded_fedper(
                global_model, client_models, spec, clients, support_masked=masked_flag
            )
            yt_all, yp_all, tr_all = [], [], []
            for cid in sorted(clients.keys()):
                idx = clients[cid]["s2"]
                yt_all.append(spec.y[idx])
                yp_all.append(predict_model(guarded[cid], X_all[idx], device))
                tr_all.append(spec.trip[idx])
            res[key_name] = trip_vote_accuracy(
                np.concatenate(yt_all), np.concatenate(yp_all), np.concatenate(tr_all)
            )
    else:
        res["s2_trip_acc_guarded"] = float("nan")
        res["s2_trip_acc_guarded_naive"] = float("nan")

    return res, per_client_s2_dep, per_client_s2_seen


def standardize_train_only(X: np.ndarray, clients: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    tr_idx = np.concatenate([c["train"] for c in clients.values()])
    X_scaled, _, _ = train_only_standardize(X, tr_idx, axis=(0, 2), eps=1e-6)
    return X_scaled


def _run_seed_worker(args_tuple):
    dkey, seed, methods, cfg, threads_per_worker = args_tuple
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(threads_per_worker)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, int(threads_per_worker)))
    torch.set_num_threads(max(1, int(threads_per_worker)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {"UAH": load_uah, "HAR": load_har, "WISDM": load_wisdm}
    spec = loaders[dkey]()
    drop_ch = cfg.get("drop_channels", [])
    if drop_ch:
        keep_ch = [c for c in range(spec.X.shape[1]) if c not in drop_ch]
        spec.X = np.ascontiguousarray(spec.X[:, keep_ch, :])
    clients = build_client_splits(
        spec, seed, cfg["s2_session_frac"], cfg["s1_window_frac"], split_mode=cfg["split_mode"]
    )
    X_use = standardize_train_only(spec.X, clients) if spec.standardize else spec.X

    seed_rows = []
    per_method_c = {}
    per_method_c_seen = {}
    logs = []
    for m in methods:
        g_model, c_models = run_fl_method(m, spec, X_use, clients, cfg, seed, device)
        metrics, per_c, per_c_seen = evaluate_protocol(m, spec, X_use, clients, g_model, c_models, device)
        row = {
            "dataset": dkey,
            "split_mode": cfg["split_mode"],
            "backbone": cfg["backbone"],
            "ditto_lam": cfg["ditto_lam"],
            "pfedme_lam": cfg["pfedme_lam"],
            "pfedme_K": cfg["pfedme_K"],
            "drop_channels": ",".join(str(c) for c in drop_ch),
            "n_clients": len(clients),
            "seed": seed,
            "method": m,
        }
        row.update(metrics)
        seed_rows.append(row)
        per_method_c[m] = per_c
        per_method_c_seen[m] = per_c_seen
        logs.append(
            f"  [{dkey} {cfg['split_mode'][:5]} s={seed}] {m:7s} | "
            f"S1t={metrics['s1_trip_acc_deployed']:.4f} | "
            f"S2t(dep/uni/seen/guard)={metrics['s2_trip_acc_deployed']:.4f}/"
            f"{metrics['s2_trip_acc_uniform']:.4f}/"
            f"{metrics['s2_trip_acc_dep_seen_cls']:.4f}/"
            f"{metrics['s2_trip_acc_guarded']:.4f} | "
            f"zero_sup={metrics['s2_zero_support_trip_frac']:.3f}"
        )
    return seed, seed_rows, per_method_c, per_method_c_seen, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["UAH", "HAR", "WISDM"], choices=["UAH", "HAR", "WISDM"])
    parser.add_argument("--methods", nargs="+", default=["FedAvg", "FedProx", "FedPer", "pFedMe", "Ditto"])
    parser.add_argument("--split-mode", choices=["unstratified", "stratified"], default="unstratified")
    parser.add_argument("--backbone", choices=["cnn", "gru"], default="cnn")
    parser.add_argument("--ditto-lam", type=float, default=DEFAULT_CONFIG["ditto_lam"])
    parser.add_argument("--pfedme-lam", type=float, default=DEFAULT_CONFIG["pfedme_lam"])
    parser.add_argument("--pfedme-K", type=int, default=DEFAULT_CONFIG["pfedme_K"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--drop-channels", nargs="*", type=int, default=[])
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_CONFIG["local_epochs"])
    parser.add_argument("--fedper-final-sync", choices=["overwrite", "none", "overwrite_ft1"], default="overwrite")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_CONFIG["seeds"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "phase3_federated" / "custom"))
    args = parser.parse_args()

    n_cpus = os.cpu_count() or 8
    workers = max(1, min(int(args.workers), len(args.seeds)))
    # On small 1D-CNN per-client batches, 1 thread per worker eliminates OpenMP barrier thrashing
    threads_per_worker = 1 if workers > 1 else max(1, min(4, n_cpus))
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)
    cfg = dict(DEFAULT_CONFIG)
    cfg["rounds"] = args.rounds
    cfg["local_epochs"] = args.local_epochs
    cfg["seeds"] = args.seeds
    cfg["split_mode"] = args.split_mode
    cfg["backbone"] = args.backbone
    cfg["ditto_lam"] = args.ditto_lam
    cfg["pfedme_lam"] = args.pfedme_lam
    cfg["pfedme_K"] = args.pfedme_K
    cfg["lr"] = args.lr
    cfg["pfedme_lr_in"] = args.lr
    cfg["drop_channels"] = list(args.drop_channels)
    cfg["fedper_final_sync"] = args.fedper_final_sync

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fl_protocol_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device} | workers={workers} (threads={threads_per_worker}) | "
        f"split={cfg['split_mode']} | backbone={cfg['backbone']} | "
        f"ditto_lam={cfg['ditto_lam']} | datasets={args.datasets} | seeds={cfg['seeds']}",
        flush=True,
    )

    loaders = {"UAH": load_uah, "HAR": load_har, "WISDM": load_wisdm}
    methods = list(args.methods)

    seed_rows = []
    client_test_rows = []

    for dkey in args.datasets:
        t0 = time.time()
        spec = loaders[dkey]()
        print(f"\n=== Dataset: {spec.display} (N={len(spec.y)}, clients={len(np.unique(spec.driver))}) ===", flush=True)

        client_scores: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
        client_scores_seen: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}

        if workers > 1:
            tasks = [(dkey, seed, methods, cfg, threads_per_worker) for seed in cfg["seeds"]]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_run_seed_worker, tasks))
            results.sort(key=lambda x: x[0])
            for _seed, s_rows, pm_c, pm_c_seen, logs in results:
                for line in logs:
                    print(line, flush=True)
                seed_rows.extend(s_rows)
                for m in methods:
                    for cid, val in pm_c[m].items():
                        client_scores[m].setdefault(cid, []).append(val)
                    for cid, val in pm_c_seen[m].items():
                        if np.isfinite(val):
                            client_scores_seen[m].setdefault(cid, []).append(val)
        else:
            for seed in cfg["seeds"]:
                _seed, s_rows, pm_c, pm_c_seen, logs = _run_seed_worker(
                    (dkey, seed, methods, cfg, threads_per_worker)
                )
                for line in logs:
                    print(line, flush=True)
                seed_rows.extend(s_rows)
                for m in methods:
                    for cid, val in pm_c[m].items():
                        client_scores[m].setdefault(cid, []).append(val)
                    for cid, val in pm_c_seen[m].items():
                        if np.isfinite(val):
                            client_scores_seen[m].setdefault(cid, []).append(val)

        if "FedAvg" in methods:
            common_cids = sorted(set(client_scores["FedAvg"].keys()))
            fedavg_vec = np.array([np.mean(client_scores["FedAvg"][c]) for c in common_cids])
            alt_methods = [m for m in methods if m != "FedAvg"]
            raw_p_map = {}
            tmp = []
            for m in alt_methods:
                m_vec = np.array([np.mean(client_scores[m][c]) for c in common_cids])
                diff = m_vec - fedavg_vec
                sig = exact_paired_sign_flip_test(diff, max_exact_n=18, n_perm=200000, seed=42)
                raw_p_map[m] = sig["p_value"]
                # Also seen-class-only client diff
                seen_cids = [c for c in common_cids if c in client_scores_seen[m] and c in client_scores_seen["FedAvg"]]
                if seen_cids:
                    d_seen = np.array([np.mean(client_scores_seen[m][c]) - np.mean(client_scores_seen["FedAvg"][c]) for c in seen_cids])
                    sig_seen = exact_paired_sign_flip_test(d_seen, max_exact_n=18, n_perm=200000, seed=42)
                    md_seen, pv_seen = float(np.mean(d_seen)), float(sig_seen["p_value"])
                else:
                    md_seen, pv_seen = float("nan"), float("nan")
                tmp.append((m, float(np.mean(diff)), float(np.std(diff, ddof=1)), float(sig["p_value"]), md_seen, pv_seen))
            adj_map = holm_bonferroni(raw_p_map) if raw_p_map else {}
            for (m, md, sd, pv, md_seen, pv_seen) in tmp:
                apv = adj_map.get(m, float("nan"))
                client_test_rows.append({
                    "dataset": dkey,
                    "split_mode": cfg["split_mode"],
                    "backbone": cfg["backbone"],
                    "ditto_lam": cfg["ditto_lam"],
                    "n_clients": len(common_cids),
                    "method": m,
                    "mean_diff_vs_FedAvg_s2_trip_dep": md,
                    "sd_diff": sd,
                    "exact_p": pv,
                    "holm_bonferroni_p": apv,
                    "holm_alpha_005_pass": bool(apv < 0.05),
                    "mean_diff_seen_cls_only": md_seen,
                    "exact_p_seen_cls_only": pv_seen,
                })
        print(f"Finished {dkey} in {time.time() - t0:.1f}s.", flush=True)

    df_seed = pd.DataFrame(seed_rows)
    df_seed.to_csv(out_dir / "fl_protocol_per_seed.csv", index=False)

    metric_cols = [c for c in df_seed.columns if c.startswith(("s1_", "s2_"))]
    agg_rows = []
    for (dkey, m), grp in df_seed.groupby(["dataset", "method"], sort=False):
        r = {
            "dataset": dkey,
            "split_mode": cfg["split_mode"],
            "backbone": cfg["backbone"],
            "ditto_lam": cfg["ditto_lam"],
            "n_clients": int(grp["n_clients"].iloc[0]),
            "method": m,
        }
        for col in metric_cols:
            vals = grp[col].dropna()
            r[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            r[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(r)
    pd.DataFrame(agg_rows).to_csv(out_dir / "fl_protocol_summary_mean_sd.csv", index=False)
    if client_test_rows:
        pd.DataFrame(client_test_rows).to_csv(out_dir / "fl_protocol_client_tests.csv", index=False)
    print(f"\nAll CSV outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
