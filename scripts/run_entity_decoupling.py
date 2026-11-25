#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entity-aware split decoupling experiment (S1 / S1_matched / S2_matched / S2).

Datasets
--------
- UAH-DriveSet : 7 channels x 300 samples (30 s @ 10 Hz), entity = trip
                 (whole-trip holdout stratified by driver, mirroring the
                 survey's Box-1 protocol with 42.5% trips held out)
- UCI HAR      : 9 channels x 128 samples, entity = subject
- WISDM 1.1    : 3 channels x 45 samples re-windowed from the raw phone
                 accelerometer files, entity = subject

Design
------
For every dataset and seed, the four arms share one training set and one
1D-CNN model:

  S1          : window-random 80/20 split inside the S2 training pool (leaky)
  S1_matched  : S1 test subsampled to n = |S2_matched test|
  S2_matched  : entity-held-out test subsampled to the same n
  S2          : full entity-held-out test set

The same model is scored on all four test sets, so differences between
S1_matched and S2_matched cannot be attributed to training data or sample
size.  Leakage audits, unit-level metrics, exact paired sign-flip tests and
ICC(1)/design-effect/effective-sample-size sweeps are written as CSV.

Run
---
    python run_entity_decoupling.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fl_entity_eval import (  # noqa: E402
    EntityGroupSplitter,
    ICCDiagnostics,
    LeakageAuditor,
    UnitLevelEvaluator,
    exact_paired_sign_flip_test,
    fixed_macro_f1,
    holm_bonferroni,
    present_class_macro_f1,
    train_only_standardize,
)

RESEARCH = Path(os.environ.get("FL_ENTITY_EVAL_DATA_ROOT", str(REPO_ROOT / "datasets")))
PROCESSED = RESEARCH / "data" / "processed"
RAW_UAH_NPZ = Path(os.environ.get("FL_ENTITY_EVAL_RAW_UAH", str(REPO_ROOT / "datasets" / "uah_raw")))
WISDM_ZIP = RESEARCH / "data" / "raw" / "WISDM" / "wisdm-dataset.zip"
OUT = REPO_ROOT / "results" / "phase2_decoupling"

SEEDS = [1, 2, 3, 4, 5, 6]
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
OVERLAPS = [0.0, 0.5, 0.75]
WISDM_LABELS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@dataclass
class DatasetSpec:
    key: str
    display: str
    X: np.ndarray              # (N, channels, length), float32
    y: np.ndarray              # (N,)
    driver: np.ndarray         # (N,) strings
    trip: np.ndarray           # (N,) strings; session / segment for audit & voting
    time: np.ndarray           # (N,) seconds inside the recording unit
    entity: np.ndarray         # (N,) split entity (trip for UAH, subject otherwise)
    stratify: Optional[np.ndarray]
    entity_col: str
    trip_col: str
    test_frac: float
    n_classes: int
    native_len: int
    fs: float
    entity_unit: str
    standardize: bool
    icc_long: pd.DataFrame
    icc_lengths: List[int]
    fs_note: str = "native"
    icc_group_unit: str = "recording"


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_uah() -> DatasetSpec:
    # Load genuine RAW physical-unit UAH windows (no pre-split z-scoring!)
    d = np.load(RAW_UAH_NPZ / "windows_L30_raw.npz", allow_pickle=False)
    X = np.transpose(d["X"], (0, 2, 1)).astype(np.float32)   # (N, 7, 300)
    y = d["y"].astype(np.int64)
    driver = d["driver"].astype(str)
    trip = d["trip"].astype(str)
    t0 = d["t0"].astype(np.float64)

    # Continuous RAW physical-unit accelerometer magnitude (in g) per trip,
    # rebuilt from the L10 raw windows (100 samples, stride 50 = 5 s).
    l10 = np.load(RAW_UAH_NPZ / "windows_L10_raw.npz", allow_pickle=False)
    X10 = l10["X"].astype(np.float32)
    trip10 = l10["trip"].astype(str)
    t10 = l10["t0"].astype(np.float64)
    frames = []
    for tr in np.unique(trip10):
        m = trip10 == tr
        order = np.argsort(t10[m], kind="stable")
        ts = t10[m][order]
        xs = X10[m][order]
        cuts = [0] + [i + 1 for i in np.where(np.diff(ts) > 5.5)[0]] + [len(ts)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            parts = [xs[a]] + [xs[j, 50:, :] for j in range(a + 1, b)]
            arr = np.concatenate(parts, axis=0)
            # channels: 0 speed (km/h), 1..3 raw linear acceleration ax, ay, az (g)
            mag = np.linalg.norm(arr[:, 1:4], axis=1).astype(np.float32)
            frames.append(pd.DataFrame({"mag": mag, "group": tr}))
    icc_long = pd.concat(frames, ignore_index=True)

    return DatasetSpec(
        key="UAH", display="UAH-DriveSet", X=X, y=y, driver=driver, trip=trip,
        time=t0, entity=trip, stratify=driver, entity_col="entity",
        trip_col="trip", test_frac=0.425, n_classes=3, native_len=300,
        fs=10.0, entity_unit="trip", standardize=True,
        icc_long=icc_long, icc_lengths=[50, 100, 300], icc_group_unit="trip",
    )


def _har_sessions(subject: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Session = contiguous run of one activity inside one subject."""
    trip = np.empty(len(y), dtype=object)
    t = np.zeros(len(y), dtype=np.float64)
    for s in np.unique(subject):
        idx = np.where(subject == s)[0]
        ys = y[idx]
        st = 0
        k = 0
        for i in range(1, len(ys) + 1):
            if i == len(ys) or ys[i] != ys[st]:
                blk = idx[st:i]
                trip[blk] = f"S{int(s)}_{k}"
                t[blk] = np.arange(len(blk), dtype=np.float64) * 1.28
                k += 1
                st = i
    return trip.astype(str), t


def load_har() -> DatasetSpec:
    u = np.load(PROCESSED / "uci_har_windows.npz", allow_pickle=False)
    X = np.concatenate([u["X_train"], u["X_test"]]).astype(np.float32)
    y = np.concatenate([u["y_train"], u["y_test"]]).astype(np.int64)
    subject = np.concatenate([u["client_train"], u["client_test"]]).astype(np.int64)
    subject_str = np.array([f"S{int(s)}" for s in subject])
    trip, t = _har_sessions(subject, y)

    # Continuous body-acceleration magnitude per subject, rebuilt from the
    # 128-sample windows (stride 64 = 50% overlap) within each activity run.
    frames = []
    for s in np.unique(subject):
        idx = np.where(subject == s)[0]
        ys = y[idx]
        st = 0
        k = 0
        for i in range(1, len(ys) + 1):
            if i == len(ys) or ys[i] != ys[st]:
                blk = idx[st:i]
                parts = [X[blk[0]]] + [X[j, :, 64:] for j in blk[1:]]
                arr = np.concatenate(parts, axis=1)           # (9, L)
                mag = np.linalg.norm(arr[0:3, :], axis=0).astype(np.float32)
                frames.append(pd.DataFrame({"mag": mag, "group": f"S{int(s)}_{k}"}))
                k += 1
                st = i
    icc_long = pd.concat(frames, ignore_index=True)

    return DatasetSpec(
        key="HAR", display="UCI HAR", X=X, y=y, driver=subject_str, trip=trip,
        time=t, entity=subject_str, stratify=None, entity_col="entity",
        trip_col="trip", test_frac=0.33, n_classes=6, native_len=128,
        fs=50.0, entity_unit="subject", standardize=False,
        icc_long=icc_long, icc_lengths=[64, 128, 256],
        icc_group_unit="activity session",
    )


def load_wisdm() -> DatasetSpec:
    """Rebuild raw 3-axis windows from the WISDM 1.1 zip (no cache needed)."""
    if not WISDM_ZIP.exists():
        raise FileNotFoundError(WISDM_ZIP)
    zf = zipfile.ZipFile(WISDM_ZIP)
    files = [n for n in zf.namelist() if "/phone/accel/" in n and n.endswith(".txt")]
    files.sort(key=lambda s: int(re.search(r"data_(\d+)_", s).group(1)))

    X_parts: List[np.ndarray] = []
    y_parts: List[int] = []
    subj_parts: List[int] = []
    trip_parts: List[int] = []
    tstart_parts: List[float] = []
    mag_parts: List[np.ndarray] = []
    mag_groups: List[np.ndarray] = []
    seg_counter = 0
    W, stride = 45, 22

    for name in files:
        sid = int(re.search(r"data_(\d+)_", name).group(1))

        def flush(lab: Optional[str], buf: List[Tuple[float, float, float, float]]) -> None:
            nonlocal seg_counter
            if buf and lab in WISDM_LABELS and len(buf) >= W:
                arr = np.asarray(buf, dtype=np.float64)
                ts = arr[:, 0]
                gid = seg_counter
                for s in range(0, len(arr) - W + 1, stride):
                    X_parts.append(arr[s:s + W, 1:4].astype(np.float32))
                    y_parts.append(WISDM_LABELS[lab])
                    subj_parts.append(sid)
                    trip_parts.append(gid)
                    tstart_parts.append(float((ts[s] - ts[0]) / 1e9))
                mag = np.linalg.norm(arr[:, 1:4], axis=1).astype(np.float32)
                mag_parts.append(mag)
                mag_groups.append(np.full(len(mag), gid, dtype=np.int32))
            seg_counter += 1

        cur_lab: Optional[str] = None
        buf: List[Tuple[float, float, float, float]] = []
        with zf.open(name) as f:
            for raw in f:
                parts = raw.decode().strip().rstrip(";").split(",")
                if len(parts) < 6:
                    continue
                lab = parts[1]
                if lab not in WISDM_LABELS:
                    if buf:
                        flush(cur_lab, buf)
                    buf, cur_lab = [], None
                    continue
                if lab != cur_lab:
                    if buf:
                        flush(cur_lab, buf)
                    buf, cur_lab = [], lab
                buf.append((float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])))
        if buf:
            flush(cur_lab, buf)

    X = np.stack(X_parts).transpose(0, 2, 1).astype(np.float32)   # (N, 3, 45)
    y = np.asarray(y_parts, dtype=np.int64)
    subject = np.asarray(subj_parts, dtype=np.int64)
    trip = np.asarray(trip_parts, dtype=np.int64)
    t_start = np.asarray(tstart_parts, dtype=np.float64)
    subject_str = np.array([f"U{int(s)}" for s in subject])
    trip_str = np.array([f"G{int(t)}" for t in trip])

    mag = np.concatenate(mag_parts)
    grp = np.concatenate(mag_groups)
    icc_long = pd.DataFrame({"mag": mag, "group": grp})
    # WISDM 1.1 mixes ~20 Hz, ~25 Hz and ~50 Hz phone clocks across subjects;
    # use a nominal 20 Hz only for converting sample windows to a time axis.
    fs = 20.0

    return DatasetSpec(
        key="WISDM", display="WISDM 1.1", X=X, y=y, driver=subject_str,
        trip=trip_str, time=t_start, entity=subject_str, stratify=None,
        entity_col="entity", trip_col="trip", test_frac=0.33,
        n_classes=6, native_len=45, fs=fs, entity_unit="subject",
        standardize=True, icc_long=icc_long, icc_lengths=[15, 45, 90],
        fs_note="mixed 20/25/50 Hz clocks; nominal 20 Hz for seconds",
        icc_group_unit="activity segment",
    )


# ---------------------------------------------------------------------------
# 1D-CNN backbone and training
# ---------------------------------------------------------------------------

class Conv1DNet(nn.Module):
    """Two-conv backbone used in the review's FL code, length-agnostic."""

    def __init__(self, n_channels: int, n_classes: int, length: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (length // 4), 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def fixed_entity_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, drivers: np.ndarray,
    trips: np.ndarray, n_classes: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "window_macro_f1_fixed": fixed_macro_f1(y_true, y_pred, n_classes),
    }
    drv_f1 = [
        fixed_macro_f1(y_true[drivers == d], y_pred[drivers == d], n_classes)
        for d in np.unique(drivers)
    ]
    drv_pf1 = [
        present_class_macro_f1(y_true[drivers == d], y_pred[drivers == d])
        for d in np.unique(drivers)
    ]
    drv_acc = [
        float(np.mean(y_true[drivers == d] == y_pred[drivers == d]))
        for d in np.unique(drivers)
    ]
    out["driver_mean_macro_f1_fixed"] = float(np.mean(drv_f1))
    out["driver_worst_macro_f1_fixed"] = float(np.min(drv_f1))
    out["driver_mean_present_f1"] = float(np.mean(drv_pf1))
    out["driver_worst_present_f1"] = float(np.min(drv_pf1))
    out["driver_mean_window_acc"] = float(np.mean(drv_acc))
    out["driver_worst_window_acc"] = float(np.min(drv_acc))

    vote_true, vote_pred = [], []
    for t in np.unique(trips):
        m = trips == t
        vote_true.append(int(np.bincount(y_true[m], minlength=n_classes).argmax()))
        vote_pred.append(int(np.bincount(y_pred[m], minlength=n_classes).argmax()))
    vote_true = np.asarray(vote_true)
    vote_pred = np.asarray(vote_pred)
    out["trip_vote_acc_fixed"] = float(np.mean(vote_true == vote_pred))
    out["trip_vote_macro_f1_fixed"] = fixed_macro_f1(vote_true, vote_pred, n_classes)
    return out


def fit_model(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> Tuple[nn.Module, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Conv1DNet(X.shape[1], n_classes, X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    drop_last = (len(ds) % BATCH_SIZE == 1)
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    t0 = time.time()
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model, time.time() - t0


def predict(model: nn.Module, X: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch])).argmax(dim=1).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_dataset(spec: DatasetSpec) -> Tuple[List[dict], List[dict], List[dict]]:
    meta = pd.DataFrame({
        "entity": spec.entity,
        "driver": spec.driver,
        "trip": spec.trip,
        "t": spec.time,
    })
    result_rows: List[dict] = []
    audit_rows: List[dict] = []
    split_rows: List[dict] = []

    for seed in SEEDS:
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(
            meta, entity_col=spec.entity_col, test_frac=spec.test_frac,
            stratify_col=("driver" if spec.stratify is not None else None),
            seed=seed,
        )
        splits = EntityGroupSplitter.split_size_matched_pair(
            meta, trip_col=spec.trip_col, s2_train_idx=s2_tr,
            s2_test_idx=s2_te, s1_test_frac=0.2, seed=seed,
        )
        shared_train = splits["S1"][0]

        if spec.standardize:
            X_all, _, _ = train_only_standardize(spec.X, shared_train, axis=(0, 2), eps=1e-6)
        else:
            X_all = spec.X

        model, train_sec = fit_model(
            np.ascontiguousarray(X_all[shared_train]),
            spec.y[shared_train], spec.n_classes, seed,
        )

        native_window_sec = float(spec.native_len / spec.fs)
        for split_name, (tr, te) in splits.items():
            aud = LeakageAuditor.audit_split(
                meta, tr, te, driver_col="driver", trip_col="trip",
                time_col="t", window_sec=native_window_sec, inclusive=False,
            )
            audit_rows.append({
                "dataset": spec.key, "seed": seed, "split": split_name,
                **{k: v for k, v in aud.items()},
            })

            te_meta = meta.iloc[te]
            split_rows.append({
                "dataset": spec.key, "seed": seed, "split": split_name,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_test_drivers": int(te_meta["driver"].nunique()),
                "n_test_entities": int(te_meta["entity"].nunique()),
                "n_test_trips": int(te_meta["trip"].nunique()),
                "classes_present": int(len(np.unique(spec.y[te]))),
                "class_counts": "|".join(
                    str(int(v)) for v in np.bincount(spec.y[te], minlength=spec.n_classes)
                ),
            })

            pred = predict(model, np.ascontiguousarray(X_all[te]))
            ev = UnitLevelEvaluator.evaluate_predictions(
                spec.y[te], pred, spec.driver[te], spec.trip[te], n_classes=spec.n_classes,
            )
            fx = fixed_entity_metrics(
                spec.y[te], pred, spec.driver[te], spec.trip[te], spec.n_classes,
            )
            err_icc = ICCDiagnostics.compute_error_deff(
                spec.y[te], pred, spec.trip[te]
            )
            train_entities = set(meta.iloc[shared_train]["entity"])
            entity_seen = bool(all(e in train_entities for e in te_meta["entity"]))
            result_rows.append({
                "dataset": spec.key, "seed": seed, "split": split_name,
                "entity_unit": spec.entity_unit,
                "entity_seen_in_train": entity_seen,
                "trip_independence": ("pseudo_independent" if entity_seen else "held_out"),
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_test_drivers": int(te_meta["driver"].nunique()),
                "n_test_trips": int(te_meta["trip"].nunique()),
                "classes_present": int(len(np.unique(spec.y[te]))),
                "train_time_s": round(train_sec, 2),
                "err_icc": float(err_icc["icc"]),
                "err_deff": float(err_icc["deff"]),
                "err_neff": float(err_icc["n_eff"]),
                "err_neff_ratio": float(err_icc["neff_ratio"]),
                "err_deff_unbalanced": float(err_icc["deff_unbalanced"]),
                **{k: (float(v) if isinstance(v, (int, np.integer, float, np.floating)) else v)
                   for k, v in ev.items()},
                **fx,
            })
        print(f"  [{spec.key}] seed {seed} done "
              f"(train={len(shared_train)}, test S1={len(splits['S1'][1])}, "
              f"S2={len(splits['S2'][1])}, {train_sec:.1f}s)", flush=True)

    return result_rows, audit_rows, split_rows


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    metrics = ["window_acc", "window_macro_f1_fixed",
               "driver_mean_macro_f1_fixed", "driver_worst_macro_f1_fixed",
               "driver_mean_present_f1", "driver_worst_present_f1",
               "driver_mean_window_acc", "driver_worst_window_acc",
               "trip_vote_acc_fixed", "trip_vote_macro_f1_fixed"]
    rows = []
    for (ds, split), g in res.groupby(["dataset", "split"], sort=True):
        for m in metrics:
            v = g[m].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            rows.append({
                "dataset": ds, "split": split, "metric": m, "n_seeds": len(v),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()),
            })
    return pd.DataFrame(rows)


def paired_tests(res: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("S1_matched", "S2_matched"),
        ("S2_matched", "S2"),
        ("S1", "S1_matched"),
        ("S1", "S2"),
    ]
    metrics = ["window_acc", "window_macro_f1_fixed",
               "driver_mean_macro_f1_fixed", "driver_worst_macro_f1_fixed",
               "driver_mean_present_f1", "driver_worst_present_f1",
               "driver_mean_window_acc", "driver_worst_window_acc",
               "trip_vote_acc_fixed", "trip_vote_macro_f1_fixed"]
    rows = []
    for ds in sorted(res["dataset"].unique()):
        sub = res[res["dataset"] == ds]
        for m in metrics:
            pvals: Dict[str, float] = {}
            tmp = []
            for a, b in comparisons:
                va = sub[sub["split"] == a].sort_values("seed")[m].to_numpy(dtype=float)
                vb = sub[sub["split"] == b].sort_values("seed")[m].to_numpy(dtype=float)
                if len(va) == 0 or len(va) != len(vb):
                    continue
                diffs = va - vb
                if np.all(~np.isfinite(diffs)):
                    continue
                sig = exact_paired_sign_flip_test(diffs)
                key = f"{a}_minus_{b}"
                pvals[key] = sig["p_value"]
                tmp.append((key, a, b, diffs, sig))
            adj = holm_bonferroni(pvals) if pvals else {}
            for key, a, b, diffs, sig in tmp:
                rows.append({
                    "dataset": ds, "metric": m, "comparison": key,
                    "direction": f"{a} - {b}", "n_seeds": len(diffs),
                    "mean_diff": float(np.mean(diffs)),
                    "sd_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                    "wins": int(np.sum(diffs > 0)),
                    "losses": int(np.sum(diffs < 0)),
                    "exact_p": sig["p_value"],
                    "min_achievable_p": sig["min_achievable_p"],
                    "holm_p_within_dataset_metric": adj.get(key, np.nan),
                })
    return pd.DataFrame(rows)


def icc_tables(specs: List[DatasetSpec]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sweep_rows = []
    native_rows = []
    for spec in specs:
        sweep = ICCDiagnostics.sweep_window_overlap(
            spec.icc_long, "mag", "group",
            window_lengths=spec.icc_lengths, overlap_ratios=OVERLAPS,
        )
        sweep["window_len_sec"] = sweep["window_len"] / spec.fs
        sweep["dataset"] = spec.key
        sweep["signal"] = "accel_magnitude"
        sweep["group_unit"] = spec.icc_group_unit
        sweep_rows.append(sweep)
        # Also compute class-residualized ICC on native classification windows
        if spec.key == "UAH":
            win_mag = np.linalg.norm(spec.X[:, 1:4, :], axis=1).mean(axis=1)
        else:
            win_mag = np.linalg.norm(spec.X[:, 0:3, :], axis=1).mean(axis=1)
        cr = ICCDiagnostics.compute_class_residualized_icc(win_mag, spec.trip, spec.y)
        nat = sweep[(sweep["window_len"] == spec.native_len) &
                    (np.isclose(sweep["overlap_ratio"], 0.5))]
        for _, r in nat.iterrows():
            native_rows.append({
                "dataset": spec.key, "signal": "accel_magnitude",
                "group_unit": spec.icc_group_unit,
                "window_len_samples": int(r["window_len"]),
                "window_len_sec": float(r["window_len_sec"]),
                "overlap_ratio": float(r["overlap_ratio"]),
                "n_obs": float(r["n_obs"]), "n_groups": float(r["n_groups"]),
                "mean_cluster_size": float(r["mean_cluster_size"]),
                "cv_cluster_size": float(r["cv_cluster_size"]),
                "icc": float(r["icc"]), "deff": float(r["deff"]),
                "n_eff": float(r["n_eff"]), "neff_ratio": float(r["neff_ratio"]),
                "deff_unbalanced": float(r["deff_unbalanced"]),
                "n_eff_unbalanced": float(r["n_eff_unbalanced"]),
                "neff_ratio_unbalanced": float(r["neff_ratio_unbalanced"]),
                "class_resid_icc": float(cr["icc"]),
                "class_resid_deff": float(cr["deff"]),
                "class_resid_neff": float(cr["n_eff"]),
                "class_resid_neff_ratio": float(cr["neff_ratio"]),
            })
    sweep_df = pd.concat(sweep_rows, ignore_index=True)
    cols = ["dataset", "signal", "group_unit", "window_len", "window_len_sec",
            "overlap_ratio", "n_obs", "n_groups", "mean_cluster_size", "cv_cluster_size",
            "icc", "deff", "n_eff", "neff_ratio", "deff_unbalanced", "n_eff_unbalanced",
            "neff_ratio_unbalanced"]
    return sweep_df[cols], pd.DataFrame(native_rows)


def dataset_summary(specs: List[DatasetSpec]) -> pd.DataFrame:
    rows = []
    for s in specs:
        rows.append({
            "dataset": s.key, "display_name": s.display,
            "n_windows": int(len(s.y)),
            "n_channels": int(s.X.shape[1]),
            "window_len_samples": int(s.X.shape[2]),
            "window_len_sec": round(s.X.shape[2] / s.fs, 3),
            "sampling_hz": round(s.fs, 3),
            "sampling_hz_note": s.fs_note,
            "n_classes": s.n_classes,
            "class_counts": "|".join(str(int(v)) for v in np.bincount(s.y, minlength=s.n_classes)),
            "n_drivers": int(len(np.unique(s.driver))),
            "entity_unit": s.entity_unit,
            "n_entities": int(len(np.unique(s.entity))),
            "n_recordings": int(len(np.unique(s.trip))),
            "test_frac": s.test_frac,
            "standardize_on_train": s.standardize,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    print("torch threads:", torch.get_num_threads(), flush=True)

    print("loading UAH ...", flush=True)
    specs = [load_uah()]
    print("loading HAR ...", flush=True)
    specs.append(load_har())
    print("loading WISDM raw zip (this takes a minute) ...", flush=True)
    specs.append(load_wisdm())
    dataset_summary(specs).to_csv(OUT / "dataset_summary.csv", index=False,
                                  encoding="utf-8-sig")

    all_res, all_audit, all_splits = [], [], []
    for spec in specs:
        print(f"running {spec.display} ...", flush=True)
        res, audit, splits = run_dataset(spec)
        all_res += res
        all_audit += audit
        all_splits += splits

    res_df = pd.DataFrame(all_res)
    audit_df = pd.DataFrame(all_audit)
    split_df = pd.DataFrame(all_splits)

    res_df.to_csv(OUT / "decoupling_results.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OUT / "leakage_audit_summary.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(OUT / "split_summary.csv", index=False, encoding="utf-8-sig")
    summarize(res_df).to_csv(OUT / "decoupling_summary_mean_sd.csv", index=False,
                             encoding="utf-8-sig")
    paired_tests(res_df).to_csv(OUT / "paired_tests.csv", index=False,
                                encoding="utf-8-sig")

    print("computing ICC / design-effect sweeps ...", flush=True)
    sweep_df, native_df = icc_tables(specs)
    sweep_df.to_csv(OUT / "icc_neff_decay_curve.csv", index=False, encoding="utf-8-sig")
    native_df.to_csv(OUT / "icc_native_window_summary.csv", index=False,
                     encoding="utf-8-sig")

    print("\n=== files ===")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.name)
    print("\n=== native-window ICC ===")
    if len(native_df):
        print(native_df.round(4).to_string(index=False))
    print("\n=== leakage verdicts (unique) ===")
    print(audit_df.groupby(["dataset", "split"])["audit_verdict"]
          .agg(lambda s: ",".join(sorted(set(s)))).to_string())
    print("\ndone ->", OUT)


if __name__ == "__main__":
    main()
