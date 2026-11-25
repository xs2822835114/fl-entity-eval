#!/usr/bin/env python3
"""Sensitivity checks for the entity-aware evaluation benchmark:
1. Second time-series backbone (GRU1DNet, 25 epochs, 3 seeds) decoupling on raw-unit UAH & UCI HAR.
2. WISDM 1.1 uniform 20 Hz timestamp resampling (3 seeds, 30 epochs) + explicit LeakageAuditor
   verification under both uniform 20 Hz (window_sec=2.25 s) and per-segment true sampling rate.
3. 5-fold subject-grouped CV (25 epochs, reporting both present-class and fixed macro-F1) on
   UCI HAR (n=30) & WISDM uniform 20 Hz (n=51).
4. Leave-one-driver-out (LODO, n=6) unseen-driver vs. within-driver S1 check on raw-unit UAH.
"""
from __future__ import annotations

import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_entity_eval import (  # noqa: E402
    EntityGroupSplitter,
    LeakageAuditor,
    exact_paired_sign_flip_test,
    fixed_macro_f1,
    present_class_macro_f1,
    train_only_standardize,
)
from run_entity_decoupling import (  # noqa: E402
    WISDM_LABELS,
    WISDM_ZIP,
    Conv1DNet,
    DatasetSpec,
    fixed_entity_metrics,
    load_har,
    load_uah,
    predict,
)


class GRU1DNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, length: int) -> None:
        super().__init__()
        self.stem = nn.AvgPool1d(kernel_size=4, stride=4)
        self.gru = nn.GRU(n_channels, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.stem(x).transpose(1, 2))
        return self.head(out.mean(dim=1))


def fit_backbone(model: nn.Module, X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 30) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=(len(ds) % 64 == 1),
                    generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def load_wisdm_uniform_20hz() -> DatasetSpec:
    """Resample every WISDM 1.1 subject stream to uniform 20 Hz via linear interpolation before windowing."""
    zf = zipfile.ZipFile(WISDM_ZIP)
    files = [n for n in zf.namelist() if "/phone/accel/" in n and n.endswith(".txt")]
    files.sort(key=lambda s: int(re.search(r"data_(\d+)_", s).group(1)))

    X_parts, y_parts, subj_parts, trip_parts, tstart_parts = [], [], [], [], []
    seg_counter = 0
    FS = 20.0
    W, stride = 45, 22

    for name in files:
        sid = int(re.search(r"data_(\d+)_", name).group(1))

        def flush(lab: Optional[str], buf: List[Tuple[float, float, float, float]]) -> None:
            nonlocal seg_counter
            if buf and lab in WISDM_LABELS and len(buf) >= 20:
                arr = np.asarray(buf, dtype=np.float64)
                order = np.argsort(arr[:, 0], kind="stable")
                arr = arr[order]
                t_sec = (arr[:, 0] - arr[0, 0]) / 1e9
                keep = np.concatenate([[True], np.diff(t_sec) > 1e-4])
                t_sec = t_sec[keep]
                xyz = arr[keep, 1:4]
                if len(t_sec) >= 20 and t_sec[-1] >= 2.25:
                    grid = np.arange(0.0, t_sec[-1], 1.0 / FS)
                    if len(grid) >= W:
                        resamp = np.stack([np.interp(grid, t_sec, xyz[:, c]) for c in range(3)], axis=1).astype(np.float32)
                        gid = seg_counter
                        for s in range(0, len(resamp) - W + 1, stride):
                            X_parts.append(resamp[s : s + W])
                            y_parts.append(WISDM_LABELS[lab])
                            subj_parts.append(sid)
                            trip_parts.append(gid)
                            tstart_parts.append(float(grid[s]))
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
                buf.append((float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
        if buf:
            flush(cur_lab, buf)

    X = np.stack(X_parts).transpose(0, 2, 1).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    subj_str = np.array([f"U{int(s)}" for s in subj_parts])
    trip_str = np.array([f"G{int(t)}" for t in trip_parts])
    return DatasetSpec(
        key="WISDM_20HZ", display="WISDM 1.1 (Uniform 20 Hz)", X=X, y=y, driver=subj_str,
        trip=trip_str, time=np.asarray(tstart_parts, dtype=np.float64), entity=subj_str,
        stratify=None, entity_col="entity", trip_col="trip", test_frac=0.33, n_classes=6,
        native_len=45, fs=20.0, entity_unit="subject", standardize=True,
        icc_long=pd.DataFrame(), icc_lengths=[45],
    )


def load_wisdm_true_rate_meta() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build native-rate WISDM 1.1 window metadata plus each segment's true median window duration."""
    zf = zipfile.ZipFile(WISDM_ZIP)
    files = [n for n in zf.namelist() if "/phone/accel/" in n and n.endswith(".txt")]
    files.sort(key=lambda s: int(re.search(r"data_(\d+)_", s).group(1)))

    subj_parts, trip_parts, tstart_parts = [], [], []
    seg_win_sec: Dict[str, float] = {}
    seg_counter = 0
    W, stride = 45, 22

    for name in files:
        sid = int(re.search(r"data_(\d+)_", name).group(1))

        def flush(lab: Optional[str], buf: List[float]) -> None:
            nonlocal seg_counter
            if buf and lab in WISDM_LABELS and len(buf) >= W:
                ts = np.sort(np.asarray(buf, dtype=np.float64))
                t_sec = (ts - ts[0]) / 1e9
                diffs = np.diff(t_sec)
                pos = diffs[diffs > 1e-4]
                dt_med = float(np.median(pos)) if len(pos) > 0 else 0.05
                gid = f"G{seg_counter}"
                seg_win_sec[gid] = float(W * dt_med)
                for s in range(0, len(t_sec) - W + 1, stride):
                    subj_parts.append(f"U{sid}")
                    trip_parts.append(gid)
                    tstart_parts.append(float(t_sec[s]))
            seg_counter += 1

        cur_lab: Optional[str] = None
        buf: List[float] = []
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
                buf.append(float(parts[2]))
        if buf:
            flush(cur_lab, buf)

    meta = pd.DataFrame({
        "entity": np.array(subj_parts),
        "driver": np.array(subj_parts),
        "trip": np.array(trip_parts),
        "t": np.asarray(tstart_parts, dtype=np.float64),
    })
    return meta, seg_win_sec


def audit_wisdm_temporal_overlap(wisdm20: DatasetSpec, seeds=(1, 2, 3, 4, 5, 6)) -> List[dict]:
    """Explicit LeakageAuditor verification under uniform 20 Hz and per-segment true sampling rate."""
    rows: List[dict] = []
    # (a) Uniform 20 Hz: window_sec = 45 / 20 = 2.25 s
    meta20 = pd.DataFrame({"entity": wisdm20.entity, "driver": wisdm20.driver, "trip": wisdm20.trip, "t": wisdm20.time})
    for seed in seeds:
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(meta20, "entity", test_frac=0.33, seed=seed)
        splits = EntityGroupSplitter.split_size_matched_pair(meta20, "trip", s2_tr, s2_te, s1_test_frac=0.2, seed=seed)
        aud = LeakageAuditor.audit_split(
            meta20, *splits["S1"], driver_col="driver", trip_col="trip", time_col="t", window_sec=2.25, inclusive=False
        )
        rows.append({
            "mode": "uniform_20hz",
            "seed": seed,
            "n_windows": len(meta20),
            "n_test_s1": int(len(splits["S1"][1])),
            "trip_overlap_rate": float(aud["trip_overlap_rate"]),
            "temporal_proximity_rate": float(aud["temporal_proximity_leak_rate"]),
            "temporal_proximity_hits": int(aud["temporal_proximity_hits"]),
        })

    # (b) Raw mixed-rate WISDM with per-segment true window_sec = 45 * median_dt
    meta_raw, seg_win_sec = load_wisdm_true_rate_meta()
    for seed in seeds:
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(meta_raw, "entity", test_frac=0.33, seed=seed)
        splits = EntityGroupSplitter.split_size_matched_pair(meta_raw, "trip", s2_tr, s2_te, s1_test_frac=0.2, seed=seed)
        tr_idx, te_idx = splits["S1"]
        tr_df = meta_raw.iloc[tr_idx]
        te_df = meta_raw.iloc[te_idx]
        tr_by_seg = {gid: np.sort(g["t"].to_numpy()) for gid, g in tr_df.groupby("trip")}
        hits = 0
        trip_hits = 0
        for gid, g_te in te_df.groupby("trip"):
            tr_t = tr_by_seg.get(gid)
            if tr_t is None or len(tr_t) == 0:
                continue
            trip_hits += len(g_te)
            wsec = seg_win_sec[gid]
            for ts in g_te["t"].to_numpy():
                pos = int(np.searchsorted(tr_t, ts))
                ok = False
                if pos < len(tr_t) and abs(tr_t[pos] - ts) < wsec - 1e-9:
                    ok = True
                if pos > 0 and abs(tr_t[pos - 1] - ts) < wsec - 1e-9:
                    ok = True
                if ok:
                    hits += 1
        rows.append({
            "mode": "native_per_segment_true_rate",
            "seed": seed,
            "n_windows": len(meta_raw),
            "n_test_s1": int(len(te_idx)),
            "trip_overlap_rate": float(trip_hits / len(te_idx)),
            "temporal_proximity_rate": float(hits / len(te_idx)),
            "temporal_proximity_hits": int(hits),
        })
    return rows


def run_gru_decoupling(spec: DatasetSpec, seeds=(1, 2, 3)) -> List[dict]:
    meta = pd.DataFrame({"entity": spec.entity, "driver": spec.driver, "trip": spec.trip, "t": spec.time})
    rows = []
    for seed in seeds:
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(
            meta, entity_col=spec.entity_col, test_frac=spec.test_frac,
            stratify_col=("driver" if spec.stratify is not None else None), seed=seed,
        )
        splits = EntityGroupSplitter.split_size_matched_pair(
            meta, trip_col=spec.trip_col, s2_train_idx=s2_tr, s2_test_idx=s2_te, s1_test_frac=0.2, seed=seed,
        )
        sh_tr = splits["S1"][0]
        if spec.standardize:
            X_all, _, _ = train_only_standardize(spec.X, sh_tr)
        else:
            X_all = spec.X
        model = fit_backbone(GRU1DNet(X_all.shape[1], spec.n_classes, X_all.shape[2]),
                             np.ascontiguousarray(X_all[sh_tr]), spec.y[sh_tr], seed=seed, epochs=25)
        for sp_name in ("S1_matched", "S2_matched", "S2"):
            _, te = splits[sp_name]
            yp = predict(model, np.ascontiguousarray(X_all[te]))
            fx = fixed_entity_metrics(spec.y[te], yp, spec.driver[te], spec.trip[te], spec.n_classes)
            rows.append({
                "dataset": spec.key, "backbone": "GRU1D", "seed": seed,
                "split": sp_name, "n_windows": int(len(spec.y)), **fx,
            })
    return rows


def run_wisdm20_decoupling(spec: DatasetSpec, seeds=(1, 2, 3)) -> List[dict]:
    meta = pd.DataFrame({"entity": spec.entity, "driver": spec.driver, "trip": spec.trip, "t": spec.time})
    rows = []
    for seed in seeds:
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(
            meta, entity_col=spec.entity_col, test_frac=spec.test_frac, stratify_col=None, seed=seed,
        )
        splits = EntityGroupSplitter.split_size_matched_pair(
            meta, trip_col=spec.trip_col, s2_train_idx=s2_tr, s2_test_idx=s2_te, s1_test_frac=0.2, seed=seed,
        )
        sh_tr = splits["S1"][0]
        X_all, _, _ = train_only_standardize(spec.X, sh_tr)
        model = fit_backbone(Conv1DNet(X_all.shape[1], spec.n_classes, X_all.shape[2]),
                             np.ascontiguousarray(X_all[sh_tr]), spec.y[sh_tr], seed=seed, epochs=30)
        for sp_name in ("S1_matched", "S2_matched", "S2"):
            _, te = splits[sp_name]
            yp = predict(model, np.ascontiguousarray(X_all[te]))
            fx = fixed_entity_metrics(spec.y[te], yp, spec.driver[te], spec.trip[te], spec.n_classes)
            rows.append({
                "dataset": spec.key, "backbone": "Conv1DNet", "seed": seed,
                "split": sp_name, "n_windows": int(len(spec.y)), **fx,
            })
    return rows


def run_subject_level_cv(spec: DatasetSpec, n_folds: int = 5, seed: int = 42) -> dict:
    """5-fold subject-grouped CV giving per-subject S1 vs S2 paired differences across ALL n subjects."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(np.unique(spec.driver)))
    rng.shuffle(subs)
    folds = np.array_split(subs, n_folds)

    s1_fix, s2_fix = {}, {}
    s1_pres, s2_pres = {}, {}
    for fi, test_subs in enumerate(folds):
        s2_te = np.where(np.isin(spec.driver, test_subs))[0]
        s2_tr = np.where(~np.isin(spec.driver, test_subs))[0]
        tr_list, s1_list = [], []
        for s in np.unique(spec.driver[s2_tr]):
            idx = s2_tr[spec.driver[s2_tr] == s]
            perm = rng.permutation(idx)
            k = max(1, int(round(len(idx) * 0.2)))
            s1_list.append(perm[:k])
            tr_list.append(perm[k:])
        tr_idx = np.concatenate(tr_list)
        s1_idx = np.concatenate(s1_list)

        if spec.standardize:
            X_all, _, _ = train_only_standardize(spec.X, tr_idx)
        else:
            X_all = spec.X

        model = fit_backbone(Conv1DNet(X_all.shape[1], spec.n_classes, X_all.shape[2]),
                             np.ascontiguousarray(X_all[tr_idx]), spec.y[tr_idx], seed=seed + fi, epochs=25)
        yp_s1 = predict(model, np.ascontiguousarray(X_all[s1_idx]))
        yp_s2 = predict(model, np.ascontiguousarray(X_all[s2_te]))

        for s in np.unique(spec.driver[s1_idx]):
            m = spec.driver[s1_idx] == s
            s1_fix.setdefault(str(s), []).append(fixed_macro_f1(spec.y[s1_idx][m], yp_s1[m], spec.n_classes))
            s1_pres.setdefault(str(s), []).append(present_class_macro_f1(spec.y[s1_idx][m], yp_s1[m]))
        for s in test_subs:
            m = spec.driver[s2_te] == s
            s2_fix[str(s)] = fixed_macro_f1(spec.y[s2_te][m], yp_s2[m], spec.n_classes)
            s2_pres[str(s)] = present_class_macro_f1(spec.y[s2_te][m], yp_s2[m])

    common = sorted(set(s1_fix.keys()).intersection(set(s2_fix.keys())))
    diffs_fix = np.array([float(np.mean(s1_fix[s])) - float(s2_fix[s]) for s in common])
    diffs_pres = np.array([float(np.mean(s1_pres[s])) - float(s2_pres[s]) for s in common])
    sig_fix = exact_paired_sign_flip_test(diffs_fix, max_exact_n=18, n_perm=200000, seed=42)
    sig_pres = exact_paired_sign_flip_test(diffs_pres, max_exact_n=18, n_perm=200000, seed=42)
    return {
        "dataset": spec.key,
        "n_subjects": len(common),
        "s1_mean_present": float(np.mean([np.mean(s1_pres[s]) for s in common])),
        "s2_mean_present": float(np.mean([s2_pres[s] for s in common])),
        "mean_diff_present": float(np.mean(diffs_pres)),
        "sd_diff_present": float(np.std(diffs_pres, ddof=1)),
        "exact_p_present": float(sig_pres["p_value"]),
        "s1_mean": float(np.mean([np.mean(s1_fix[s]) for s in common])),
        "s2_mean": float(np.mean([s2_fix[s] for s in common])),
        "mean_diff_s1_minus_s2": float(np.mean(diffs_fix)),
        "sd_diff": float(np.std(diffs_fix, ddof=1)),
        "exact_p_subject_level": float(sig_fix["p_value"]),
    }


def run_uah_lodo(spec: DatasetSpec, seed: int = 42) -> List[dict]:
    """Leave-one-driver-out (LODO, n=6) unseen-driver vs. within-driver S1 check on raw-unit UAH."""
    rng = np.random.default_rng(seed)
    drivers = sorted(np.unique(spec.driver))
    rows: List[dict] = []
    for di, test_drv in enumerate(drivers):
        s2_te = np.where(spec.driver == test_drv)[0]
        s2_tr = np.where(spec.driver != test_drv)[0]
        # Inside s2_tr, hold out 20% random windows per seen driver as leaky S1
        tr_list, s1_list = [], []
        for d in np.unique(spec.driver[s2_tr]):
            idx = s2_tr[spec.driver[s2_tr] == d]
            perm = rng.permutation(idx)
            k = max(1, int(round(len(idx) * 0.2)))
            s1_list.append(perm[:k])
            tr_list.append(perm[k:])
        tr_idx = np.concatenate(tr_list)
        s1_idx = np.concatenate(s1_list)

        X_all, _, _ = train_only_standardize(spec.X, tr_idx)
        model = fit_backbone(Conv1DNet(X_all.shape[1], spec.n_classes, X_all.shape[2]),
                             np.ascontiguousarray(X_all[tr_idx]), spec.y[tr_idx], seed=seed + di, epochs=30)
        yp_s1 = predict(model, np.ascontiguousarray(X_all[s1_idx]))
        yp_s2 = predict(model, np.ascontiguousarray(X_all[s2_te]))
        fx_s1 = fixed_entity_metrics(spec.y[s1_idx], yp_s1, spec.driver[s1_idx], spec.trip[s1_idx], spec.n_classes)
        fx_s2 = fixed_entity_metrics(spec.y[s2_te], yp_s2, spec.driver[s2_te], spec.trip[s2_te], spec.n_classes)
        rows.append({
            "held_out_driver": str(test_drv),
            "n_test_windows": int(len(s2_te)),
            "n_test_trips": int(len(np.unique(spec.trip[s2_te]))),
            "s1_window_macro_f1": fx_s1["window_macro_f1_fixed"],
            "lodo_window_macro_f1": fx_s2["window_macro_f1_fixed"],
            "lodo_present_f1": fx_s2["driver_mean_present_f1"],
            "s1_trip_vote_acc": fx_s1["trip_vote_acc_fixed"],
            "lodo_trip_vote_acc": fx_s2["trip_vote_acc_fixed"],
        })
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gru-lodo", action="store_true", help="Reuse freshly written gru_decoupling.csv and uah_lodo_unseen_driver.csv")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    out = REPO_ROOT / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    har = load_har()
    if not args.skip_gru_lodo:
        uah = load_uah()
        print("1) Running GRU1D second-backbone decoupling on raw-unit UAH & UCI HAR ...", flush=True)
        gru_rows = run_gru_decoupling(uah, seeds=(1, 2, 3)) + run_gru_decoupling(har, seeds=(1, 2, 3))
        pd.DataFrame(gru_rows).to_csv(out / "gru_decoupling.csv", index=False)

        print("2) Running UAH leave-one-driver-out (LODO, n=6) unseen-driver check ...", flush=True)
        lodo_rows = run_uah_lodo(uah, seed=42)
        pd.DataFrame(lodo_rows).to_csv(out / "uah_lodo_unseen_driver.csv", index=False)
        df_lodo = pd.DataFrame(lodo_rows)
        print(
            f"UAH LODO (n=6): S1 WinF1={df_lodo['s1_window_macro_f1'].mean():.4f} vs LODO WinF1={df_lodo['lodo_window_macro_f1'].mean():.4f} "
            f"| S1 TripAcc={df_lodo['s1_trip_vote_acc'].mean():.4f} vs LODO TripAcc={df_lodo['lodo_trip_vote_acc'].mean():.4f}",
            flush=True,
        )

    print("3) Running 5-fold subject-level paired CV on UCI HAR (n=30) ...", flush=True)
    har_cv = run_subject_level_cv(har, n_folds=5)
    print("HAR 5-fold subject-level CV:", har_cv, flush=True)

    print("4) Loading uniform-20Hz resampled WISDM 1.1 & running LeakageAuditor overlap check ...", flush=True)
    wisdm20 = load_wisdm_uniform_20hz()
    print(f"WISDM uniform 20Hz windows: N={len(wisdm20.y)}, subjects={len(np.unique(wisdm20.driver))}", flush=True)
    ov_rows = audit_wisdm_temporal_overlap(wisdm20, seeds=(1, 2, 3, 4, 5, 6))
    pd.DataFrame(ov_rows).to_csv(out / "wisdm_overlap_audit.csv", index=False)
    for mode, grp in pd.DataFrame(ov_rows).groupby("mode"):
        print(
            f"  WISDM overlap [{mode}]: rate={grp['temporal_proximity_rate'].min():.4f}--{grp['temporal_proximity_rate'].max():.4f} "
            f"(mean={grp['temporal_proximity_rate'].mean():.4f})",
            flush=True,
        )

    w20_rows = run_wisdm20_decoupling(wisdm20, seeds=(1, 2, 3))
    pd.DataFrame(w20_rows).to_csv(out / "wisdm_uniform20hz_decoupling.csv", index=False)

    print("5) Running 5-fold subject-level paired CV on WISDM uniform 20Hz (n=51) ...", flush=True)
    w20_cv = run_subject_level_cv(wisdm20, n_folds=5)
    print("WISDM20 5-fold subject-level CV:", w20_cv, flush=True)

    pd.DataFrame([har_cv, w20_cv]).to_csv(out / "subject_level_5fold_cv.csv", index=False)
    print(f"All sensitivity checks done in {time.time()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
