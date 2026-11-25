#!/usr/bin/env python3
"""Rebuild UAH-DriveSet windows in RAW physical units (NO pre-split z-scoring).

Identical alignment, gap splitting, and physical clipping as preprocess_uah.py,
strictly preserving native physical sensor units (speed in km/h, ax/ay/az in g,
roll/pitch/yaw in rad/deg) so that train-only standardisation can be fitted
strictly inside each train-test split (Protocol Item 5) and signal-level ICC(1)
is computed on genuine raw accelerometer magnitude.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
import numpy as np

RAW_ROOT = Path(os.environ.get("UAH_RAW_ROOT", "datasets/UAH-DRIVESET-v1"))
OUT_DIR = Path(os.environ.get("FL_ENTITY_EVAL_RAW_UAH", str(Path(__file__).resolve().parent.parent / "datasets" / "uah_raw")))

FS = 10.0
GAP_SPLIT_S = 1.0
WINDOW_LENGTHS = [10, 30]
LABEL_MAP = {"NORMAL": 0, "DROWSY": 1, "AGGRESSIVE": 2}
ACC_COLS = [0, 2, 3, 4, 8, 9, 10]  # t, ax, ay, az, roll, pitch, yaw
GPS_COLS = [0, 1]                  # t, speed


def loadtxt_clean(path: Path, cols) -> np.ndarray:
    arr = np.genfromtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr[:, cols]
    arr = arr[np.isfinite(arr[:, 0])]
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 0])
    return arr[keep]


def split_segments(t: np.ndarray, gap_s: float = GAP_SPLIT_S):
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > gap_s)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(t)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1]


def process_trip(trip_dir: Path):
    parts = trip_dir.name.split("-")
    driver, behavior, road = parts[2], parts[3], parts[4]
    label = LABEL_MAP[re.sub(r"\d+$", "", behavior)]

    acc = loadtxt_clean(trip_dir / "RAW_ACCELEROMETERS.txt", ACC_COLS)
    gps = loadtxt_clean(trip_dir / "RAW_GPS.txt", GPS_COLS)
    if len(acc) < 100 or len(gps) < 10:
        return None

    t0 = max(acc[0, 0], gps[0, 0])
    t1 = min(acc[-1, 0], gps[-1, 0])
    if t1 - t0 < 60:
        return None
    grid = np.arange(t0, t1, 1.0 / FS)

    segs_acc = split_segments(acc[:, 0])
    segs = []
    for i0, i1 in segs_acc:
        a = acc[i0:i1]
        if len(a) < FS * 10:
            continue
        m = (grid >= a[0, 0]) & (grid <= a[-1, 0])
        if m.sum() < FS * 10:
            continue
        g = grid[m]
        chans = [np.interp(g, a[:, 0], a[:, c]) for c in range(1, acc.shape[1])]
        sp = np.interp(g, gps[:, 0], gps[:, 1], left=np.nan, right=np.nan)
        ok = np.isfinite(sp)
        if (~ok).all():
            continue
        sp = np.interp(g, g[ok], sp[ok])
        X = np.stack([sp] + chans, axis=1)  # (T, 7): speed, ax, ay, az, roll, pitch, yaw
        np.clip(X[:, 0], 0, 300, out=X[:, 0])
        np.clip(X[:, 1:4], -4, 4, out=X[:, 1:4])
        np.clip(X[:, 4:6], -90, 90, out=X[:, 4:6])
        segs.append((g, X))
    if not segs:
        return None
    return dict(driver=driver, behavior=behavior, road=road, label=label, segs=segs, trip=trip_dir.name)


def main() -> None:
    trips = []
    for d in sorted(RAW_ROOT.glob("D*")):
        for trip_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            t = process_trip(trip_dir)
            if t is not None:
                trips.append(t)
    print(f"Rebuilt RAW physical-unit UAH trips: {len(trips)} | drivers: {sorted({t['driver'] for t in trips})}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    store = {L: dict(X=[], y=[], driver=[], trip=[], road=[], t0=[]) for L in WINDOW_LENGTHS}
    for t in trips:
        for g, X in t["segs"]:
            # NO pre-split normalisation! Keep raw physical units.
            for L in WINDOW_LENGTHS:
                W = int(L * FS)
                stride = W // 2
                for s in range(0, len(X) - W + 1, stride):
                    store[L]["X"].append(X[s : s + W].astype(np.float32))
                    store[L]["y"].append(t["label"])
                    store[L]["driver"].append(t["driver"])
                    store[L]["trip"].append(t["trip"])
                    store[L]["road"].append(t["road"])
                    store[L]["t0"].append(float(g[s]))

    for L in WINDOW_LENGTHS:
        d = store[L]
        X_arr = np.stack(d["X"])
        np.savez_compressed(
            OUT_DIR / f"windows_L{L}_raw.npz",
            X=X_arr,
            y=np.asarray(d["y"], dtype=np.int64),
            driver=np.asarray(d["driver"]),
            trip=np.asarray(d["trip"]),
            road=np.asarray(d["road"]),
            t0=np.asarray(d["t0"]),
        )
        print(
            f"L={L}s: shape={X_arr.shape}, speed mean={X_arr[:,:,0].mean():.2f} km/h (std={X_arr[:,:,0].std():.2f}), "
            f"accel norm mean={np.linalg.norm(X_arr[:,:,1:4], axis=2).mean():.4f} g"
        )


if __name__ == "__main__":
    main()
