#!/usr/bin/env python3
"""Convert the official UCI HAR Dataset directory (Inertial Signals) into the compact
`uci_har_windows.npz` archive consumed by `run_entity_decoupling.py` and
`run_fl_protocol_sensitivity.py`.

Usage
-----
Download and extract the official UCI HAR Dataset zip from the UCI Machine Learning
Repository (Anguita et al., 2013), then run:

    python scripts/prepare_uci_har_npz.py \
        --raw-dir /path/to/UCI_HAR_Dataset \
        --out-path datasets/data/processed/uci_har_windows.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_split(raw_dir: Path, split: str):
    sig_dir = raw_dir / split / "Inertial Signals"
    channels = []
    for name in SIGNAL_NAMES:
        arr = np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32)
        channels.append(arr)
    X = np.stack(channels, axis=1).astype(np.float32)  # (N, 9, 128)
    y = np.loadtxt(raw_dir / split / f"y_{split}.txt", dtype=np.int64) - 1
    subj = np.loadtxt(raw_dir / split / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Extracted UCI HAR Dataset folder")
    parser.add_argument("--out-path", required=True, help="Destination .npz file path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    X_tr, y_tr, c_tr = _load_split(raw_dir, "train")
    X_te, y_te, c_te = _load_split(raw_dir, "test")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_tr, y_train=y_tr, client_train=c_tr,
        X_test=X_te, y_test=y_te, client_test=c_te,
    )
    print(f"Saved {out_path} | train={X_tr.shape}, test={X_te.shape}, subjects={len(np.unique(np.concatenate([c_tr, c_te])))}")


if __name__ == "__main__":
    main()
