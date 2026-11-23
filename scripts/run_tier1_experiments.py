#!/usr/bin/env python3
"""
Tier-1 Experiment Runner for Evaluation Validity & Leakage Decoupling.
Supports:
  1. --demo : runs end-to-end synthetic driving demo (S1 / S1_matched / S2_matched / S2 + LeakageAuditor + ICC + Driver Sign-Flip Test).
  2. --csv <path> : runs the exact same pipeline on real UAH-DriveSet / UCI HAR / WISDM / PeMS CSV exported from your local FL runner.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl_entity_eval import (
    EntityGroupSplitter,
    LeakageAuditor,
    ICCDiagnostics,
    UnitLevelEvaluator,
    exact_paired_sign_flip_test,
    holm_bonferroni,
)


def make_synthetic_driving_df(n_drivers: int = 12, trips_per_driver: int = 6, windows_per_trip: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        drv_bias = rng.normal(0, 1.2)
        for t in range(trips_per_driver):
            trip_id = f"D{d:02d}_T{t:02d}"
            label = t % 3
            trip_bias = rng.normal(0, 0.8)
            for w in range(windows_per_trip):
                feat = drv_bias + trip_bias + label * 0.6 + rng.normal(0, 0.25)
                rows.append({
                    "driver_id": f"D{d:02d}",
                    "trip_id": trip_id,
                    "timestamp_sec": w * 15.0,
                    "signal_feat": feat,
                    "label": label,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run built-in synthetic demo")
    parser.add_argument("--csv", type=str, default="", help="Path to real dataset CSV with driver_id, trip_id, timestamp_sec, signal_feat, label")
    parser.add_argument("--out_dir", type=str, default="tier1_eval_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv) if args.csv else make_synthetic_driving_df(seed=args.seed)

    s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(df, entity_col="trip_id", test_frac=0.33, stratify_col="driver_id", seed=args.seed)
    splits = EntityGroupSplitter.split_size_matched_pair(df, trip_col="trip_id", s2_train_idx=s2_tr, s2_test_idx=s2_te, s1_test_frac=0.2, seed=args.seed)

    audit_rows = []
    for name, (tr_i, te_i) in splits.items():
        aud = LeakageAuditor.audit_split(df, tr_i, te_i, driver_col="driver_id", trip_col="trip_id", time_col="timestamp_sec", window_sec=30.0)
        sha = EntityGroupSplitter.export_manifest(tr_i, te_i, {"split": name, "seed": args.seed}, os.path.join(args.out_dir, f"manifest_{name}.json"))
        aud["split"] = name
        aud["manifest_sha256"] = sha[:12]
        audit_rows.append(aud)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(args.out_dir, "leakage_audit_summary.csv"), index=False)

    icc_trip = ICCDiagnostics.compute_icc_and_neff(df["signal_feat"].to_numpy(), df["trip_id"].to_numpy())
    icc_sweep = ICCDiagnostics.sweep_window_overlap(df, "signal_feat", "trip_id", window_lengths=[4, 8, 16], overlap_ratios=[0.0, 0.5, 0.75])
    icc_sweep.to_csv(os.path.join(args.out_dir, "icc_neff_decay_curve.csv"), index=False)

    rng = np.random.default_rng(args.seed)
    _, te_s2 = splits["S2"]
    te_df = df.iloc[te_s2]
    y_true = te_df["label"].to_numpy()
    y_pred_base = np.where(rng.random(len(y_true)) < 0.65, y_true, (y_true + 1) % 3)
    y_pred_alt = np.where(rng.random(len(y_true)) < 0.78, y_true, (y_true + 1) % 3)
    ev_base = UnitLevelEvaluator.evaluate_predictions(y_true, y_pred_base, te_df["driver_id"].to_numpy(), te_df["trip_id"].to_numpy())
    ev_alt = UnitLevelEvaluator.evaluate_predictions(y_true, y_pred_alt, te_df["driver_id"].to_numpy(), te_df["trip_id"].to_numpy())

    drv_diffs = []
    for drv, g in te_df.groupby("driver_id"):
        pos = np.isin(te_df.index, g.index)
        drv_diffs.append(float(np.mean(y_pred_alt[pos] == y_true[pos]) - np.mean(y_pred_base[pos] == y_true[pos])))
    sig = exact_paired_sign_flip_test(np.array(drv_diffs))
    adj = holm_bonferroni({"MethodA_vs_FedAvg": sig["p_value"]})

    print("=== LEAKAGE AUDIT SUMMARY ===")
    print(audit_df[["split", "n_train", "n_test", "trip_overlap_rate", "temporal_proximity_leak_rate", "audit_verdict"]].to_string(index=False))
    print("\n=== TRIP-LEVEL ICC & EFFECTIVE SAMPLE SIZE ===")
    print(icc_trip)
    print("\n=== S2 EVALUATION & DRIVER-LEVEL EXACT SIGN-FLIP TEST ===")
    print("Baseline:", ev_base)
    print("MethodA :", ev_alt)
    print("Exact test:", sig, "Holm-adjusted:", adj)


if __name__ == "__main__":
    main()
