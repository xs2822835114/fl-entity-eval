from typing import Dict, Optional
import numpy as np
import pandas as pd


class LeakageAuditor:
    """Automated train-test entity overlap and temporal-proximity leakage auditor."""

    @staticmethod
    def audit_split(
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        driver_col: Optional[str] = None,
        trip_col: Optional[str] = None,
        time_col: Optional[str] = None,
        window_sec: float = 30.0,
    ) -> Dict[str, object]:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        n_test = len(test_df)
        report: Dict[str, object] = {
            "n_train": int(len(train_df)),
            "n_test": int(n_test),
            "index_overlap": int(len(set(train_idx).intersection(set(test_idx)))),
        }

        if driver_col and driver_col in df.columns:
            tr_drv = set(train_df[driver_col].unique())
            te_drv = set(test_df[driver_col].unique())
            overlap_mask = test_df[driver_col].isin(tr_drv)
            report["n_train_drivers"] = int(len(tr_drv))
            report["n_test_drivers"] = int(len(te_drv))
            report["n_unseen_test_drivers"] = int(len(te_drv - tr_drv))
            report["driver_overlap_rate"] = float(overlap_mask.mean()) if n_test else 0.0

        if trip_col and trip_col in df.columns:
            tr_trp = set(train_df[trip_col].unique())
            te_trp = set(test_df[trip_col].unique())
            overlap_mask = test_df[trip_col].isin(tr_trp)
            report["n_train_trips"] = int(len(tr_trp))
            report["n_test_trips"] = int(len(te_trp))
            report["n_unseen_test_trips"] = int(len(te_trp - tr_trp))
            report["trip_overlap_rate"] = float(overlap_mask.mean()) if n_test else 0.0

        if trip_col and time_col and trip_col in df.columns and time_col in df.columns:
            tr_times_by_trip = {
                k: np.sort(g[time_col].to_numpy(dtype=float))
                for k, g in train_df.groupby(trip_col)
            }
            prox_hits = 0
            for _, row in test_df[[trip_col, time_col]].iterrows():
                t_arr = tr_times_by_trip.get(row[trip_col])
                if t_arr is not None and len(t_arr) > 0:
                    dist = np.min(np.abs(t_arr - float(row[time_col])))
                    if dist <= window_sec:
                        prox_hits += 1
            report["temporal_proximity_hits"] = int(prox_hits)
            report["temporal_proximity_leak_rate"] = float(prox_hits / n_test) if n_test else 0.0

        trip_leak = float(report.get("trip_overlap_rate", 0.0))
        drv_leak = float(report.get("driver_overlap_rate", 0.0))
        if report["index_overlap"] > 0 or trip_leak > 0.0:
            report["audit_verdict"] = "FAIL_TRIP_LEAKAGE"
        elif drv_leak > 0.0:
            report["audit_verdict"] = "WARN_WITHIN_DRIVER_ONLY"
        else:
            report["audit_verdict"] = "PASS_ENTITY_ISOLATED"

        return report
