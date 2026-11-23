import itertools
from typing import Dict, Optional
import numpy as np
import pandas as pd


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp)) if len(yt) > 0 else 0.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.unique(np.concatenate([yt, yp]))
    if len(classes) == 0:
        return 0.0
    f1s = []
    for c in classes:
        tp = float(np.sum((yt == c) & (yp == c)))
        fp = float(np.sum((yt != c) & (yp == c)))
        fn = float(np.sum((yt == c) & (yp != c)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def cohen_kappa(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """Pure-NumPy unweighted Cohen's kappa for inter-rater reliability."""
    a = np.asarray(rater1, dtype=str)
    b = np.asarray(rater2, dtype=str)
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a).union(set(b)))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == lab)) * float(np.mean(b == lab)) for lab in labels)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return float((po - pe) / (1.0 - pe))


def exact_paired_sign_flip_test(diffs: np.ndarray, max_exact_n: int = 16, n_perm: int = 100000, seed: int = 42) -> Dict[str, float]:
    """Two-sided paired sign-flip permutation test at the independent-entity level."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return {"n_units": 0, "mean_diff": 0.0, "p_value": 1.0, "min_achievable_p": 1.0}
    obs = abs(float(np.mean(d)))
    min_p = 2.0 / (2 ** n) if n <= 50 else 0.0

    if n <= max_exact_n:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        perm_means = np.abs(np.mean(signs * d, axis=1))
        p_val = float(np.mean(perm_means >= obs - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        perm_means = np.abs(np.mean(signs * d, axis=1))
        p_val = float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))

    return {
        "n_units": int(n),
        "mean_diff": float(np.mean(d)),
        "p_value": p_val,
        "min_achievable_p": float(min_p),
    }


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running_max = max(running_max, adj)
        adjusted[k] = float(running_max)
    return adjusted


class UnitLevelEvaluator:
    """Reports window-level, trip-voted, and driver-level mean/worst-entity performance."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        driver_ids: np.ndarray,
        trip_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "driver": driver_ids})
        res: Dict[str, float] = {
            "n_windows": int(len(df)),
            "n_drivers": int(df["driver"].nunique()),
            "window_acc": _accuracy(y_true, y_pred),
            "window_macro_f1": _macro_f1(y_true, y_pred),
        }

        drv_f1s = {}
        for d_id, g in df.groupby("driver"):
            drv_f1s[d_id] = _macro_f1(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        res["driver_mean_macro_f1"] = float(np.mean(list(drv_f1s.values())))
        res["driver_worst_macro_f1"] = float(np.min(list(drv_f1s.values())))

        if trip_ids is not None:
            df["trip"] = trip_ids
            trip_true, trip_pred, trip_drv = [], [], []
            for t_id, g in df.groupby("trip"):
                tt = g["y_true"].mode().iloc[0]
                tp = g["y_pred"].mode().iloc[0]
                trip_true.append(tt)
                trip_pred.append(tp)
                trip_drv.append(g["driver"].iloc[0])
            res["n_trips"] = int(len(trip_true))
            res["trip_vote_acc"] = _accuracy(np.array(trip_true), np.array(trip_pred))
            res["trip_vote_macro_f1"] = _macro_f1(np.array(trip_true), np.array(trip_pred))

            tdf = pd.DataFrame({"driver": trip_drv, "correct": (np.array(trip_true) == np.array(trip_pred)).astype(float)})
            drv_trip_acc = tdf.groupby("driver")["correct"].mean()
            res["driver_mean_trip_acc"] = float(drv_trip_acc.mean())
            res["driver_worst_trip_acc"] = float(drv_trip_acc.min())

        return res
