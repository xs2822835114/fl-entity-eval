import itertools
from typing import Dict, Optional
import numpy as np
import pandas as pd


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp)) if len(yt) > 0 else 0.0


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    classes = np.arange(n_classes) if n_classes is not None else np.unique(np.concatenate([yt, yp]))
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


def fixed_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over a fixed class index set 0..n_classes-1."""
    return _macro_f1(y_true, y_pred, n_classes=n_classes)


def present_class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 averaged strictly over classes present in y_true.

    Unlike fixed_macro_f1, a 100% perfect oracle attains 1.000 even when an
    entity's held-out recordings contain only a subset of the global label set.
    False predictions into absent classes still reduce recall of the present
    classes via false negatives. Note that when only a single class is present
    (|P_e| = 1), FP = 0 so F1 simplifies to 2r / (1 + r) >= r; pair with
    per-entity accuracy (driver_worst_window_acc) for single-class entities.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    present = np.unique(yt)
    if len(present) == 0:
        return 0.0
    f1s = []
    for c in present:
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


def exact_paired_sign_flip_test(diffs: np.ndarray, max_exact_n: int = 18, n_perm: int = 200000, seed: int = 42) -> Dict[str, float]:
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
        mc_res = min_p
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        perm_means = np.abs(np.mean(signs * d, axis=1))
        p_val = float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))
        mc_res = 1.0 / (n_perm + 1)

    return {
        "n_units": int(n),
        "mean_diff": float(np.mean(d)),
        "p_value": p_val,
        "min_achievable_p": float(mc_res),
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
    """Reports window-level, trip-voted, driver-level (fixed & present-class), and FL protocol metrics."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        driver_ids: np.ndarray,
        trip_ids: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
    ) -> Dict[str, float]:
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "driver": driver_ids})
        res: Dict[str, float] = {
            "n_windows": int(len(df)),
            "n_drivers": int(df["driver"].nunique()),
            "window_acc": _accuracy(y_true, y_pred),
            "window_macro_f1": _macro_f1(y_true, y_pred, n_classes=n_classes),
        }

        drv_f1s = {}
        drv_present_f1s = {}
        drv_accs = {}
        for d_id, g in df.groupby("driver"):
            yt_g = g["y_true"].to_numpy()
            yp_g = g["y_pred"].to_numpy()
            drv_f1s[d_id] = _macro_f1(yt_g, yp_g, n_classes=n_classes)
            drv_present_f1s[d_id] = present_class_macro_f1(yt_g, yp_g)
            drv_accs[d_id] = _accuracy(yt_g, yp_g)
        res["driver_mean_macro_f1"] = float(np.mean(list(drv_f1s.values())))
        res["driver_worst_macro_f1"] = float(np.min(list(drv_f1s.values())))
        res["driver_mean_present_f1"] = float(np.mean(list(drv_present_f1s.values())))
        res["driver_worst_present_f1"] = float(np.min(list(drv_present_f1s.values())))
        res["driver_mean_window_acc"] = float(np.mean(list(drv_accs.values())))
        res["driver_worst_window_acc"] = float(np.min(list(drv_accs.values())))

        if trip_ids is not None:
            df["trip"] = trip_ids
            trip_true, trip_pred, trip_drv = [], [], []
            for _t_id, g in df.groupby("trip"):
                tt = g["y_true"].mode().iloc[0]
                tp = g["y_pred"].mode().iloc[0]
                trip_true.append(tt)
                trip_pred.append(tp)
                trip_drv.append(g["driver"].iloc[0])
            res["n_trips"] = int(len(trip_true))
            res["trip_vote_acc"] = _accuracy(np.array(trip_true), np.array(trip_pred))
            res["trip_vote_macro_f1"] = _macro_f1(
                np.array(trip_true), np.array(trip_pred), n_classes=n_classes
            )

            tdf = pd.DataFrame({"driver": trip_drv, "correct": (np.array(trip_true) == np.array(trip_pred)).astype(float)})
            drv_trip_acc = tdf.groupby("driver")["correct"].mean()
            res["driver_mean_trip_acc"] = float(drv_trip_acc.mean())
            res["driver_worst_trip_acc"] = float(drv_trip_acc.min())

        return res

    @staticmethod
    def evaluate_protocol_sensitivity(
        y_true_by_client: Dict[str, np.ndarray],
        y_pred_deployed_by_client: Dict[str, np.ndarray],
        y_pred_uniform_by_client: Dict[str, np.ndarray],
        trip_by_client: Dict[str, np.ndarray],
        train_classes_by_client: Dict[str, set],
        n_classes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Item-8 helper: compare deployed vs. uniform inference and log local zero-support rate."""
        yt_all, ypd_all, ypu_all, tr_all = [], [], [], []
        seen_corr, seen_tot, unseen_corr, unseen_tot = 0, 0, 0, 0
        per_client_dep, per_client_uni = [], []

        for cid in sorted(y_true_by_client.keys()):
            yt = np.asarray(y_true_by_client[cid], dtype=np.int64)
            ypd = np.asarray(y_pred_deployed_by_client[cid], dtype=np.int64)
            ypu = np.asarray(y_pred_uniform_by_client[cid], dtype=np.int64)
            tr = np.asarray(trip_by_client[cid])
            tr_ns = np.array([f"{cid}::{u}" for u in tr], dtype=object)
            yt_all.append(yt)
            ypd_all.append(ypd)
            ypu_all.append(ypu)
            tr_all.append(tr_ns)

            c_dep_corr, c_uni_corr, c_tot = 0, 0, 0
            local_seen = set(train_classes_by_client.get(cid, set()))
            for u_tr in np.unique(tr):
                m = tr == u_tr
                t_lab = int(np.bincount(yt[m]).argmax())
                pd_lab = int(np.bincount(ypd[m]).argmax())
                pu_lab = int(np.bincount(ypu[m]).argmax())
                c_dep_corr += int(t_lab == pd_lab)
                c_uni_corr += int(t_lab == pu_lab)
                c_tot += 1
                if t_lab in local_seen:
                    seen_corr += int(t_lab == pd_lab)
                    seen_tot += 1
                else:
                    unseen_corr += int(t_lab == pd_lab)
                    unseen_tot += 1
            if c_tot > 0:
                per_client_dep.append(c_dep_corr / c_tot)
                per_client_uni.append(c_uni_corr / c_tot)

        yt_cat = np.concatenate(yt_all)
        ypd_cat = np.concatenate(ypd_all)
        ypu_cat = np.concatenate(ypu_all)
        tr_cat = np.concatenate(tr_all)

        def _trip_acc(yt_c: np.ndarray, yp_c: np.ndarray, tr_c: np.ndarray) -> float:
            corr, tot = 0, 0
            for u in np.unique(tr_c):
                m = tr_c == u
                corr += int(np.bincount(yt_c[m]).argmax() == np.bincount(yp_c[m]).argmax())
                tot += 1
            return float(corr / tot) if tot > 0 else float("nan")

        tot_trips = seen_tot + unseen_tot
        return {
            "n_clients": float(len(per_client_dep)),
            "win_f1_deployed": _macro_f1(yt_cat, ypd_cat, n_classes=n_classes),
            "win_f1_uniform": _macro_f1(yt_cat, ypu_cat, n_classes=n_classes),
            "trip_acc_deployed": _trip_acc(yt_cat, ypd_cat, tr_cat),
            "trip_acc_uniform": _trip_acc(yt_cat, ypu_cat, tr_cat),
            "client_mean_trip_acc_deployed": float(np.mean(per_client_dep)) if per_client_dep else float("nan"),
            "client_mean_trip_acc_uniform": float(np.mean(per_client_uni)) if per_client_uni else float("nan"),
            "zero_support_trip_frac": float(unseen_tot / tot_trips) if tot_trips > 0 else 0.0,
            "seen_cls_trip_acc_deployed": float(seen_corr / seen_tot) if seen_tot > 0 else float("nan"),
            "unseen_cls_trip_acc_deployed": float(unseen_corr / unseen_tot) if unseen_tot > 0 else float("nan"),
        }
