from typing import Dict, List
import numpy as np
import pandas as pd


class ICCDiagnostics:
    """Computes one-way random-effects ICC(1), Design Effect (DEFF), and Effective Sample Size (N_eff)."""

    @staticmethod
    def compute_icc_and_neff(values: np.ndarray, group_ids: np.ndarray) -> Dict[str, float]:
        y = np.asarray(values, dtype=float)
        g = np.asarray(group_ids)
        unique_g, inv = np.unique(g, return_inverse=True)
        k = len(unique_g)
        N = len(y)
        if k < 2 or N <= k:
            m_bar = (N / float(k)) if k > 0 else 0.0
            return {
                "n_obs": float(N),
                "n_groups": float(k),
                "mean_cluster_size": float(m_bar),
                "cv_cluster_size": 0.0,
                "icc": 0.0,
                "deff": 1.0,
                "n_eff": float(N),
                "neff_ratio": 1.0 if N > 0 else 0.0,
                "deff_unbalanced": 1.0,
                "n_eff_unbalanced": float(N),
                "neff_ratio_unbalanced": 1.0 if N > 0 else 0.0,
            }

        grand_mean = np.mean(y)
        counts = np.bincount(inv).astype(float)
        group_means = np.bincount(inv, weights=y) / counts

        ss_between = float(np.sum(counts * (group_means - grand_mean) ** 2))
        ss_within = float(np.sum((y - group_means[inv]) ** 2))

        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (N - k)

        n0 = (N - float(np.sum(counts ** 2)) / N) / (k - 1)
        denom = ms_between + (n0 - 1.0) * ms_within
        icc = max(0.0, (ms_between - ms_within) / denom) if denom > 0 else 0.0

        m_bar = N / float(k)
        cv_m = float(np.std(counts, ddof=1) / m_bar) if (k > 1 and m_bar > 0) else 0.0
        deff = 1.0 + (m_bar - 1.0) * icc
        n_eff = N / deff if deff > 0 else float(N)
        deff_unbal = 1.0 + ((cv_m ** 2 + 1.0) * m_bar - 1.0) * icc
        n_eff_unbal = N / deff_unbal if deff_unbal > 0 else float(N)
        return {
            "n_obs": float(N),
            "n_groups": float(k),
            "mean_cluster_size": float(m_bar),
            "cv_cluster_size": float(cv_m),
            "icc": float(icc),
            "deff": float(deff),
            "n_eff": float(n_eff),
            "neff_ratio": float(n_eff / N),
            "deff_unbalanced": float(deff_unbal),
            "n_eff_unbalanced": float(n_eff_unbal),
            "neff_ratio_unbalanced": float(n_eff_unbal / N),
        }

    @staticmethod
    def compute_class_residualized_icc(
        values: np.ndarray, group_ids: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute ICC(1)/DEFF after subtracting class means to remove between-class signal variance.

        Note: passing residuals to one-way ANOVA uses k - 1 rather than k - C between-group
        degrees of freedom, which makes DEFF_resid slightly conservative (by ~0.4%-3.3% on
        single-label recordings due to nonlinear MSB propagation in ICC(1)).
        """
        y = np.asarray(values, dtype=float)
        lab = np.asarray(labels)
        resid = np.empty_like(y)
        for c in np.unique(lab):
            m = lab == c
            resid[m] = y[m] - np.mean(y[m])
        return ICCDiagnostics.compute_icc_and_neff(resid, group_ids)

    @staticmethod
    def compute_error_deff(
        y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray
    ) -> Dict[str, float]:
        """Compute ICC(1)/DEFF/N_eff directly on binary model prediction correctness indicators."""
        correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
        return ICCDiagnostics.compute_icc_and_neff(correct, group_ids)

    @staticmethod
    def sweep_window_overlap(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        window_lengths: List[int],
        overlap_ratios: List[float],
        allow_short: bool = False,
    ) -> pd.DataFrame:
        rows = []
        for w in window_lengths:
            for ov in overlap_ratios:
                step = max(1, int(round(w * (1.0 - ov))))
                win_vals, win_grps = [], []
                for gid, grp in df.groupby(group_col):
                    arr = grp[value_col].to_numpy(dtype=float)
                    if len(arr) < w:
                        if allow_short and len(arr) > 0:
                            win_vals.append(float(np.mean(arr)))
                            win_grps.append(gid)
                        continue
                    for start in range(0, len(arr) - w + 1, step):
                        seg = arr[start : start + w]
                        win_vals.append(float(np.mean(seg)))
                        win_grps.append(gid)
                stats = ICCDiagnostics.compute_icc_and_neff(np.array(win_vals), np.array(win_grps))
                stats["window_len"] = w
                stats["overlap_ratio"] = ov
                rows.append(stats)
        return pd.DataFrame(rows)
