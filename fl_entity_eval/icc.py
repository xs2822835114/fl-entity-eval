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
            return {"n_obs": float(N), "n_groups": float(k), "icc": 0.0, "deff": 1.0, "n_eff": float(N), "neff_ratio": 1.0}

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
        deff = 1.0 + (m_bar - 1.0) * icc
        n_eff = N / deff if deff > 0 else float(N)
        return {
            "n_obs": float(N),
            "n_groups": float(k),
            "mean_cluster_size": float(m_bar),
            "icc": float(icc),
            "deff": float(deff),
            "n_eff": float(n_eff),
            "neff_ratio": float(n_eff / N),
        }

    @staticmethod
    def sweep_window_overlap(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        window_lengths: List[int],
        overlap_ratios: List[float],
    ) -> pd.DataFrame:
        rows = []
        for w in window_lengths:
            for ov in overlap_ratios:
                step = max(1, int(round(w * (1.0 - ov))))
                win_vals, win_grps = [], []
                for gid, grp in df.groupby(group_col):
                    arr = grp[value_col].to_numpy(dtype=float)
                    for start in range(0, max(1, len(arr) - w + 1), step):
                        seg = arr[start : start + w]
                        if len(seg) > 0:
                            win_vals.append(float(np.mean(seg)))
                            win_grps.append(gid)
                stats = ICCDiagnostics.compute_icc_and_neff(np.array(win_vals), np.array(win_grps))
                stats["window_len"] = w
                stats["overlap_ratio"] = ov
                rows.append(stats)
        return pd.DataFrame(rows)
