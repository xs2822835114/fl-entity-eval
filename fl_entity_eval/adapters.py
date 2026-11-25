"""Lightweight adapters integrating fl-entity-eval with PyTorch and Flower pipelines."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .splitter import EntityGroupSplitter


def train_only_standardize(
    X: np.ndarray,
    train_idx: np.ndarray,
    axis: Tuple[int, ...] = (0, 2),
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Item-5 helper: fit channel-wise z-score statistics strictly on train_idx and transform X."""
    x_arr = np.asarray(X, dtype=np.float32)
    idx = np.asarray(train_idx, dtype=np.int64)
    mean = x_arr[idx].mean(axis=axis, keepdims=True)
    std = x_arr[idx].std(axis=axis, keepdims=True)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return ((x_arr - mean) / std).astype(np.float32), mean.astype(np.float32), std


def make_pytorch_subsets(
    dataset: Any,
    meta_df: pd.DataFrame,
    entity_col: str,
    test_frac: float = 0.3,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    """Return entity-isolated (train_subset, test_subset, train_idx, test_idx) for any PyTorch Dataset."""
    from torch.utils.data import Subset

    tr_idx, te_idx = EntityGroupSplitter.split_whole_entity(
        meta_df, entity_col=entity_col, test_frac=test_frac, stratify_col=stratify_col, seed=seed
    )
    return Subset(dataset, tr_idx.tolist()), Subset(dataset, te_idx.tolist()), tr_idx, te_idx


class FlowerEntityPartitioner:
    """Layer-1 (client) + Layer-2 (within-client session) index partitioner for Flower pipelines.

    Unlike Flower's NaturalIdPartitioner (Layer 1 only), this lightweight index
    partitioner binds a whole-session holdout inside each participating client,
    with optional class stratification (to prevent zero-support local class
    truncation) and optional unseen-client holdout (`unseen_client_frac > 0`).
    It requires only NumPy/Pandas so users can drop it into Flower
    NumPyClient/ClientApp loaders without a mandatory flwr_datasets dependency.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        client_col: str,
        session_col: str,
        label_col: Optional[str] = None,
        test_frac: float = 0.30,
        stratified: bool = True,
        unseen_client_frac: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.meta_df = meta_df.reset_index(drop=True)
        self.client_col = client_col
        self.session_col = session_col
        self.label_col = label_col
        self.test_frac = test_frac
        self.stratified = stratified
        self.unseen_client_frac = float(unseen_client_frac)
        self.seed = seed
        self._unseen_partitions: Dict[str, np.ndarray] = {}
        self._partitions: Dict[str, Dict[str, np.ndarray]] = self._build()

    def _build(self) -> Dict[str, Dict[str, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        all_cids = sorted(self.meta_df[self.client_col].unique(), key=str)
        unseen_set = set()
        if self.unseen_client_frac > 0.0 and len(all_cids) >= 2:
            perm_c = rng.permutation(all_cids)
            n_unseen = min(len(all_cids) - 1, max(1, int(round(len(all_cids) * self.unseen_client_frac))))
            unseen_set = set(str(c) for c in perm_c[:n_unseen])
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for cid, grp in self.meta_df.groupby(self.client_col, sort=True):
            c_idx = grp.index.to_numpy()
            if str(cid) in unseen_set:
                self._unseen_partitions[str(cid)] = np.sort(c_idx)
                continue
            sess_vals = grp[self.session_col].to_numpy()
            if self.stratified and self.label_col is not None and self.label_col in grp.columns:
                labels = grp[self.label_col].to_numpy()
                by_cls: Dict[Any, list] = {}
                for s in np.unique(sess_vals):
                    sm = sess_vals == s
                    vals, cnts = np.unique(labels[sm], return_counts=True)
                    maj = vals[int(np.argmax(cnts))]
                    by_cls.setdefault(maj, []).append(s)
                te_sess = set()
                for _cls, s_list in sorted(by_cls.items(), key=lambda kv: str(kv[0])):
                    if len(s_list) < 2:
                        continue
                    perm = rng.permutation(s_list)
                    n_te = min(len(s_list) - 1, max(1, int(round(len(s_list) * self.test_frac))))
                    te_sess.update(perm[:n_te].tolist())
            else:
                uniq_s = np.unique(sess_vals)
                if len(uniq_s) < 2:
                    continue
                perm = rng.permutation(uniq_s)
                n_te = min(len(uniq_s) - 1, max(1, int(round(len(uniq_s) * self.test_frac))))
                te_sess = set(perm[:n_te].tolist())
            te_mask = np.isin(sess_vals, list(te_sess))
            tr_idx = c_idx[~te_mask]
            te_idx = c_idx[te_mask]
            if len(tr_idx) > 0 and len(te_idx) > 0:
                out[str(cid)] = {"train": tr_idx, "test": te_idx}
        return out

    @property
    def client_ids(self):
        return sorted(self._partitions.keys())

    @property
    def unseen_client_ids(self):
        return sorted(self._unseen_partitions.keys())

    def load_partition(self, client_id: str) -> Dict[str, np.ndarray]:
        """Return {'train': np.ndarray, 'test': np.ndarray} indices for a participating Flower client."""
        return self._partitions[str(client_id)]

    def load_unseen_client(self, client_id: str) -> np.ndarray:
        """Return all test indices for a held-out unseen Flower client."""
        return self._unseen_partitions[str(client_id)]


def support_masked_head_avg(
    client_weights: Dict[str, np.ndarray],
    client_biases: Dict[str, np.ndarray],
    client_class_counts: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Post-hoc support-masked classifier-head aggregation (FedPer-Guard diagnostic).

    Averages each fallback class row c strictly across clients with count[c] > 0
    (weighted by per-class sample count), avoiding negative-logit pollution from
    unsupported clients. Note: requires uploading personal head rows and per-class
    support counts after training (use secure aggregation in privacy-sensitive fleets).

    Returns (fallback_weight, fallback_bias, guarded_heads_by_client).
    """
    cids = sorted(client_weights.keys())
    if not cids:
        raise ValueError("client_weights must be non-empty.")
    ref_w = np.asarray(client_weights[cids[0]], dtype=np.float32)
    ref_b = np.asarray(client_biases[cids[0]], dtype=np.float32)
    n_classes = ref_w.shape[0]
    fallback_w = np.zeros_like(ref_w, dtype=np.float64)
    fallback_b = np.zeros_like(ref_b, dtype=np.float64)
    mean_w = np.mean([np.asarray(client_weights[c], dtype=np.float64) for c in cids], axis=0)
    mean_b = np.mean([np.asarray(client_biases[c], dtype=np.float64) for c in cids], axis=0)

    for cls in range(n_classes):
        tot = 0.0
        for cid in cids:
            cnt = float(np.asarray(client_class_counts[cid], dtype=np.float64)[cls])
            if cnt > 0.0:
                fallback_w[cls] += np.asarray(client_weights[cid], dtype=np.float64)[cls] * cnt
                fallback_b[cls] += float(np.asarray(client_biases[cid], dtype=np.float64)[cls]) * cnt
                tot += cnt
        if tot > 0.0:
            fallback_w[cls] /= tot
            fallback_b[cls] /= tot
        else:
            fallback_w[cls] = mean_w[cls]
            fallback_b[cls] = mean_b[cls]

    fb_w = fallback_w.astype(np.float32)
    fb_b = fallback_b.astype(np.float32)
    guarded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cid in cids:
        gw = np.array(client_weights[cid], dtype=np.float32, copy=True)
        gb = np.array(client_biases[cid], dtype=np.float32, copy=True)
        cnts = np.asarray(client_class_counts[cid], dtype=np.float64)
        for cls in range(n_classes):
            if cnts[cls] <= 0.0:
                gw[cls] = fb_w[cls]
                gb[cls] = fb_b[cls]
        guarded[cid] = (gw, gb)
    return fb_w, fb_b, guarded
