import hashlib
import json
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class EntityGroupSplitter:
    """Entity-aware and size-matched splitters for federated driving datasets.

    Supports:
    - S2: Whole-entity (whole-trip / whole-driver / whole-route) holdout with zero entity overlap.
    - S1: Standard leaky window-level random split within seen trips.
    - S1_matched / S2_matched: Size-matched test splits sharing the exact same training set and
      identical test window count, decoupling data leakage from sample-size confounding.
    """

    @staticmethod
    def split_whole_entity(
        df: pd.DataFrame,
        entity_col: str,
        test_frac: float = 0.3,
        stratify_col: Optional[str] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        if stratify_col is None:
            entities = np.array(sorted(df[entity_col].unique()))
            rng.shuffle(entities)
            n_test = max(1, int(round(len(entities) * test_frac)))
            test_entities = set(entities[:n_test])
        else:
            test_entities = set()
            for _, grp in df.groupby(stratify_col):
                ents = np.array(sorted(grp[entity_col].unique()))
                rng.shuffle(ents)
                n_test = max(1, int(round(len(ents) * test_frac))) if len(ents) > 1 else 0
                test_entities.update(ents[:n_test])
        test_mask = df[entity_col].isin(test_entities).to_numpy()
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx

    @staticmethod
    def split_window_random(
        df: pd.DataFrame,
        test_frac: float = 0.2,
        pool_indices: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df)) if pool_indices is None else np.array(pool_indices, copy=True)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx = np.sort(idx[:n_test])
        train_idx = np.sort(idx[n_test:])
        return train_idx, test_idx

    @staticmethod
    def split_size_matched_pair(
        df: pd.DataFrame,
        trip_col: str,
        s2_train_idx: np.ndarray,
        s2_test_idx: np.ndarray,
        s1_test_frac: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        shared_train, s1_test = EntityGroupSplitter.split_window_random(
            df, test_frac=s1_test_frac, pool_indices=s2_train_idx, seed=seed
        )
        n_match = min(len(s1_test), len(s2_test_idx))
        s1_matched_test = np.sort(rng.choice(s1_test, size=n_match, replace=False))
        s2_matched_test = np.sort(rng.choice(s2_test_idx, size=n_match, replace=False))
        return {
            "S1": (shared_train, s1_test),
            "S1_matched": (shared_train, s1_matched_test),
            "S2_matched": (shared_train, s2_matched_test),
            "S2": (shared_train, np.sort(s2_test_idx)),
        }

    @staticmethod
    def export_manifest(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        metadata: Dict,
        out_path: str,
    ) -> str:
        payload = {
            "metadata": metadata,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_indices": [int(i) for i in train_idx],
            "test_indices": [int(i) for i in test_idx],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload["sha256"]
