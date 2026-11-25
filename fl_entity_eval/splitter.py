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
            if len(entities) < 2:
                raise ValueError(f"split_whole_entity requires >= 2 unique '{entity_col}' values; got {len(entities)}.")
            rng.shuffle(entities)
            n_test = min(len(entities) - 1, max(1, int(round(len(entities) * test_frac))))
            test_entities = set(entities[:n_test])
        else:
            test_entities = set()
            for _, grp in df.groupby(stratify_col):
                ents = np.array(sorted(grp[entity_col].unique()))
                rng.shuffle(ents)
                n_test = min(len(ents) - 1, max(1, int(round(len(ents) * test_frac)))) if len(ents) > 1 else 0
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
        trip_col: Optional[str],
        s2_train_idx: np.ndarray,
        s2_test_idx: np.ndarray,
        s1_test_frac: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if trip_col is not None:
            if trip_col not in df.columns:
                raise KeyError(f"trip_col '{trip_col}' not found in DataFrame columns.")
            tr_trips = set(df.iloc[s2_train_idx][trip_col].unique())
            te_trips = set(df.iloc[s2_test_idx][trip_col].unique())
            if not tr_trips.isdisjoint(te_trips):
                raise ValueError(
                    f"S2 split violates recording isolation: {len(tr_trips & te_trips)} overlapping '{trip_col}' IDs."
                )
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
    def split_chronological(
        df: pd.DataFrame,
        time_col: str,
        test_frac: float = 0.3,
        entity_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chronological tail-block holdout globally or within each entity (Table 5 Item 3)."""
        if time_col not in df.columns:
            raise KeyError(f"time_col '{time_col}' not found in DataFrame columns.")
        if entity_col is None:
            order = np.argsort(df[time_col].to_numpy(), kind="stable")
            n_test = max(1, int(round(len(order) * test_frac))) if len(order) > 1 else 0
            cut = len(order) - n_test
            return np.sort(order[:cut]), np.sort(order[cut:])
        if entity_col not in df.columns:
            raise KeyError(f"entity_col '{entity_col}' not found in DataFrame columns.")
        ent_vals = df[entity_col].to_numpy()
        time_vals = df[time_col].to_numpy()
        tr_parts, te_parts = [], []
        for eid in sorted(pd.unique(ent_vals), key=str):
            pos = np.where(ent_vals == eid)[0]
            order = pos[np.argsort(time_vals[pos], kind="stable")]
            n_test = max(1, int(round(len(order) * test_frac))) if len(order) > 1 else 0
            cut = len(order) - n_test
            tr_parts.append(order[:cut])
            te_parts.append(order[cut:])
        tr = np.sort(np.concatenate(tr_parts)) if tr_parts else np.array([], dtype=np.int64)
        te = np.sort(np.concatenate(te_parts)) if te_parts else np.array([], dtype=np.int64)
        return tr, te

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

    @staticmethod
    def verify_manifest(manifest_path: str) -> bool:
        """Verify SHA-256 integrity of a split manifest JSON exported by export_manifest."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recorded = data.get("sha256")
        if not isinstance(recorded, str) or len(recorded) != 64:
            return False
        payload = {k: v for k, v in data.items() if k != "sha256"}
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest() == recorded
