#!/usr/bin/env python3
"""Compute inter-rater Cohen's kappa across dual-coded columns and 2020-2026 temporal trends."""

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl_entity_eval import cohen_kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Tier1_Master_57_FullText_Corpus.csv after Coder 2 fills columns")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    pairs = [
        ("split_granularity_code", "coder2_split_granularity"),
        ("unseen_entity_holdout", "coder2_unseen_entity_holdout"),
        ("stats_or_seeds_code", "coder2_stats_or_seeds"),
    ]
    print("=== INTER-RATER RELIABILITY (COHEN'S KAPPA) ===")
    for c1, c2 in pairs:
        sub = df[[c1, c2]].dropna()
        sub = sub[(sub[c1].astype(str).str.strip() != "") & (sub[c2].astype(str).str.strip() != "")]
        if len(sub) == 0:
            print(f"  {c1}: Coder 2 column is still empty.")
            continue
        a = sub[c1].astype(str).str.strip().to_numpy()
        b = sub[c2].astype(str).str.strip().to_numpy()
        agree = float((a == b).mean())
        kappa = cohen_kappa(a, b)
        print(f"  {c1:28s} | n={len(sub):2d} | raw_agreement={agree:.3f} | Cohen_kappa={kappa:.3f}")

    print("\n=== TEMPORAL EVOLUTION BY YEAR ===")
    if "year" in df.columns:
        grp = df.groupby("year").agg(
            n_papers=("orig_ref_id", "count"),
            unseen_entity_rate=("unseen_entity_holdout", lambda s: (s == "Yes").mean()),
            unspecified_split_rate=("split_granularity_code", lambda s: (s == "Unspecified").mean()),
            formal_test_rate=("stats_or_seeds_code", lambda s: (s == "Formal_Test").mean()),
        )
        print(grp.round(3).to_string())


if __name__ == "__main__":
    main()
