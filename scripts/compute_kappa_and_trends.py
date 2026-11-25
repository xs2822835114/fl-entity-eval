#!/usr/bin/env python3
"""Compute inter-rater Cohen's kappa (when Coder 2 columns are present),
2020-2026 temporal evolution, and full vs. pure road-vehicle FL audit summaries.

Works out-of-the-box with `data/full_text_audit_en.csv` as well as legacy dual-coded CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fl_entity_eval import cohen_kappa  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "data" / "full_text_audit_en.csv"),
        help="Path to audit CSV (defaults to data/full_text_audit_en.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig", keep_default_na=False)
    id_col = "record_id" if "record_id" in df.columns else "orig_ref_id"
    if "included_in_56_audit" in df.columns:
        df = df[df["included_in_56_audit"].astype(str).str.lower() == "true"].copy()

    pairs = [
        ("split_granularity_code", "coder2_split_granularity"),
        ("unseen_entity_holdout", "coder2_unseen_entity_holdout"),
        ("stats_or_seeds_code", "coder2_stats_or_seeds"),
    ]
    print("=== INTER-RATER RELIABILITY (COHEN'S KAPPA) ===")
    has_coder2 = any(c2 in df.columns for _, c2 in pairs)
    if not has_coder2:
        print("  Single-coder verified audit CSV detected (no coder2_* columns present).")
    else:
        for c1, c2 in pairs:
            if c1 not in df.columns or c2 not in df.columns:
                continue
            sub = df[[c1, c2]].dropna()
            sub = sub[(sub[c1].astype(str).str.strip() != "") & (sub[c2].astype(str).str.strip() != "")]
            if len(sub) == 0:
                print(f"  {c1}: Coder 2 column is empty.")
                continue
            a = sub[c1].astype(str).str.strip().to_numpy()
            b = sub[c2].astype(str).str.strip().to_numpy()
            agree = float((a == b).mean())
            kappa = cohen_kappa(a, b)
            print(f"  {c1:28s} | n={len(sub):2d} | raw_agreement={agree:.3f} | Cohen_kappa={kappa:.3f}")

    sup_col = (
        "is_supervised_split_applicable"
        if "is_supervised_split_applicable" in df.columns
        else ("included_in_49_supervised" if "included_in_49_supervised" in df.columns else None)
    )
    if sup_col is not None:
        sup = df[df[sup_col].astype(str).str.lower() == "true"]
    else:
        sup = df

    print(f"\n=== OVERALL AUDIT SUMMARY (N={len(df)}, N_sup={len(sup)}) ===")
    print(f"  Unseen entity holdout (of N_sup): {(sup['unseen_entity_holdout'] == 'Yes').sum()}/{len(sup)} "
          f"({100.0 * (sup['unseen_entity_holdout'] == 'Yes').mean():.1f}%)")
    unspec_mask = sup["split_granularity_code"].isin(["Unspecified", "Unspecified_Supervised"])
    print(f"  Unspecified split (of N_sup):     {unspec_mask.sum()}/{len(sup)} "
          f"({100.0 * unspec_mask.mean():.1f}%)")
    print(f"  Formal statistical test (of N):   {(df['stats_or_seeds_code'] == 'Formal_Test').sum()}/{len(df)} "
          f"({100.0 * (df['stats_or_seeds_code'] == 'Formal_Test').mean():.1f}%)")
    any_ms = df["stats_or_seeds_code"] != "None"
    print(f"  Multi-run/seeds or formal (of N): {any_ms.sum()}/{len(df)} "
          f"({100.0 * any_ms.mean():.1f}%)")
    if "reports_dispersion_or_formal_test" in df.columns:
        disp = df["reports_dispersion_or_formal_test"].astype(str).str.lower() == "true"
        print(f"  Dispersion/formal test (of N):    {disp.sum()}/{len(df)} "
              f"({100.0 * disp.mean():.1f}%)")

    if "pure_road_vehicle_fl" in df.columns:
        pure_all = df[df["pure_road_vehicle_fl"].astype(str).str.lower() == "true"]
        pure_sup = df[df["pure_road_supervised_fl"].astype(str).str.lower() == "true"]
        p_unspec = pure_sup["split_granularity_code"].isin(["Unspecified", "Unspecified_Supervised"])
        print(f"\n=== VEHICULAR/TRAFFIC-DOMAIN FL SENSITIVITY (N={len(pure_all)}, N_sup={len(pure_sup)}) ===")
        print(f"  Unseen entity holdout (of N_sup): {(pure_sup['unseen_entity_holdout'] == 'Yes').sum()}/{len(pure_sup)} "
              f"({100.0 * (pure_sup['unseen_entity_holdout'] == 'Yes').mean():.1f}%)")
        print(f"  Unspecified split (of N_sup):     {p_unspec.sum()}/{len(pure_sup)} "
              f"({100.0 * p_unspec.mean():.1f}%)")
        print(f"  Formal statistical test (of N):   {(pure_all['stats_or_seeds_code'] == 'Formal_Test').sum()}/{len(pure_all)} "
              f"({100.0 * (pure_all['stats_or_seeds_code'] == 'Formal_Test').mean():.1f}%)")
        if "reports_dispersion_or_formal_test" in pure_all.columns:
            p_disp = pure_all["reports_dispersion_or_formal_test"].astype(str).str.lower() == "true"
            print(f"  Dispersion/formal test (of N):    {p_disp.sum()}/{len(pure_all)} "
                  f"({100.0 * p_disp.mean():.1f}%)")

    print("\n=== TEMPORAL EVOLUTION BY YEAR (N_all / N_sup) ===")
    if "year" in df.columns:
        rows = []
        for yr, g_all in df.groupby("year", sort=True):
            g_sup = sup[sup["year"] == yr]
            ns = len(g_sup)
            rows.append({
                "year": int(yr),
                "n_all": len(g_all),
                "n_sup": ns,
                "unseen_entity_rate_sup": float((g_sup["unseen_entity_holdout"] == "Yes").mean()) if ns else 0.0,
                "unspecified_split_rate_sup": float(g_sup["split_granularity_code"].isin(["Unspecified", "Unspecified_Supervised"]).mean()) if ns else 0.0,
                "formal_test_rate_all": float((g_all["stats_or_seeds_code"] == "Formal_Test").mean()),
            })
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
