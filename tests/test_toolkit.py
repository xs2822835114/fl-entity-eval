import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from fl_entity_eval import (
    EntityGroupSplitter,
    LeakageAuditor,
    ICCDiagnostics,
    UnitLevelEvaluator,
    cohen_kappa,
    exact_paired_sign_flip_test,
    holm_bonferroni,
)


class TestFLEntityEval(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        rows = []
        for d in range(6):
            for t in range(4):
                tid = f"D{d}_T{t}"
                for w in range(20):
                    rows.append({
                        "driver_id": f"D{d}",
                        "trip_id": tid,
                        "timestamp_sec": w * 15.0,
                        "feat": d * 2.0 + t * 0.5 + rng.normal(0, 0.1),
                        "label": t % 3,
                    })
        self.df = pd.DataFrame(rows)

    def test_size_matched_and_auditor(self):
        s2_tr, s2_te = EntityGroupSplitter.split_whole_entity(self.df, "trip_id", test_frac=0.25, stratify_col="driver_id", seed=42)
        splits = EntityGroupSplitter.split_size_matched_pair(self.df, "trip_id", s2_tr, s2_te, s1_test_frac=0.2, seed=42)
        self.assertEqual(len(splits["S1_matched"][1]), len(splits["S2_matched"][1]))
        aud_s1 = LeakageAuditor.audit_split(self.df, *splits["S1"], driver_col="driver_id", trip_col="trip_id", time_col="timestamp_sec", window_sec=30.0)
        aud_s2 = LeakageAuditor.audit_split(self.df, *splits["S2"], driver_col="driver_id", trip_col="trip_id", time_col="timestamp_sec", window_sec=30.0)
        self.assertEqual(aud_s1["audit_verdict"], "FAIL_TRIP_LEAKAGE")
        self.assertGreater(aud_s1["trip_overlap_rate"], 0.9)
        self.assertEqual(aud_s2["trip_overlap_rate"], 0.0)
        self.assertEqual(aud_s2["audit_verdict"], "WARN_WITHIN_DRIVER_ONLY")

        d_tr, d_te = EntityGroupSplitter.split_whole_entity(self.df, "driver_id", test_frac=0.33, seed=42)
        aud_d = LeakageAuditor.audit_split(self.df, d_tr, d_te, driver_col="driver_id", trip_col="trip_id")
        self.assertEqual(aud_d["audit_verdict"], "PASS_ENTITY_ISOLATED")

        with tempfile.TemporaryDirectory() as td:
            sha = EntityGroupSplitter.export_manifest(d_tr, d_te, {"test": True}, os.path.join(td, "m.json"))
            self.assertEqual(len(sha), 64)

    def test_icc_and_neff(self):
        res = ICCDiagnostics.compute_icc_and_neff(self.df["feat"].to_numpy(), self.df["trip_id"].to_numpy())
        self.assertGreater(res["icc"], 0.9)
        self.assertLess(res["n_eff"], res["n_obs"] * 0.2)

    def test_sign_flip_holm_and_kappa(self):
        res = exact_paired_sign_flip_test(np.array([0.1, 0.2, 0.15, 0.05, 0.12, 0.08]))
        self.assertAlmostEqual(res["p_value"], 2.0 / 64.0, places=6)
        adj = holm_bonferroni({"m1": 0.01, "m2": 0.04, "m3": 0.20})
        self.assertAlmostEqual(adj["m1"], 0.03, places=6)
        self.assertAlmostEqual(adj["m2"], 0.08, places=6)
        self.assertAlmostEqual(cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 1.0, places=6)

    def test_unit_evaluator(self):
        y = self.df["label"].to_numpy()
        ev = UnitLevelEvaluator.evaluate_predictions(y, y, self.df["driver_id"].to_numpy(), self.df["trip_id"].to_numpy())
        self.assertEqual(ev["n_drivers"], 6)
        self.assertEqual(ev["n_trips"], 24)
        self.assertAlmostEqual(ev["trip_vote_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
