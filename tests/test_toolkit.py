import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from fl_entity_eval import (
    EntityGroupSplitter,
    FlowerEntityPartitioner,
    LeakageAuditor,
    ICCDiagnostics,
    UnitLevelEvaluator,
    cohen_kappa,
    exact_paired_sign_flip_test,
    fixed_macro_f1,
    holm_bonferroni,
    make_pytorch_subsets,
    present_class_macro_f1,
    support_masked_head_avg,
    train_only_standardize,
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
        self.assertIn("deff_unbalanced", res)
        cr = ICCDiagnostics.compute_class_residualized_icc(
            self.df["feat"].to_numpy(), self.df["trip_id"].to_numpy(), self.df["label"].to_numpy()
        )
        self.assertGreaterEqual(cr["icc"], 0.0)

    def test_pytorch_and_flower_adapters(self):
        import torch
        from torch.utils.data import TensorDataset

        ds = TensorDataset(torch.zeros((len(self.df), 2)), torch.from_numpy(self.df["label"].to_numpy()))
        tr_sub, te_sub, tr_idx, te_idx = make_pytorch_subsets(ds, self.df, entity_col="driver_id", test_frac=0.33, seed=1)
        self.assertEqual(len(tr_sub) + len(te_sub), len(self.df))
        self.assertEqual(len(set(tr_idx).intersection(set(te_idx))), 0)

        # Stratified FlowerEntityPartitioner guarantees zero-support classes are not dropped
        rows = []
        for d in range(4):
            for c in range(3):
                for rep in range(2):
                    tid = f"D{d}_C{c}_R{rep}"
                    for w in range(5):
                        rows.append({"driver_id": f"D{d}", "trip_id": tid, "label": c})
        df_strat = pd.DataFrame(rows)
        part = FlowerEntityPartitioner(
            df_strat, client_col="driver_id", session_col="trip_id", label_col="label", test_frac=0.5, stratified=True, seed=7
        )
        self.assertEqual(len(part.client_ids), 4)
        for cid in part.client_ids:
            p = part.load_partition(cid)
            tr_cls = set(df_strat.iloc[p["train"]]["label"].unique())
            te_cls = set(df_strat.iloc[p["test"]]["label"].unique())
            self.assertTrue(te_cls.issubset(tr_cls))

    def test_strict_temporal_boundary_and_fixed_f1(self):
        # Contiguous non-overlapping windows at exact step == window_sec should NOT trigger strict overlap
        df_b = pd.DataFrame({"driver_id": ["D0", "D0"], "trip_id": ["T0", "T0"], "t": [0.0, 2.56]})
        aud_strict = LeakageAuditor.audit_split(df_b, np.array([0]), np.array([1]), driver_col="driver_id", trip_col="trip_id", time_col="t", window_sec=2.56, inclusive=False)
        aud_incl = LeakageAuditor.audit_split(df_b, np.array([0]), np.array([1]), driver_col="driver_id", trip_col="trip_id", time_col="t", window_sec=2.56, inclusive=True)
        self.assertEqual(aud_strict["temporal_proximity_hits"], 0)
        self.assertEqual(aud_incl["temporal_proximity_hits"], 1)
        self.assertAlmostEqual(fixed_macro_f1(np.array([0, 1]), np.array([0, 1]), n_classes=3), 2.0 / 3.0)

    def test_sign_flip_holm_and_kappa(self):
        res = exact_paired_sign_flip_test(np.array([0.1, 0.2, 0.15, 0.05, 0.12, 0.08]))
        self.assertAlmostEqual(res["p_value"], 2.0 / 64.0, places=6)
        adj = holm_bonferroni({"m1": 0.01, "m2": 0.04, "m3": 0.20})
        self.assertAlmostEqual(adj["m1"], 0.03, places=6)
        self.assertAlmostEqual(adj["m2"], 0.08, places=6)
        self.assertAlmostEqual(cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 1.0, places=6)

    def test_unit_evaluator(self):
        y = self.df["label"].to_numpy()
        ev = UnitLevelEvaluator.evaluate_predictions(
            y, y, self.df["driver_id"].to_numpy(), self.df["trip_id"].to_numpy(), n_classes=3
        )
        self.assertEqual(ev["n_drivers"], 6)
        self.assertEqual(ev["n_trips"], 24)
        self.assertAlmostEqual(ev["trip_vote_acc"], 1.0)
        self.assertAlmostEqual(ev["driver_worst_present_f1"], 1.0)

        # Single-class oracle: fixed_macro_f1(n_classes=3)=1/3, present_class_macro_f1=1.0
        yt_1 = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(fixed_macro_f1(yt_1, yt_1, n_classes=3), 1.0 / 3.0)
        self.assertAlmostEqual(present_class_macro_f1(yt_1, yt_1), 1.0)

    def test_train_only_standardize_error_deff_and_protocol_helper(self):
        X = np.arange(60, dtype=np.float32).reshape(6, 2, 5)
        X_scaled, mean, std = train_only_standardize(X, np.array([0, 1, 2]))
        self.assertEqual(X_scaled.shape, X.shape)
        self.assertAlmostEqual(float(X_scaled[:3].mean()), 0.0, places=5)
        self.assertTrue(np.all(std > 0))

        # Model-error DEFF
        y_t = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        y_p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        grp = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"])
        err = ICCDiagnostics.compute_error_deff(y_t, y_p, grp)
        self.assertGreater(err["deff"], 1.5)

        # Item-8 protocol sensitivity helper
        proto = UnitLevelEvaluator.evaluate_protocol_sensitivity(
            y_true_by_client={"c0": np.array([0, 0, 1, 1]), "c1": np.array([0, 0, 1, 1])},
            y_pred_deployed_by_client={"c0": np.array([0, 0, 0, 0]), "c1": np.array([0, 0, 1, 1])},
            y_pred_uniform_by_client={"c0": np.array([0, 0, 1, 1]), "c1": np.array([0, 0, 1, 1])},
            trip_by_client={"c0": np.array(["t0", "t0", "t1", "t1"]), "c1": np.array(["t2", "t2", "t3", "t3"])},
            train_classes_by_client={"c0": {0}, "c1": {0, 1}},
            n_classes=2,
        )
        self.assertAlmostEqual(proto["zero_support_trip_frac"], 0.25)
        self.assertAlmostEqual(proto["trip_acc_deployed"], 0.75)
        self.assertAlmostEqual(proto["trip_acc_uniform"], 1.0)
        self.assertAlmostEqual(proto["seen_cls_trip_acc_deployed"], 1.0)
        self.assertAlmostEqual(proto["unseen_cls_trip_acc_deployed"], 0.0)

        # trip_col isolation guard in split_size_matched_pair
        with self.assertRaises(ValueError):
            EntityGroupSplitter.split_size_matched_pair(
                self.df, "trip_id", np.array([0, 1, 2]), np.array([3, 4, 5]), seed=1
            )

    def test_rev9_chronological_manifest_unseen_client_and_guard(self):
        # Chronological split within entity
        tr_c, te_c = EntityGroupSplitter.split_chronological(
            self.df, time_col="timestamp_sec", test_frac=0.25, entity_col="trip_id"
        )
        self.assertEqual(len(tr_c) + len(te_c), len(self.df))
        for tid, grp in self.df.groupby("trip_id"):
            tr_t = self.df.iloc[np.intersect1d(tr_c, grp.index)]["timestamp_sec"].max()
            te_t = self.df.iloc[np.intersect1d(te_c, grp.index)]["timestamp_sec"].min()
            self.assertLess(tr_t, te_t)

        # Manifest round-trip verification & tamper detection
        with tempfile.TemporaryDirectory() as td:
            mpath = os.path.join(td, "manifest.json")
            EntityGroupSplitter.export_manifest(tr_c, te_c, {"split": "chrono"}, mpath)
            self.assertTrue(EntityGroupSplitter.verify_manifest(mpath))
            with open(mpath, "r", encoding="utf-8") as fh:
                txt = fh.read().replace('"chrono"', '"tampered"')
            with open(mpath, "w", encoding="utf-8") as fh:
                fh.write(txt)
            self.assertFalse(EntityGroupSplitter.verify_manifest(mpath))

        # FlowerEntityPartitioner unseen_client_frac
        part = FlowerEntityPartitioner(
            self.df, client_col="driver_id", session_col="trip_id", label_col="label",
            test_frac=0.25, stratified=False, unseen_client_frac=0.33, seed=42,
        )
        self.assertEqual(len(part.unseen_client_ids), 2)
        self.assertEqual(len(part.client_ids), 4)
        self.assertTrue(set(part.client_ids).isdisjoint(set(part.unseen_client_ids)))
        self.assertGreater(len(part.load_unseen_client(part.unseen_client_ids[0])), 0)

        # support_masked_head_avg ignores unsupported client row (-100)
        w = {"c0": np.array([[1.0, 2.0], [-100.0, -100.0]], dtype=np.float32),
             "c1": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)}
        b = {"c0": np.array([0.1, -50.0], dtype=np.float32),
             "c1": np.array([0.3, 0.5], dtype=np.float32)}
        cnts = {"c0": np.array([10, 0]), "c1": np.array([10, 8])}
        fb_w, fb_b, guarded = support_masked_head_avg(w, b, cnts)
        np.testing.assert_allclose(fb_w[1], [5.0, 6.0], rtol=1e-5)
        np.testing.assert_allclose(guarded["c0"][0][1], [5.0, 6.0], rtol=1e-5)
        self.assertAlmostEqual(float(guarded["c0"][1][1]), 0.5, places=5)

        # Colliding trip IDs across clients in evaluate_protocol_sensitivity
        proto = UnitLevelEvaluator.evaluate_protocol_sensitivity(
            y_true_by_client={"c0": np.array([0, 0]), "c1": np.array([1, 1])},
            y_pred_deployed_by_client={"c0": np.array([0, 0]), "c1": np.array([1, 1])},
            y_pred_uniform_by_client={"c0": np.array([0, 0]), "c1": np.array([1, 1])},
            trip_by_client={"c0": np.array(["trip_0", "trip_0"]), "c1": np.array(["trip_0", "trip_0"])},
            train_classes_by_client={"c0": {0}, "c1": {1}},
            n_classes=2,
        )
        self.assertAlmostEqual(proto["trip_acc_deployed"], 1.0)

        # sweep_window_overlap skips short recordings by default
        short_df = pd.DataFrame({"g": ["A", "A", "B", "B", "B", "B"], "v": [1, 2, 3, 4, 5, 6]})
        sw = ICCDiagnostics.sweep_window_overlap(short_df, "v", "g", [4], [0.0])
        self.assertEqual(int(sw["n_obs"].iloc[0]), 1)

    def test_rev10_non_range_index_and_degenerate_icc(self):
        # Filtered DataFrame with non-contiguous index must still yield valid 0..N-1 positional indices
        df_sub = self.df[self.df["driver_id"].isin(["D1", "D3"])].copy()
        self.assertNotEqual(int(df_sub.index[0]), 0)
        tr_c, te_c = EntityGroupSplitter.split_chronological(
            df_sub, time_col="timestamp_sec", test_frac=0.25, entity_col="trip_id"
        )
        self.assertEqual(len(tr_c) + len(te_c), len(df_sub))
        self.assertLess(int(tr_c.max()), len(df_sub))
        self.assertLess(int(te_c.max()), len(df_sub))
        aud = LeakageAuditor.audit_split(df_sub, tr_c, te_c, driver_col="driver_id", trip_col="trip_id")
        self.assertEqual(aud["index_overlap"], 0)

        # Degenerate ICC branch must return complete schema including unbalanced keys
        deg = ICCDiagnostics.compute_icc_and_neff(np.array([1.0, 2.0, 3.0]), np.array(["G0", "G0", "G0"]))
        for k in ("mean_cluster_size", "cv_cluster_size", "deff_unbalanced", "n_eff_unbalanced", "neff_ratio_unbalanced"):
            self.assertIn(k, deg)


if __name__ == "__main__":
    unittest.main()
