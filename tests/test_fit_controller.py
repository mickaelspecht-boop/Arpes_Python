from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.fit import MdcFitter
from arpes.core.session import FileEntry, FitParams


class FakeAP:
    def __init__(self):
        self.calls = []

    def fit_mdc_peak_pairs(self, data, kpar, ev, **kwargs):
        self.calls.append((data, kpar, ev, kwargs))
        return {
            "e_fitted": np.array([-0.1, 0.0]),
            "kF_minus": [np.array([0.1, 0.2])],
            "xg": np.array([0.0, 0.02]),
            "gamma_brut": [np.array([0.10, 0.12])],
            "gamma_corrige": [np.array([0.01, 0.02])],
            "gamma_min": [np.array([0.09, 0.10])],
        }


class TestFitController(unittest.TestCase):
    def test_fit_kwargs_from_fit_params(self):
        fp = FitParams(
            n_pairs=2,
            ev_start=-0.5,
            ev_end=-0.1,
            pairs=[
                {"kF_init": 0.2, "gamma_init": 0.05, "gamma_max": 0.2},
                {"kF_init": 0.4, "gamma_init": 0.05, "gamma_max": 0.2},
            ],
            dE_meV=25.0,
            dk_inv_a=0.006,
        )
        kwargs = MdcFitter.fit_kwargs(fp, resolution_source="estime")
        self.assertEqual(kwargs["n_pairs"], 2)
        self.assertEqual(kwargs["kF_init"], [0.2, 0.4])
        self.assertAlmostEqual(kwargs["dE_eV"], 0.025)
        self.assertEqual(kwargs["dk_inv_a"], 0.006)
        self.assertEqual(kwargs["resolution_source"], "estime")
        self.assertFalse(kwargs["verbose"])

    def test_run_full_fit_calls_arpes_plots(self):
        ap = FakeAP()
        fp = FitParams(dE_meV=20.0)
        fr = MdcFitter(ap).run_full_fit(
            np.zeros((2, 2)),
            np.array([0.0, 1.0]),
            np.array([-0.1, 0.0]),
            fp,
            resolution_source="manual",
        )
        self.assertIn("e_fitted", fr)
        self.assertEqual(ap.calls[0][3]["resolution_source"], "manual")
        self.assertAlmostEqual(ap.calls[0][3]["dE_eV"], 0.020)

    def test_summarize_marks_resolution_limited(self):
        fr = {
            "e_fitted": [-0.1, 0.0],
            "kF_minus": [[0.1, np.nan]],
            "xg": [0.0, 0.02],
            "gamma_brut": [[0.10, 0.12]],
            "gamma_corrige": [[0.01, 0.02]],
        }
        summary = MdcFitter.summarize(fr)
        self.assertEqual(summary.n_points, 2)
        self.assertEqual(summary.n_ok, 1)
        self.assertTrue(summary.resolution_dominates)
        self.assertIn("Γ med", summary.label_text)

    def test_update_entry_after_fit(self):
        entry = FileEntry()
        fp = FitParams(n_pairs=3)
        MdcFitter.update_entry_after_fit(
            entry,
            fp,
            ef_offset=0.01,
            edcnorm=False,
            view_mode="Raw",
            hv=48.0,
        )
        self.assertIs(entry.fit_params, fp)
        self.assertEqual(entry.ef_offset, 0.01)
        self.assertFalse(entry.edcnorm)
        self.assertEqual(entry.view_mode, "Raw")
        self.assertEqual(entry.meta.hv, 48.0)


if __name__ == "__main__":
    unittest.main()


class TestParamsHash(unittest.TestCase):
    def test_hash_stable_same_params(self):
        from arpes.physics.fit import compute_fit_params_hash
        fp = FitParams()
        h1 = compute_fit_params_hash(fp, ef_offset=0.05, view_mode="Raw", hv=100.0)
        h2 = compute_fit_params_hash(fp, ef_offset=0.05, view_mode="Raw", hv=100.0)
        self.assertEqual(h1, h2)

    def test_hash_changes_with_ef(self):
        from arpes.physics.fit import compute_fit_params_hash
        fp = FitParams()
        h1 = compute_fit_params_hash(fp, ef_offset=0.05, view_mode="Raw")
        h2 = compute_fit_params_hash(fp, ef_offset=0.06, view_mode="Raw")
        self.assertNotEqual(h1, h2)

    def test_hash_changes_with_distortion(self):
        from arpes.physics.fit import compute_fit_params_hash
        fp = FitParams()
        h1 = compute_fit_params_hash(fp, bm_distortion={"enabled": False})
        h2 = compute_fit_params_hash(fp, bm_distortion={"enabled": True,
                                                          "trapezoid": {"slope_left": 0.1}})
        self.assertNotEqual(h1, h2)

    def test_hash_changes_with_fp_field(self):
        from arpes.physics.fit import compute_fit_params_hash
        fp1 = FitParams()
        fp2 = FitParams(smooth_fit=fp1.smooth_fit + 1.0)
        self.assertNotEqual(
            compute_fit_params_hash(fp1),
            compute_fit_params_hash(fp2),
        )
