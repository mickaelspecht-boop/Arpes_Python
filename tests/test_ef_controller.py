"""Tests for the EF controller (`arpes_ef_controller`).

These tests pin the scientific decisions extracted from the UI: scalar/poly
calibration, per-file propagation, and reference guards.
"""

from __future__ import annotations

import unittest

from arpes.physics.ef_calibration import (
    ReferenceError,
    already_applied,
    apply_reference_to_target,
    axis_zero_in_kinetic,
    compute_calibration_update,
)


class TestAxisZeroInKinetic(unittest.TestCase):
    def test_prefers_nominal(self):
        self.assertEqual(
            axis_zero_in_kinetic({"ef_kinetic_nominal": 43.4, "ef_kinetic_from_hv": 75.0}),
            43.4,
        )

    def test_fallback_to_hv(self):
        self.assertEqual(axis_zero_in_kinetic({"ef_kinetic_from_hv": "75.5"}), 75.5)

    def test_missing_or_bad_returns_none(self):
        self.assertIsNone(axis_zero_in_kinetic({}))
        self.assertIsNone(axis_zero_in_kinetic({"ef_kinetic_nominal": "bad"}))


class TestCalibrationUpdate(unittest.TestCase):
    def test_scalar_update_and_reference_payload(self):
        payload = {
            "mode": "scalar",
            "ef_shift": 0.025,
            "T": 20.0,
            "fwhm_res": 0.012,
        }
        out = compute_calibration_update(
            payload,
            current_ef_offset=0.052,
            source_meta={"ef_kinetic_nominal": 43.4, "energy_reference": "ses_center_energy"},
            source_path="/data/Au.ibw",
        )
        self.assertAlmostEqual(out.new_ef_offset, 0.027)
        self.assertEqual(out.ef_correction, {})
        self.assertEqual(out.ref_payload["mode"], "scalar")
        self.assertAlmostEqual(out.ref_payload["source_ef_kin_nominal"], 43.4)
        self.assertEqual(out.ref_payload["source_energy_reference"], "ses_center_energy")
        self.assertIn("Scalar EF", out.msg)

    def test_poly_update_forces_zero_offset(self):
        payload = {
            "mode": "poly",
            "poly_coefs": [0.01, -0.02],
            "k_min": -0.5,
            "k_max": 0.5,
            "T": 30.0,
            "fwhm_res": 0.020,
            "rms": 0.003,
            "n_valid": 42,
        }
        out = compute_calibration_update(
            payload,
            current_ef_offset=0.052,
            source_meta={},
            source_path="/data/Au.ibw",
        )
        self.assertAlmostEqual(out.new_ef_offset, 0.0)
        self.assertEqual(out.ef_correction["mode"], "poly")
        self.assertEqual(out.ef_correction["source"], "self")
        self.assertEqual(out.ref_payload, out.ef_correction)
        self.assertIn("Per-column EF", out.msg)


class TestReferenceApplication(unittest.TestCase):
    def test_already_applied_sources(self):
        self.assertTrue(already_applied({"source": "reference"}))
        self.assertTrue(already_applied({"source": "reference_scalar"}))
        self.assertFalse(already_applied({"source": "self"}))
        self.assertFalse(already_applied({}))

    def test_poly_reference(self):
        ref = {"mode": "poly", "poly_coefs": [1.0], "source": "self"}
        out = apply_reference_to_target(
            ref,
            current_ef_offset=0.052,
            target_meta={},
            ref_path_str="Au.ibw",
        )
        self.assertAlmostEqual(out.new_ef_offset, 0.0)
        self.assertEqual(out.ef_correction["source"], "reference")
        self.assertIn("Poly", out.msg)

    def test_scalar_reference_naive_when_kinetics_missing(self):
        ref = {"mode": "scalar", "ef_shift": 0.020, "source_file": "/data/Au.ibw"}
        out = apply_reference_to_target(
            ref,
            current_ef_offset=0.052,
            target_meta={},
            ref_path_str="Au.ibw",
        )
        self.assertAlmostEqual(out.new_ef_offset, 0.032)
        self.assertAlmostEqual(out.ef_correction["ref_effective_shift"], 0.020)
        self.assertIn("ef_kin_nominal missing", out.msg)

    def test_scalar_reference_per_file_when_modes_match(self):
        ref = {
            "mode": "scalar",
            "ef_shift": 0.020,
            "source_file": "/data/Au.ibw",
            "source_ef_kin_nominal": 43.4,
            "source_energy_reference": "ses_center_energy",
        }
        out = apply_reference_to_target(
            ref,
            current_ef_offset=0.0,
            target_meta={"ef_kinetic_nominal": 43.0, "energy_reference": "ses_center_energy"},
            ref_path_str="Au.ibw",
        )
        # effective shift = 0.020 + (43.4 - 43.0) = 0.420 eV
        self.assertAlmostEqual(out.new_ef_offset, -0.420)
        self.assertAlmostEqual(out.ef_correction["ref_kinetic_correction"], 0.4)
        self.assertIn("per-file", out.msg)

    def test_scalar_reference_warns_when_energy_modes_differ(self):
        ref = {
            "mode": "scalar",
            "ef_shift": 0.020,
            "source_ef_kin_nominal": 43.4,
            "source_energy_reference": "ses_center_energy",
        }
        out = apply_reference_to_target(
            ref,
            current_ef_offset=0.0,
            target_meta={"ef_kinetic_nominal": 43.0, "energy_reference": "hv_minus_work_function"},
            ref_path_str="Au.ibw",
        )
        self.assertAlmostEqual(out.new_ef_offset, -0.020)
        self.assertIn("different energy modes", out.msg)

    def test_bad_reference_mode_raises(self):
        with self.assertRaises(ReferenceError):
            apply_reference_to_target(
                {"mode": "bad"},
                current_ef_offset=0.0,
                target_meta={},
                ref_path_str="Au.ibw",
            )


if __name__ == "__main__":
    unittest.main()
