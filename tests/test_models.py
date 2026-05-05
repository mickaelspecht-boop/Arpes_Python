from __future__ import annotations

import unittest

from arpes_models import (
    FitGammaSummary,
    LoadContext,
    MetadataSource,
    ResolutionInfo,
)


class TestRuntimeModels(unittest.TestCase):
    def test_resolution_info_accepts_current_fit_result_shape(self):
        info = ResolutionInfo.from_dict({
            "dE_eV": 0.025,
            "dk_inv_a": 0.006,
            "source": "estime PE=50 DA30",
        })
        self.assertAlmostEqual(info.dE_meV, 25.0)
        self.assertAlmostEqual(info.dE_eV, 0.025)
        self.assertEqual(info.to_dict()["source"], "estime PE=50 DA30")

    def test_resolution_info_defaults_for_legacy_missing_dict(self):
        info = ResolutionInfo.from_dict({})
        self.assertAlmostEqual(info.dE_meV, 15.0)
        self.assertAlmostEqual(info.dk_inv_a, 0.005)
        self.assertEqual(info.source, "default")

    def test_metadata_source_accepts_legacy_raw_value(self):
        src = MetadataSource.from_dict(48.0, key="hv")
        self.assertEqual(src.key, "hv")
        self.assertEqual(src.value, 48.0)
        self.assertEqual(src.source, "unknown")

    def test_metadata_source_round_trip(self):
        src = MetadataSource(
            key="pol",
            value="LH",
            source="logbook",
            detail="Light Polarization",
        )
        restored = MetadataSource.from_dict(src.to_dict())
        self.assertEqual(restored, src)

    def test_fit_gamma_summary_uses_gamma_alias_for_old_results(self):
        summary = FitGammaSummary.from_fit_result({
            "gamma": [[0.04, 0.06, None], [0.02]],
        })
        self.assertAlmostEqual(summary.gamma_brut_median, 0.04)
        self.assertIsNone(summary.gamma_corrige_median)
        self.assertFalse(summary.resolution_limited)

    def test_fit_gamma_summary_marks_resolution_limited(self):
        summary = FitGammaSummary.from_fit_result({
            "gamma_brut": [[0.10, 0.12]],
            "gamma_corrige": [[0.01, 0.02]],
            "gamma_min": [[0.09, 0.10]],
        })
        self.assertTrue(summary.resolution_limited)
        self.assertAlmostEqual(summary.gamma_min_median, 0.095)

    def test_load_context_round_trip(self):
        ctx = LoadContext(
            hv=100.0,
            temperature=20.0,
            azi=45.0,
            pol="LH",
            angle_offsets={"theta0_deg": 0.2},
            bessy_energy_reference="ses_center_energy",
        )
        restored = LoadContext.from_dict(ctx.to_dict())
        self.assertEqual(restored, ctx)


if __name__ == "__main__":
    unittest.main()
