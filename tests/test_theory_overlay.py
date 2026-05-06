from __future__ import annotations

import unittest

import numpy as np

from arpes.theory.models import (
    TheoryBandData,
    TheoryOverlayConfig,
    compare_fit_to_theory,
    filter_bands_for_view,
    normalize_direction_label,
    segment_from_direction,
)


class TestTheoryOverlayModels(unittest.TestCase):
    def test_direction_normalization(self):
        self.assertEqual(normalize_direction_label("Gamma-X"), "Γ-X")
        self.assertEqual(normalize_direction_label("G-M"), "Γ-M")
        self.assertEqual(normalize_direction_label("g m"), "Γ-M")

    def test_segment_from_direction_matches_labels(self):
        labels = [{"label": "Γ", "k": -1.0}, {"label": "X", "k": 0.0}, {"label": "M", "k": 1.0}]
        self.assertEqual(segment_from_direction("Gamma-X", labels), "Γ-X")
        self.assertEqual(segment_from_direction("M-X", labels), "X-M")
        self.assertEqual(segment_from_direction("K-M", labels), "")

    def test_filter_bands_transforms_and_limits(self):
        data = TheoryBandData(
            source="test",
            material_id="mp-test",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[
                [-0.1, 0.0, 0.1],
                [3.0, 3.1, 3.2],
                [-0.2, -0.1, 0.0],
            ],
        )
        cfg = TheoryOverlayConfig(
            enabled=True,
            energy_shift=0.1,
            k_shift=0.2,
            k_scale=2.0,
            max_bands=1,
        )
        curves = filter_bands_for_view(data, cfg, xlim=(-3, 3), ylim=(-0.5, 0.5))
        self.assertEqual(len(curves), 1)
        k, e = curves[0]
        np.testing.assert_allclose(k, [-1.8, 0.2, 2.2])
        self.assertTrue(np.nanmax(e) <= 0.3)

    def test_filter_masks_outside_selected_segment(self):
        data = TheoryBandData(
            source="test",
            material_id="mp-test",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[[-0.1, 0.0, 0.1]],
            labels=[{"label": "Γ", "k": -1.0}, {"label": "X", "k": 0.0}],
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X", max_bands=1)
        curves = filter_bands_for_view(data, cfg, xlim=(-2, 2), ylim=(-1, 1))
        self.assertEqual(len(curves), 1)
        self.assertTrue(np.isnan(curves[0][1][-1]))

    def test_compare_fit_to_theory_scores_best_band(self):
        data = TheoryBandData(
            source="test",
            material_id="mp-test",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[
                [-0.2, 0.0, 0.2],
                [1.0, 1.0, 1.0],
            ],
        )
        cfg = TheoryOverlayConfig(enabled=True)
        fit = {
            "e_fitted": [-0.2, 0.0, 0.2],
            "kF_plus": [[-1.0, 0.0, 1.0]],
            "kF_minus": [],
        }
        out = compare_fit_to_theory(data, cfg, fit, min_points=2)
        self.assertEqual(out[0]["band_index"], 0)
        self.assertAlmostEqual(out[0]["rms_e"], 0.0)
        self.assertEqual(out[0]["n_points"], 3)

    def test_compare_fit_to_theory_requires_overlap(self):
        data = TheoryBandData(
            source="test",
            material_id="mp-test",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[[-0.2, 0.0, 0.2]],
        )
        fit = {
            "e_fitted": [-0.2, 0.0, 0.2],
            "kF_plus": [[4.0, 5.0, 6.0]],
        }
        self.assertEqual(compare_fit_to_theory(data, TheoryOverlayConfig(enabled=True), fit, min_points=2), [])


if __name__ == "__main__":
    unittest.main()
