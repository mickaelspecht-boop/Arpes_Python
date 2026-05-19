from __future__ import annotations

import unittest

import numpy as np

from arpes.analysis.self_energy import real_self_energy
from arpes.theory.models import TheoryBandData, TheoryOverlayConfig


class TestSelfEnergy(unittest.TestCase):
    def test_real_self_energy_recovers_known_offset(self):
        data = TheoryBandData(
            source="test",
            material_id="mp-test",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[[-0.4, 0.0, 0.4]],
        )
        cfg = TheoryOverlayConfig(enabled=True)
        k = np.linspace(-0.5, 0.5, 7)
        e_dft = 0.4 * k
        offset = 0.035
        fit = {
            "e_fitted": (e_dft + offset).tolist(),
            "kF_plus": [k.tolist()],
            "kF_minus": [],
        }
        overlay = {"data": data.to_dict(), "config": cfg.to_dict()}

        out = real_self_energy(fit, overlay, min_points=4)

        self.assertEqual(out.band_index, 0)
        self.assertEqual(out.branch, "kF_plus")
        np.testing.assert_allclose(out.re_sigma, offset, atol=1e-12)
        self.assertAlmostEqual(out.rms_e, offset)

    def test_real_self_energy_uses_displayed_branch_axis(self):
        data = TheoryBandData(
            source="materials_project",
            material_id="mp-test",
            k_distance=[0.0, 1.0, 2.0, 3.0],
            bands=[[9.0, -0.2, 0.0, 0.2]],
            branches=[{"name": "\\Gamma-X", "start": 1, "end": 3}],
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X")
        fit = {
            "e_fitted": [-0.2, 0.0, 0.2],
            "kF_plus": [[0.0, 0.5, 1.0]],
            "kF_minus": [],
        }
        overlay = {"data": data.to_dict(), "config": cfg.to_dict()}

        out = real_self_energy(fit, overlay, min_points=3)

        np.testing.assert_allclose(out.e_dft, [-0.2, 0.0, 0.2])
        np.testing.assert_allclose(out.re_sigma, 0.0, atol=1e-12)

    def test_real_self_energy_requires_overlay_and_fit(self):
        with self.assertRaises(ValueError):
            real_self_energy({}, {})
        with self.assertRaises(ValueError):
            real_self_energy({}, {"data": TheoryBandData("test", "mp").to_dict()})


if __name__ == "__main__":
    unittest.main()
