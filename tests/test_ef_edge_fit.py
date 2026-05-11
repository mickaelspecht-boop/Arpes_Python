from __future__ import annotations

import unittest

import numpy as np

from arpes.ui.widgets.plots.common import _robust_ef_polyfit


class TestRobustEfPolyfit(unittest.TestCase):
    def test_uses_reasonable_weighting_with_tiny_errors(self):
        k = np.linspace(-1.6, 1.6, 101)
        ef = 0.003 * np.sin(2.0 * k) - 0.002
        err = np.full_like(k, 0.010)

        # Simule quelques incertitudes numeriquement trop optimistes. Avec
        # w=1/err^2, ces points peuvent imposer une parabole absurde aux bords.
        edge = np.r_[0:4, -4:0]
        ef[edge] += np.r_[0.035, 0.025, -0.020, -0.030, -0.025, -0.015, 0.020, 0.030]
        err[edge] = 1e-4

        coefs, smooth = _robust_ef_polyfit(k, ef, err, poly_deg=2)

        self.assertEqual(coefs.shape, (3,))
        self.assertLess(np.nanmax(np.abs(smooth)), 0.020)
        self.assertLess(abs(np.nanmedian(smooth) - np.nanmedian(ef)), 0.006)


if __name__ == "__main__":
    unittest.main()
