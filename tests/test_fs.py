"""Tests filet sécurité avant split arpes/physics/fs.py.

Couvre les helpers purs (FSParams, _robust_norm, extract_fs_map) pour détecter
toute régression sémantique au moment du déplacement Qt → ui/widgets/fs.py.
"""
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from arpes.physics.fs import FSParams, _robust_norm, extract_fs_map
    HAS_FS = True
except Exception:  # pragma: no cover
    HAS_FS = False


@unittest.skipUnless(HAS_FS, "arpes.physics.fs indisponible")
class TestFSParamsDefaults(unittest.TestCase):
    def test_defaults(self):
        p = FSParams()
        self.assertAlmostEqual(p.a_lattice, 3.96)
        self.assertAlmostEqual(p.b_lattice, 3.96)
        self.assertAlmostEqual(p.ef_window, 0.030)
        self.assertEqual(p.cmap, "inferno")
        self.assertTrue(p.normalize_profile)
        self.assertTrue(p.overlay_bz)


@unittest.skipUnless(HAS_FS, "arpes.physics.fs indisponible")
class TestRobustNorm(unittest.TestCase):
    def test_constant_image(self):
        img = np.full((10, 20), 5.0)
        out = _robust_norm(img)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_all_nan(self):
        img = np.full((5, 5), np.nan)
        out = _robust_norm(img)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.all(np.isnan(out)))

    def test_normal_image_clipped_0_1(self):
        rng = np.random.default_rng(42)
        img = rng.normal(size=(40, 40)) + np.linspace(-1, 1, 40)[None, :]
        out = _robust_norm(img)
        self.assertGreaterEqual(out.min(), 0.0 - 1e-9)
        self.assertLessEqual(out.max(), 1.0 + 1e-9)


def _make_kxky_volume(n_kx=20, n_ky=18, n_e=12, ef_idx=6, peak=(0.0, 0.0)):
    kx = np.linspace(-1.0, 1.0, n_kx)
    ky = np.linspace(-0.8, 0.8, n_ky)
    ev = np.linspace(-0.150, 0.060, n_e)
    ev = ev - ev[ef_idx]  # EF=0
    KX, KY = np.meshgrid(kx, ky)
    px, py = peak
    fs_at_ef = np.exp(-((KX - px) ** 2 + (KY - py) ** 2) / (2 * 0.2 ** 2))
    vol = np.zeros((n_ky, n_kx, n_e))
    for i, e in enumerate(ev):
        vol[:, :, i] = fs_at_ef * np.exp(-(e ** 2) / (2 * 0.030 ** 2))
    return kx, ky, ev, vol


@unittest.skipUnless(HAS_FS, "arpes.physics.fs indisponible")
class TestExtractFSMap(unittest.TestCase):
    def test_kxky_nominal(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        params = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)
        kx_o, ky_o, fs, title = extract_fs_map(raw, params)
        self.assertEqual(fs.shape, (len(ky), len(kx)))
        self.assertTrue(np.all(fs >= 0) and np.all(fs <= 1.0 + 1e-9))
        self.assertIn("synthetic", title)

    def test_fallback_BM_when_no_volume(self):
        kx = np.linspace(-1.0, 1.0, 30)
        ev = np.linspace(-0.2, 0.1, 50)
        data = np.exp(-(kx[:, None] ** 2 + ev[None, :] ** 2) / 0.05)
        raw = {"data": data, "kpar": kx, "ev_arr": ev, "metadata": {}}
        params = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)
        kx_o, ky_o, fs, title = extract_fs_map(raw, params)
        self.assertEqual(ky_o.size, 1)
        self.assertEqual(fs.shape, (1, len(kx)))
        self.assertIn("MDC", title)

    def test_normalize_profile_changes_output(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        p_off = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)
        p_on = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=True,
                        norm_ref_lo=-0.150, norm_ref_hi=-0.050)
        _, _, fs_off, t_off = extract_fs_map(raw, p_off)
        _, _, fs_on, t_on = extract_fs_map(raw, p_on)
        self.assertIn("sans norm", t_off)
        self.assertNotIn("sans norm", t_on)
        self.assertEqual(fs_off.shape, fs_on.shape)

    def test_invalid_volume_shape_raises(self):
        bad = {
            "data": np.zeros((5, 5)), "kpar": np.linspace(-1, 1, 5),
            "ev_arr": np.linspace(-0.1, 0.1, 5),
            "metadata": {
                "fs_data": np.zeros((3, 3)),  # 2D au lieu de 3D
                "fs_kx": np.zeros(3), "fs_ky": np.zeros(3), "fs_energy": np.zeros(3),
                "fs_kind": "kxky",
            },
        }
        with self.assertRaises(ValueError):
            extract_fs_map(bad, FSParams())


if __name__ == "__main__":
    unittest.main()
