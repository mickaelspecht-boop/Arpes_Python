"""Safety-net tests before splitting arpes/physics/fs.py.

Covers pure helpers (FSParams, _robust_norm, extract_fs_map) to detect any
semantic regression during the Qt move → ui/widgets/fs.py.
"""
import os
import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.collections import QuadMesh

try:
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from arpes.physics.fs import FSParams, _fs_cache_key, _robust_norm, extract_fs_map
    from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas
    HAS_FS = True
except Exception:  # pragma: no cover
    HAS_FS = False

try:
    from arpes.physics.fs_gamma import detect_gamma_from_fs_map
    HAS_FS_GAMMA = True
except Exception:  # pragma: no cover
    HAS_FS_GAMMA = False


@unittest.skipUnless(HAS_FS, "arpes.physics.fs unavailable")
class TestFSParamsDefaults(unittest.TestCase):
    def test_defaults(self):
        p = FSParams()
        self.assertAlmostEqual(p.a_lattice, 0.0)
        self.assertAlmostEqual(p.b_lattice, 0.0)
        self.assertAlmostEqual(p.ef_window, 0.030)
        self.assertAlmostEqual(p.ef_resolution_meV, 0.0)
        self.assertEqual(p.cmap, "inferno")
        self.assertTrue(p.normalize_profile)
        self.assertTrue(p.overlay_bz)


@unittest.skipUnless(HAS_FS, "arpes.physics.fs unavailable")
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


def _make_shifted_ring_map(center=(0.18, -0.11), n_kx=81, n_ky=75, *, asym=False):
    kx = np.linspace(-1.0, 1.0, n_kx)
    ky = np.linspace(-0.9, 0.9, n_ky)
    KX, KY = np.meshgrid(kx, ky)
    cx, cy = center
    r = np.sqrt((KX - cx) ** 2 + (KY - cy) ** 2)
    fs = np.exp(-((r - 0.45) ** 2) / (2 * 0.035 ** 2))
    if asym:
        fs = fs * np.where((KX > cx) & (KY > cy), 0.18, 1.0)
    return kx, ky, fs


@unittest.skipUnless(HAS_FS, "arpes.physics.fs unavailable")
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
        self.assertIn("boxcar EF", title)

    def test_asymmetric_ef_window_is_reported(self):
        kx, ky, ev, vol = _make_kxky_volume()
        ev = np.linspace(-0.005, 0.080, ev.size)
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        params = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)
        _kx_o, _ky_o, _fs, title = extract_fs_map(raw, params)
        self.assertIn("asymmetric EF window", title)

    def test_resolution_weighted_ef_integration_is_reported(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        params = FSParams(
            ef_window=0.050, ef_resolution_meV=15.0, temperature_K=20.0,
            smooth_sigma=0.0, normalize_profile=False,
        )
        _kx_o, _ky_o, _fs, title = extract_fs_map(raw, params)
        self.assertIn("Fermi/resolution weighted EF", title)

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
        self.assertIn("no norm", t_off)
        self.assertNotIn("no norm", t_on)
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

    def test_fs_cache_key_ignores_overlay_only_params(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        p1 = FSParams(ef_window=0.030, smooth_sigma=0.5, normalize_profile=False,
                      kx_center=0.0, ky_center=0.0, bz_shape="rectangle")
        p2 = FSParams(ef_window=0.030, smooth_sigma=0.5, normalize_profile=False,
                      kx_center=0.4, ky_center=-0.2, bz_shape="oblique",
                      bz_angle_deg=75.0, overlay_bz=False, show_hsym=False)
        self.assertEqual(_fs_cache_key(raw, p1), _fs_cache_key(raw, p2))

    def test_fs_cache_key_changes_for_image_params(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        p1 = FSParams(ef_window=0.030, smooth_sigma=0.5, normalize_profile=False)
        p2 = FSParams(ef_window=0.050, smooth_sigma=0.5, normalize_profile=False)
        self.assertNotEqual(_fs_cache_key(raw, p1), _fs_cache_key(raw, p2))


@unittest.skipUnless(HAS_FS_GAMMA, "arpes.physics.fs_gamma unavailable")
class TestFSGammaDetection(unittest.TestCase):
    def test_detect_gamma_recovers_shifted_symmetric_ring(self):
        kx, ky, fs = _make_shifted_ring_map(center=(0.18, -0.11))
        params = SimpleNamespace(klim=0.8, kx_center=0.0, ky_center=0.0)

        res = detect_gamma_from_fs_map(kx, ky, fs, params)

        self.assertAlmostEqual(res.kx, 0.18, delta=0.035)
        self.assertAlmostEqual(res.ky, -0.11, delta=0.035)
        self.assertGreaterEqual(len(res.gamma_kx_list), 10)
        self.assertGreaterEqual(len(res.gamma_ky_list), 10)
        self.assertEqual(res.quality, "high")
        self.assertGreater(res.symmetry_score, 0.8)
        self.assertAlmostEqual(res.kx_axis_center, 0.0, delta=1e-12)
        self.assertAlmostEqual(res.ky_axis_center, 0.0, delta=1e-12)
        self.assertAlmostEqual(res.gamma_delta_kx, res.kx - res.kx_axis_center, delta=1e-12)
        self.assertAlmostEqual(res.gamma_delta_ky, res.ky - res.ky_axis_center, delta=1e-12)

    def test_detect_gamma_reports_lower_quality_for_asymmetric_intensity(self):
        kx, ky, fs_good = _make_shifted_ring_map(center=(0.18, -0.11))
        _, _, fs_bad = _make_shifted_ring_map(center=(0.18, -0.11), asym=True)
        params = SimpleNamespace(klim=0.8, kx_center=0.0, ky_center=0.0)

        good = detect_gamma_from_fs_map(kx, ky, fs_good, params)
        bad = detect_gamma_from_fs_map(kx, ky, fs_bad, params)

        self.assertLess(bad.symmetry_score, good.symmetry_score)
        self.assertIn(bad.quality, {"medium", "low"})

    def test_canvas_detect_gamma_returns_quality_metrics(self):
        if not HAS_FS:
            self.skipTest("arpes.physics.fs unavailable")
        kx, ky, ring = _make_shifted_ring_map(center=(0.16, -0.08), n_kx=81, n_ky=75)
        ev = np.linspace(-0.03, 0.03, 7)
        vol = np.repeat(ring[:, :, None], ev.size, axis=2)
        raw = {
            "data": np.zeros((81, 7)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        canvas = FermiSurfaceCanvas()

        res = canvas.detect_gamma(raw, FSParams(ef_window=1.0, smooth_sigma=0.0,
                                                normalize_profile=False, klim=0.8))

        self.assertAlmostEqual(res["kx"], 0.16, delta=0.04)
        self.assertAlmostEqual(res["ky"], -0.08, delta=0.04)
        self.assertIn("symmetry_score", res)
        self.assertIn("quality", res)
        self.assertIn("gamma_delta_kx", res)
        self.assertIn("kx_axis_center", res)


@unittest.skipUnless(HAS_FS and QApplication is not None, "FS Qt unavailable")
class TestFermiSurfaceCanvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_draw_fs_reuses_quadmesh_for_overlay_only_changes(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        canvas = FermiSurfaceCanvas()
        p1 = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False,
                      kx_center=0.0, ky_center=0.0, overlay_bz=True)
        p2 = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False,
                      kx_center=0.0, ky_center=0.0, overlay_bz=True,
                      show_hsym=False, bz_half_x=1.2)

        canvas.draw_fs(raw, p1)
        first_mesh = canvas._mesh
        canvas.draw_fs(raw, p2)

        meshes = [c for c in canvas.ax.collections if isinstance(c, QuadMesh)]
        self.assertEqual(len(meshes), 1)
        self.assertIs(canvas._mesh, first_mesh)

    def test_draw_fs_rebuilds_quadmesh_for_new_shape(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw1 = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        kx2, ky2, ev2, vol2 = _make_kxky_volume(n_kx=24, n_ky=18)
        raw2 = {
            "data": np.zeros((24, 12)), "kpar": kx2, "ev_arr": ev2,
            "metadata": {
                "fs_data": vol2, "fs_kx": kx2, "fs_ky": ky2, "fs_energy": ev2,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        canvas = FermiSurfaceCanvas()
        params = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)

        canvas.draw_fs(raw1, params)
        first_mesh = canvas._mesh
        canvas.draw_fs(raw2, params)

        meshes = [c for c in canvas.ax.collections if isinstance(c, QuadMesh)]
        self.assertEqual(len(meshes), 1)
        self.assertIsNot(canvas._mesh, first_mesh)

    def test_draw_fs_rebuilds_quadmesh_for_internal_axis_change(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw1 = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        kx2 = np.array(kx, copy=True)
        kx2[len(kx2) // 2] += 0.01
        raw2 = {
            "data": np.zeros((20, 12)), "kpar": kx2, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx2, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        canvas = FermiSurfaceCanvas()
        params = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False)

        canvas.draw_fs(raw1, params)
        first_mesh = canvas._mesh
        canvas.draw_fs(raw2, params)

        self.assertIsNot(canvas._mesh, first_mesh)

    def test_draw_fs_resets_limits_when_overlay_disabled(self):
        kx, ky, ev, vol = _make_kxky_volume()
        raw = {
            "data": np.zeros((20, 12)), "kpar": kx, "ev_arr": ev,
            "metadata": {
                "fs_data": vol, "fs_kx": kx, "fs_ky": ky, "fs_energy": ev,
                "fs_kind": "kxky", "fs_source": "synthetic",
            },
        }
        canvas = FermiSurfaceCanvas()
        p1 = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False,
                      overlay_bz=True, klim=3.0)
        p2 = FSParams(ef_window=0.030, smooth_sigma=0.0, normalize_profile=False,
                      overlay_bz=False, klim=3.0)

        canvas.draw_fs(raw, p1)
        canvas.draw_fs(raw, p2)

        self.assertLess(max(abs(v) for v in canvas.ax.get_xlim()), 3.0)


if __name__ == "__main__":
    unittest.main()
