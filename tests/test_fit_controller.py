from __future__ import annotations

import unittest

import numpy as np

from arpes.physics.fit import MdcFitter, imaginary_self_energy
from arpes.core.session import FileEntry, FitParams, Session


class TestImSigmaConvention(unittest.TestCase):
    """Im Σ = vF·HWHM. HWHM-tagged fits use vF·γ; legacy (FWHM) use (vF/2)·γ."""

    def _linear_fit_result(self, gamma=0.05, **extra):
        e = np.linspace(-0.05, 0.0, 11)
        kF = 0.20 + 0.5 * e  # clean linear dispersion -> well-defined vF
        fr = {
            "e_fitted": e.tolist(),
            "kF_minus": [(-kF).tolist()],
            "kF_plus": [kF.tolist()],
            "gamma_corrige": [np.full_like(e, gamma).tolist()],
            "n_pairs": 1,
        }
        fr.update(extra)
        return fr, e, gamma

    def test_hwhm_tag_uses_vf_times_gamma(self):
        a = 3.9
        fr, e, gamma = self._linear_fit_result(width_convention="HWHM")
        out = imaginary_self_energy(fr, a)
        self.assertGreater(out["im_sigma"].size, 0)
        vF = out["vF_eV_A"]
        expected = vF * gamma * (np.pi / a)  # vF · HWHM[Å⁻¹]
        np.testing.assert_allclose(out["im_sigma"], expected, rtol=1e-6)

    def test_legacy_untagged_stays_fwhm_half(self):
        a = 3.9
        fr, e, gamma = self._linear_fit_result()  # no width_convention -> legacy FWHM
        out = imaginary_self_energy(fr, a)
        vF = out["vF_eV_A"]
        expected = (vF / 2.0) * gamma * (np.pi / a)
        np.testing.assert_allclose(out["im_sigma"], expected, rtol=1e-6)

    def test_hwhm_is_twice_legacy(self):
        a = 3.9
        hwhm, _, _ = self._linear_fit_result(width_convention="HWHM")
        legacy, _, _ = self._linear_fit_result()
        s_h = imaginary_self_energy(hwhm, a)["im_sigma"]
        s_l = imaginary_self_energy(legacy, a)["im_sigma"]
        np.testing.assert_allclose(s_h, 2.0 * s_l, rtol=1e-6)


class TestGammaHwhmMigration(unittest.TestCase):
    def test_legacy_fit_scaled_and_tagged(self):
        from arpes.physics.fit import gamma_to_hwhm_factor, migrate_fit_result_to_hwhm
        fr = {
            "e_fitted": [-0.1, 0.0],
            "gamma_brut": [[0.10, 0.12]],
            "gamma_corrige": [[0.08, 0.10]],
            "gamma_min": [[0.02, 0.02]],
            "sigma_gamma": [[0.01, 0.01]],
            "kF_minus": [[-0.2, -0.21]],  # positions must NOT scale
            "ensemble": {"gamma_std": [[0.02, 0.02]], "kF_minus_std": [[0.01, 0.01]]},
        }
        self.assertTrue(migrate_fit_result_to_hwhm(fr))
        self.assertEqual(fr["width_convention"], "HWHM")
        np.testing.assert_allclose(fr["gamma_brut"][0], [0.05, 0.06])
        np.testing.assert_allclose(fr["gamma_corrige"][0], [0.04, 0.05])
        np.testing.assert_allclose(fr["sigma_gamma"][0], [0.005, 0.005])
        np.testing.assert_allclose(fr["ensemble"]["gamma_std"][0], [0.01, 0.01])
        # positions untouched
        np.testing.assert_allclose(fr["kF_minus"][0], [-0.2, -0.21])
        np.testing.assert_allclose(fr["ensemble"]["kF_minus_std"][0], [0.01, 0.01])
        self.assertEqual(gamma_to_hwhm_factor(fr), 1.0)
        # idempotent
        self.assertFalse(migrate_fit_result_to_hwhm(fr))
        np.testing.assert_allclose(fr["gamma_brut"][0], [0.05, 0.06])

    def test_already_hwhm_untouched(self):
        from arpes.physics.fit import migrate_fit_result_to_hwhm
        fr = {"gamma_brut": [[0.05]], "width_convention": "HWHM"}
        self.assertFalse(migrate_fit_result_to_hwhm(fr))
        np.testing.assert_allclose(fr["gamma_brut"][0], [0.05])


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

    def test_summarize_reports_sigma_when_available(self):
        fr = {
            "e_fitted": [-0.1, 0.0],
            "kF_minus": [[0.1, 0.2]],
            "xg": [0.0, 0.02],
            "sigma_kF_plus": [[0.003, 0.004]],
            "sigma_gamma": [[0.005, 0.006]],
        }
        summary = MdcFitter.summarize(fr)
        self.assertIn("σkF med", summary.label_text)
        self.assertIn("σΓ med", summary.label_text)

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


class TestDetectNPairs(unittest.TestCase):
    def test_two_symmetric_peaks_gives_one_pair(self):
        from arpes.physics.fit import detect_n_pairs
        k = np.linspace(-1, 1, 401)
        m = np.exp(-((k - 0.4) ** 2) / 0.002) + np.exp(-((k + 0.4) ** 2) / 0.002)
        n = detect_n_pairs(k, m, k_min=-1, k_max=1, center_init=0.0,
                            smooth_sigma=2.0)
        self.assertEqual(n, 1)

    def test_four_symmetric_peaks_gives_two_pairs(self):
        from arpes.physics.fit import detect_n_pairs
        k = np.linspace(-1, 1, 801)
        peaks = [-0.6, -0.2, 0.2, 0.6]
        m = sum(np.exp(-((k - p) ** 2) / 0.001) for p in peaks)
        n = detect_n_pairs(k, m, k_min=-1, k_max=1, center_init=0.0,
                            smooth_sigma=1.5)
        self.assertEqual(n, 2)

    def test_flat_mdc_falls_back_to_one(self):
        from arpes.physics.fit import detect_n_pairs
        k = np.linspace(-1, 1, 101)
        m = np.ones_like(k)
        self.assertEqual(
            detect_n_pairs(k, m, k_min=-1, k_max=1, center_init=0.0), 1,
        )


class TestFermiVelocityMstar(unittest.TestCase):
    def test_linear_kF_gives_constant_vf(self):
        from arpes.physics.fit import compute_fermi_velocity_mstar
        # E = vF·(k - k0) avec vF_pi_a = 2.0, k0 = 0.4 π/a
        e = np.linspace(-0.04, 0.04, 11)
        k = 0.4 + e / 2.0  # k(E) = 0.4 + E/vF_pi_a
        fr = {"e_fitted": e, "kF_minus": [k]}
        a = 4.0  # Å
        out = compute_fermi_velocity_mstar(fr, a)
        # vF_eV_A = |slope_pi_a| / (π/a) = 2.0 / (π/4) = 8/π
        self.assertAlmostEqual(out["vF_eV_A"], 8.0 / np.pi, places=3)
        # kF_inv_A = 0.4·(π/4) = 0.1·π
        self.assertAlmostEqual(out["kF_inv_A"], 0.4 * np.pi / 4.0, places=4)
        self.assertTrue(np.isfinite(out["mstar_over_me"]))
        self.assertGreater(out["mstar_over_me"], 0.0)

    def test_no_crystal_a_returns_nan(self):
        from arpes.physics.fit import compute_fermi_velocity_mstar
        fr = {"e_fitted": [0.0], "kF_minus": [[0.3]]}
        out = compute_fermi_velocity_mstar(fr, 0.0)
        self.assertTrue(np.isnan(out["vF_eV_A"]))


class TestImSigma(unittest.TestCase):
    def test_returns_arrays_with_expected_scale(self):
        from arpes.physics.fit import imaginary_self_energy
        e = np.linspace(-0.04, 0.04, 9)
        k = 0.4 + e / 2.0
        gamma = np.full_like(e, 0.05)  # Γ(π/a) constant
        fr = {
            "e_fitted": e,
            "kF_minus": [k],
            "gamma_corrige": [gamma],
        }
        a = 4.0
        res = imaginary_self_energy(fr, a)
        self.assertEqual(res["energy"].size, e.size)
        # Im Σ = (vF/2) * Γ·(π/a) ; vF=8/π eV·Å → Im Σ = (8/π/2)·0.05·(π/4)
        expected = (8.0 / np.pi / 2.0) * 0.05 * (np.pi / 4.0)
        self.assertAlmostEqual(float(res["im_sigma"][0]), expected, places=4)

    def test_empty_when_no_gamma(self):
        from arpes.physics.fit import imaginary_self_energy
        res = imaginary_self_energy({"e_fitted": [0.0]}, 4.0)
        self.assertEqual(res["energy"].size, 0)
        self.assertEqual(res["error"], "missing Gamma arrays")

    def test_error_when_missing_lattice_a(self):
        from arpes.physics.fit import imaginary_self_energy
        res = imaginary_self_energy({"e_fitted": [0.0], "gamma_corrige": [[0.1]]}, 0.0)
        self.assertEqual(res["energy"].size, 0)
        self.assertEqual(res["error"], "missing lattice parameter a")


class TestEnsembleFit(unittest.TestCase):
    def _make_fitter(self):
        # Fitter that returns kF_minus = kF_init + fixed noise per run.
        from arpes.physics.fit import MdcFitter
        ev = np.linspace(-0.05, 0.0, 6)

        class _AP:
            def __init__(self):
                self._call = 0

            def fit_mdc_peak_pairs(self, data, kpar, ev_arr, **kw):
                self._call += 1
                pairs = kw.get("kF_init") or [0.30]
                kf0 = float(pairs[0])
                # Centered noise: sine of call number for reproducibility.
                noise = 0.001 * np.sin(self._call)
                return {
                    "e_fitted": ev.tolist(),
                    "kF_minus": [(np.full_like(ev, -kf0) + noise).tolist()],
                    "kF_plus": [(np.full_like(ev, kf0) + noise).tolist()],
                    "gamma_corrige": [np.full_like(ev, 0.05).tolist()],
                    "gamma_brut": [np.full_like(ev, 0.05).tolist()],
                    "xg": [0.0] * ev.size,
                    "residuals": [],
                    "chi2_red": [],
                }
        return MdcFitter(_AP()), ev

    def test_ensemble_aggregates_runs(self):
        from arpes.physics.fit import ensemble_fit
        fitter, ev = self._make_fitter()
        fp = FitParams(n_pairs=1, pairs=[{"kF_init": 0.30, "gamma_init": 0.08,
                                             "gamma_max": 0.30}])
        ens = ensemble_fit(fitter, np.zeros((10, ev.size)),
                            np.linspace(-1, 1, 10), ev, fp,
                            n_runs=20, jitter_pct=0.10, seed=42)
        self.assertEqual(ens["ensemble"], True)
        self.assertGreater(ens["n_ok"], 0)
        self.assertEqual(len(ens["e_fitted"]), ev.size)
        # median kF_plus ≈ 0.30 (± jitter).
        med = float(np.nanmedian(ens["kF_plus_med"]))
        self.assertAlmostEqual(med, 0.30, delta=0.05)
        self.assertGreater(np.nanmean(ens["kF_plus_std"]), 0.0)

    def test_ensemble_sigma_keeps_per_run_covariance_when_runs_identical(self):
        from arpes.physics.fit import ensemble_fit

        ev = np.linspace(-0.15, 0.05, 8)

        class _AP:
            def fit_mdc_peak_pairs(self, *args, **kwargs):
                return {
                    "e_fitted": ev.tolist(),
                    "kF_minus": [np.full(ev.size, -0.30).tolist()],
                    "kF_plus": [np.full(ev.size, 0.30).tolist()],
                    "gamma_corrige": [np.full(ev.size, 0.05).tolist()],
                    "gamma_brut": [np.full(ev.size, 0.05).tolist()],
                    "sigma_kF_minus": [np.full(ev.size, 0.004).tolist()],
                    "sigma_kF_plus": [np.full(ev.size, 0.005).tolist()],
                    "sigma_gamma": [np.full(ev.size, 0.006).tolist()],
                }

        fp = FitParams(n_pairs=1, pairs=[{"kF_init": 0.30, "gamma_init": 0.08,
                                             "gamma_max": 0.30}])
        ens = ensemble_fit(MdcFitter(_AP()), np.zeros((10, ev.size)),
                           np.linspace(-1, 1, 10), ev, fp,
                           n_runs=5, jitter_pct=0.10, seed=1)

        self.assertAlmostEqual(float(np.nanmedian(ens["kF_minus_std"])), 0.004)
        self.assertAlmostEqual(float(np.nanmedian(ens["kF_plus_std"])), 0.005)
        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_std"])), 0.006)

    def test_ensemble_preserves_full_fit_gamma_products(self):
        from arpes.physics.fit import ensemble_fit

        ev = np.linspace(-0.15, 0.05, 8)

        class _AP:
            def fit_mdc_peak_pairs(self, *args, **kwargs):
                return {
                    "e_fitted": ev.tolist(),
                    "kF_minus": [np.full(ev.size, -0.30).tolist()],
                    "kF_plus": [np.full(ev.size, 0.30).tolist()],
                    "gamma_brut": [np.full(ev.size, 0.10).tolist()],
                    "gamma_min": [np.full(ev.size, 0.08).tolist()],
                    "gamma_corrige": [np.full(ev.size, 0.06).tolist()],
                    "gamma_left_corrige": [np.full(ev.size, 0.05).tolist()],
                    "gamma_right_corrige": [np.full(ev.size, 0.07).tolist()],
                    "sigma_gamma": [np.full(ev.size, 0.006).tolist()],
                    "resolution": {"dE_meV": 15.0, "dk_inv_a": 0.005},
                    "width_mode": "independent",
                    "shape": "lorentzian",
                    "n_pairs": 1,
                }

        fp = FitParams(n_pairs=1, pairs=[{"kF_init": 0.30, "gamma_init": 0.08,
                                             "gamma_max": 0.30}])
        ens = ensemble_fit(MdcFitter(_AP()), np.zeros((10, ev.size)),
                           np.linspace(-1, 1, 10), ev, fp,
                           n_runs=3, jitter_pct=0.10, seed=2)

        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_brut_med"])), 0.10)
        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_med"])), 0.06)
        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_min_med"])), 0.08)
        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_left_corrige_med"])), 0.05)
        self.assertAlmostEqual(float(np.nanmedian(ens["gamma_right_corrige_med"])), 0.07)
        self.assertEqual(ens["resolution"]["dE_meV"], 15.0)
        self.assertEqual(ens["width_mode"], "independent")

    def test_ensemble_zero_runs_returns_empty(self):
        from arpes.physics.fit import ensemble_fit

        class _AP_bad:
            def fit_mdc_peak_pairs(self, *a, **kw):
                raise RuntimeError("boom")
        fitter = MdcFitter(_AP_bad())
        fp = FitParams()
        ens = ensemble_fit(fitter, np.zeros((10, 6)),
                            np.linspace(-1, 1, 10), np.linspace(-0.05, 0, 6),
                            fp, n_runs=3, jitter_pct=0.10)
        self.assertEqual(ens["n_ok"], 0)

    def test_hash_includes_ensemble_settings(self):
        from arpes.physics.fit import compute_fit_params_hash
        fp = FitParams()
        h1 = compute_fit_params_hash(fp, ensemble_settings={"n": 30, "jitter": 0.10})
        h2 = compute_fit_params_hash(fp, ensemble_settings={"n": 50, "jitter": 0.10})
        self.assertNotEqual(h1, h2)


class TestVoigtModel(unittest.TestCase):
    def test_voigt_extra_eta_param(self):
        from arpes.ui.widgets.plots.fit_overlay import _make_peak_pairs_model
        # 'global' lorentzian : n_extra = 1 (w_global)
        _m, npp, nx = _make_peak_pairs_model(2, width_mode="global",
                                              shape="lorentzian")
        self.assertEqual(nx, 1)
        # 'global' voigt : n_extra = 2 (w_global + eta)
        _m, _npp, nx2 = _make_peak_pairs_model(2, width_mode="global",
                                                shape="voigt")
        self.assertEqual(nx2, 2)
        # 'symmetric' voigt : n_extra = 1 (eta only)
        _m, _npp, nx3 = _make_peak_pairs_model(1, width_mode="symmetric",
                                                shape="voigt")
        self.assertEqual(nx3, 1)

    def test_voigt_model_callable(self):
        from arpes.ui.widgets.plots.fit_overlay import _make_peak_pairs_model
        model, npp, nx = _make_peak_pairs_model(1, width_mode="symmetric",
                                                  shape="voigt")
        # p layout : bg_a, bg_b, xg, [k0,A1,A2,w], eta
        p = [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 0.05, 0.5]
        k = np.linspace(-1, 1, 51)
        y = model(k, *p)
        self.assertEqual(y.shape, k.shape)
        # Peak at k0 must exceed the minimum.
        self.assertGreater(float(y.max()), float(y.min()) + 0.1)

    def test_shape_threads_through_fit_kwargs(self):
        from arpes.physics.fit import MdcFitter
        from arpes.core.session import FitParams
        kw = MdcFitter.fit_kwargs(FitParams(shape="voigt"))
        self.assertEqual(kw["shape"], "voigt")
        self.assertEqual(MdcFitter.fit_kwargs(FitParams())["shape"],
                          "lorentzian")


class TestWidthModeAliasing(unittest.TestCase):
    def test_alias_asymmetric_maps_to_independent(self):
        from arpes.ui.widgets.plots.fit_overlay import (
            _make_peak_pairs_model, _normalize_width_mode,
        )
        self.assertEqual(_normalize_width_mode("asymmetric"), "independent")
        self.assertEqual(_normalize_width_mode("symmetric"), "symmetric")
        # n_pp must match 'independent' (5), not 'symmetric' (4).
        _m, npp_a, _ = _make_peak_pairs_model(1, width_mode="asymmetric")
        _m, npp_i, _ = _make_peak_pairs_model(1, width_mode="independent")
        self.assertEqual(npp_a, npp_i)
        self.assertEqual(npp_a, 5)

    def test_fit_kwargs_threads_independent(self):
        from arpes.physics.fit import MdcFitter
        from arpes.core.session import FitParams
        kw = MdcFitter.fit_kwargs(FitParams(width_mode="independent"))
        self.assertEqual(kw["width_mode"], "independent")


class TestImSigmaSide(unittest.TestCase):
    def _fr(self):
        e = np.linspace(-0.04, 0.04, 9)
        k = 0.4 + e / 2.0
        gL = np.full_like(e, 0.04)
        gR = np.full_like(e, 0.07)
        return {
            "e_fitted": e,
            "kF_minus": [k],
            "gamma_corrige": [0.5 * (gL + gR)],
            "gamma_left_corrige": [gL],
            "gamma_right_corrige": [gR],
            "width_mode": "independent",
        }

    def test_left_uses_gamma_left(self):
        from arpes.physics.fit import imaginary_self_energy
        a = 4.0
        rl = imaginary_self_energy(self._fr(), a, side="left")
        rr = imaginary_self_energy(self._fr(), a, side="right")
        rm = imaginary_self_energy(self._fr(), a, side="mean")
        self.assertEqual(rl["side"], "left")
        self.assertEqual(rr["side"], "right")
        self.assertGreater(float(rr["im_sigma"][0]), float(rl["im_sigma"][0]))
        # mean entre les deux
        self.assertLess(float(rl["im_sigma"][0]), float(rm["im_sigma"][0]))
        self.assertLess(float(rm["im_sigma"][0]), float(rr["im_sigma"][0]))

    def test_side_falls_back_to_mean_when_no_left(self):
        from arpes.physics.fit import imaginary_self_energy
        fr = self._fr()
        del fr["gamma_left_corrige"]
        res = imaginary_self_energy(fr, 4.0, side="left")
        # Pas de gauche dispo → tombe sur gamma_corrige (mean)
        self.assertGreater(res["energy"].size, 0)


class TestFitRunnerImSigmaLattice(unittest.TestCase):
    def test_lattice_a_uses_session_sample_before_ui_fallback(self):
        from types import SimpleNamespace
        from arpes.ui.controllers.fit_runner_controller import FitRunnerController

        session = Session()
        session.current_sample = {"a_angstrom": 4.2}
        entry = session.get_or_create("BM1")
        parent = SimpleNamespace(
            _current_path="BM1",
            _session=session,
            _params=SimpleNamespace(
                sp_crystal_a=SimpleNamespace(value=lambda: 0.0),
            ),
        )

        out = FitRunnerController(parent)._lattice_a_for_current_fit()

        self.assertEqual(out, 4.2)


class TestImagSelfEnergyDialog(unittest.TestCase):
    def test_accepts_numpy_arrays(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from arpes.ui.widgets.dialogs.imag_self_energy import ImagSelfEnergyDialog

        app = QApplication.instance() or QApplication([])
        payload = {
            "energy": np.array([-0.02, 0.0]),
            "im_sigma": np.array([0.01, 0.02]),
            "im_sigma_std": np.array([0.001, 0.002]),
            "vF_eV_A": 2.0,
            "pair_index": 0,
            "side": "mean",
        }

        dlg = ImagSelfEnergyDialog(payload)

        self.assertEqual(dlg.windowTitle(), "Self-energy Im Σ(E)")
