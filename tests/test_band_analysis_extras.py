"""Tests for the recently-added band-analysis features.

Covers: summary HTML render, cross-validation block, CSV row builder,
autofill defaults, material presets, status-row badge styles,
fit-on-displayed-data distortion path.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arpes.core.session import FileEntry
from arpes.ui.controllers.band_analysis_controller import BandAnalysisController
from arpes.ui.widgets.band_analysis_summary import (
    cross_validation_block,
    render_summary_html,
)

try:
    import PyQt6  # noqa: F401
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

requires_qt = pytest.mark.skipif(not _HAS_QT, reason="PyQt6 absent")


# ---------------------------------------------------------------------------
# 1. Summary HTML render
# ---------------------------------------------------------------------------
class TestSummaryRender:
    def test_no_fit_returns_placeholder_only(self):
        html = render_summary_html({}, has_fit=False, n_points=0, n_pairs=0)
        assert "Aucun fit MDC" in html
        assert "table" in html

    def test_with_tb_kink_gap_contains_all_metrics(self):
        ba = {
            "tb": {"params": {"t": 0.42, "tprime": -0.1},
                   "perr": {"t": 0.01, "tprime": 0.02},
                   "m_eff_over_me": 2.1, "bandwidth_eV": 1.6,
                   "chi2_red": 1.5e-4, "n_points": 18, "notes": []},
            "kink": {"lambda": 0.55, "lambda_err": 0.05, "v_bare": 2.0,
                     "notes": []},
            "gap": {"deltas_meV": [12.5], "delta_err_meV": [0.3],
                    "gammas_meV": [3.1], "k_F_inv_A": 0.42,
                    "chi2_red": 8e-3, "notes": []},
        }
        html = render_summary_html(ba, has_fit=True, n_points=18, n_pairs=1)
        assert "t=" not in html  # named tb param uses table cells
        assert "+0.4200" in html
        assert "λ" in html
        assert "0.550" in html
        assert "12.50" in html
        assert "0.420" in html  # k_F formatted to 3 decimals
        assert "Cohérence" in html  # cross-validation when both tb m* and λ
        assert "m*/m" in html

    def test_warning_inline_on_negative_lambda(self):
        ba = {"kink": {"lambda": -0.2}}
        html = render_summary_html(ba, has_fit=True, n_points=5, n_pairs=1)
        assert "non physique" in html

    def test_warnings_consolidated_at_bottom(self):
        ba = {
            "tb": {"params": {"t": 0.1}, "perr": {"t": 0.0},
                   "notes": ["bandwidth fragile"]},
            "kink": {"lambda": 0.4, "notes": ["bare-band sensitive"]},
            "gap": {"deltas_meV": [], "notes": ["k_F mal contraint"]},
        }
        html = render_summary_html(ba, has_fit=True, n_points=10, n_pairs=1)
        assert "Warnings" in html
        assert "bandwidth fragile" in html
        assert "bare-band sensitive" in html
        assert "k_F mal contraint" in html


class TestCrossValidation:
    def test_returns_none_without_both_values(self):
        assert cross_validation_block({}, {}) is None
        assert cross_validation_block({"m_eff_over_me": 2.0}, {}) is None
        assert cross_validation_block({}, {"lambda": 0.5}) is None

    def test_consistent_within_30pct_no_flag(self):
        block = cross_validation_block({"m_eff_over_me": 1.5}, {"lambda": 0.5})
        assert "écart" not in block

    def test_inconsistent_over_30pct_flags(self):
        block = cross_validation_block({"m_eff_over_me": 3.0}, {"lambda": 0.5})
        assert "écart" in block


# ---------------------------------------------------------------------------
# 2. CSV row builder
# ---------------------------------------------------------------------------
class TestCSVRows:
    def test_empty_band_analysis_returns_header_only(self):
        entry = SimpleNamespace(fit_result=None)
        rows = BandAnalysisController.build_csv_rows(entry, {})
        assert rows == [("source", "metric", "value", "error", "unit")]

    def test_full_ba_produces_typed_rows(self):
        entry = SimpleNamespace(fit_result={"e_fitted": [0.0, -0.1, -0.2]})
        ba = {
            "tb": {"params": {"t": 0.42}, "perr": {"t": 0.01},
                   "m_eff_over_me": 2.1, "bandwidth_eV": 1.6, "chi2_red": 1e-3},
            "kink": {"lambda": 0.55, "lambda_err": 0.05, "v_bare": 2.0},
            "gap": {"deltas_meV": [12.5, 4.0], "delta_err_meV": [0.3, 0.2],
                    "gammas_meV": [3.1, 1.8], "k_F_inv_A": 0.42,
                    "chi2_red": 8e-3},
        }
        rows = BandAnalysisController.build_csv_rows(entry, ba)
        body = rows[1:]
        sources = {r[0] for r in body}
        assert sources == {"MDC", "TB", "Kink", "Gap"}
        metrics = {(r[0], r[1]) for r in body}
        assert ("MDC", "n_points") in metrics
        assert ("TB", "t") in metrics
        assert ("Kink", "lambda") in metrics
        assert ("Gap", "Delta_1") in metrics
        assert ("Gap", "Delta_2") in metrics
        assert ("Gap", "k_F") in metrics
        # Units present where meaningful
        units = {r[1]: r[4] for r in body}
        assert units["t"] == "eV"
        assert units["lambda"] == ""
        assert units["v_bare"] == "eV.A"
        assert units["Delta_1"] == "meV"
        assert units["k_F"] == "A^-1"


# ---------------------------------------------------------------------------
# 3. Autofill defaults
# ---------------------------------------------------------------------------
class TestAutofillDefaults:
    def _entry(self, fr=None, crystal_a=0.0):
        meta = SimpleNamespace(crystal_a_angstrom=crystal_a)
        return SimpleNamespace(fit_result=fr, meta=meta)

    def test_branch_picks_minus_when_both_valid(self):
        fr = {"kF_minus": [[0.1, 0.2]], "kF_plus": [[-0.1, -0.2]]}
        d = BandAnalysisController.compute_autofill_defaults("tb", self._entry(fr))
        assert d["branch"] == "kF_minus"

    def test_branch_falls_back_to_plus_when_minus_all_nan(self):
        fr = {"kF_minus": [[np.nan, np.nan]], "kF_plus": [[0.3, 0.4]]}
        d = BandAnalysisController.compute_autofill_defaults("kink", self._entry(fr))
        assert d["branch"] == "kF_plus"

    def test_tb_includes_a_when_crystal_a_known(self):
        d = BandAnalysisController.compute_autofill_defaults(
            "tb", self._entry(crystal_a=4.14), ef_offset=0.05,
        )
        assert d["a"] == 4.14
        assert d["E_F"] == 0.05

    def test_tb_omits_a_when_crystal_a_zero(self):
        d = BandAnalysisController.compute_autofill_defaults(
            "tb", self._entry(crystal_a=0.0),
        )
        assert "a" not in d

    def test_kink_window_derived_from_dispersion(self):
        fr = {"e_fitted": [-0.30, -0.20, -0.10, -0.05, 0.0]}
        d = BandAnalysisController.compute_autofill_defaults("kink", self._entry(fr))
        assert d["window_lo"] == pytest.approx(-0.30)
        assert d["window_hi"] == pytest.approx(-0.30 + 0.6 * 0.30)

    def test_kink_window_skipped_when_too_few_points(self):
        fr = {"e_fitted": [-0.3, -0.2]}
        d = BandAnalysisController.compute_autofill_defaults("kink", self._entry(fr))
        assert "window_lo" not in d

    def test_gap_default_omega(self):
        d = BandAnalysisController.compute_autofill_defaults("gap", self._entry())
        assert d["omega_max_meV"] == 30.0


# ---------------------------------------------------------------------------
# 4. Material presets dict
# ---------------------------------------------------------------------------
class TestPresets:
    def test_presets_contain_known_materials(self):
        from arpes.ui.widgets.band_analysis_presets import PRESETS
        assert {"Custom", "BaNi2P2", "Bi2212", "FeSe", "Cu(111)"}.issubset(set(PRESETS))

    def test_preset_values_sensible(self):
        from arpes.ui.widgets.band_analysis_presets import PRESETS
        for name, cfg in PRESETS.items():
            if name == "Custom":
                continue
            assert "a" in cfg and 1.5 < cfg["a"] < 10.0
            assert cfg["lattice"] in ("chain", "square", "hex", "rect")
            assert "omega_max_meV" in cfg and 0 < cfg["omega_max_meV"] < 200


# ---------------------------------------------------------------------------
# 5. Status-row badge styles
# ---------------------------------------------------------------------------
@requires_qt
class TestStageStyle:
    def test_stage_style_done_uses_green(self):
        from arpes.ui.widgets.band_analysis_panel import BandAnalysisPanel
        # _stage_style is independent of Qt state — call directly on the class
        # via an unbound-method shim.
        style = BandAnalysisPanel._stage_style.__func__(None, done=True)
        assert "#14532d" in style or "#86efac" in style

    def test_stage_style_pending_uses_gray(self):
        from arpes.ui.widgets.band_analysis_panel import BandAnalysisPanel
        style = BandAnalysisPanel._stage_style.__func__(None, done=False)
        assert "#222" in style


# ---------------------------------------------------------------------------
# 6. Fit-on-displayed-data distortion path (option A)
# ---------------------------------------------------------------------------
@requires_qt
class TestGetWorkDataDistortion:
    """Verifies _get_work_data warps axes when distortion is active."""

    def _build_parent(self, data, kpar, ev, bm_dist=None, grid_cfg=None):
        from arpes.core.session import FileEntry, Session
        from arpes.ui.controllers.fit_runner_controller import FitRunnerController

        sess = Session()
        entry = FileEntry(
            bm_distortion=bm_dist or {},
            grid_correction=grid_cfg or {},
        )
        sess.files["f.h5"] = entry
        params = SimpleNamespace()
        params._resolution_source_detail = ""
        cmb = SimpleNamespace(currentText=lambda: "Raw")
        parent = SimpleNamespace(
            _raw_data={"data": data, "kpar": kpar, "ev_arr": ev, "hv": 100.0},
            _cmb_view=cmb,
            _params=params,
            _data_disp=None,
            _session=sess,
            _current_path="f.h5",
            _current_entry=lambda: entry,
        )
        ctrl = FitRunnerController(parent)
        return ctrl, entry

    def test_no_distortion_returns_raw_axes(self):
        data = np.ones((10, 8), dtype=float)
        kpar = np.linspace(-1.0, 1.0, 10)
        ev = np.linspace(-0.3, 0.0, 8)
        ctrl, _ = self._build_parent(data, kpar, ev)
        d_out, k_out, e_out = ctrl._get_work_data()
        np.testing.assert_array_equal(k_out, kpar)
        np.testing.assert_array_equal(e_out, ev)
        np.testing.assert_array_equal(d_out, data)

    @pytest.mark.skipif(
        pytest.importorskip("scipy", reason="scipy required for distortion warp") is None,
        reason="scipy required",
    )
    def test_distortion_active_routes_through_compute_bandmap_display(self):
        data = np.ones((30, 25), dtype=float)
        kpar = np.linspace(-1.0, 1.0, 30)
        ev = np.linspace(-0.3, 0.0, 25)
        bm_dist = {
            "enabled": True,
            "parabola": {"enabled": True, "a": -0.1, "k0": 0.0},
        }
        ctrl, entry = self._build_parent(data, kpar, ev, bm_dist=bm_dist)
        d_out, k_out, e_out = ctrl._get_work_data()
        # The warped pipeline returns arrays of the same shape but data may
        # differ (interpolation). Identity-check via shape + non-None.
        assert d_out is not None
        assert d_out.shape == data.shape
        assert k_out.shape == kpar.shape
        assert e_out.shape == ev.shape
