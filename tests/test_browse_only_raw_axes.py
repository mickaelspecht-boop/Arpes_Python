"""Browse-only raw-axes mode: axis swap, cache key, physics guards."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from arpes.ui.controllers.load_controller import LoadController, _PreparedEntry


# --------------------------------------------------------------- helpers
def _make_ctrl(browse_only=True, phi=0.0, a=None, statuses=None):
    parent = SimpleNamespace(
        _session=SimpleNamespace(browse_only=browse_only),
        _status=(statuses.append if statuses is not None else lambda m: None),
    )
    ctrl = LoadController(parent)
    ctrl._work_function_for_load = lambda path, prepared: phi
    ctrl._lattice_a_for_load = lambda entry: a
    return ctrl


def _prepared(hv=0.0, placeholder=False):
    return SimpleNamespace(entry=object(), hv_for_load=hv, hv_placeholder=placeholder)


def _d(n_theta=8, n_e=10, scale="kinetic", with_meta=True):
    meta = {}
    if with_meta:
        meta = {
            "theta_par_deg": np.linspace(-7, 7, n_theta),
            "energy_raw": np.linspace(95.0, 96.0, n_e),
            "energy_axis_original": scale,
        }
    return {
        "data": np.ones((n_theta, n_e)),
        "kpar": np.linspace(-1, 1, n_theta),
        "ev_arr": np.linspace(-0.5, 0.1, n_e),
        "metadata": meta,
    }


# ------------------------------------------------------- raw-axes swap
class TestMaybeApplyRawAxes:
    def test_swap_when_phi_missing(self):
        ctrl = _make_ctrl(phi=0.0, a=4.1)
        d = _d()
        ctrl._maybe_apply_raw_axes(d, _prepared(hv=21.2))
        assert d["metadata"]["axes_raw_view"] is True
        np.testing.assert_allclose(d["kpar"], d["metadata"]["theta_par_deg"])
        np.testing.assert_allclose(d["ev_arr"], d["metadata"]["energy_raw"])
        assert "θ" in d["metadata"]["axes_raw_xlabel"]
        assert "kinetic" in d["metadata"]["axes_raw_ylabel"]

    def test_swap_when_hv_missing(self):
        ctrl = _make_ctrl(phi=4.4, a=4.1)
        d = _d()
        ctrl._maybe_apply_raw_axes(d, _prepared(hv=0.0))
        assert d["metadata"]["axes_raw_view"] is True

    def test_binding_scale_label(self):
        ctrl = _make_ctrl(phi=0.0, a=None)
        d = _d(scale="binding")
        ctrl._maybe_apply_raw_axes(d, _prepared())
        assert "binding" in d["metadata"]["axes_raw_ylabel"]

    def test_no_swap_outside_browse_only(self):
        ctrl = _make_ctrl(browse_only=False, phi=0.0, a=None)
        d = _d()
        kpar_before = d["kpar"].copy()
        ctrl._maybe_apply_raw_axes(d, _prepared())
        assert "axes_raw_view" not in d["metadata"]
        np.testing.assert_allclose(d["kpar"], kpar_before)

    def test_no_swap_when_fully_calibrated(self):
        ctrl = _make_ctrl(phi=4.4, a=4.1)
        d = _d()
        ctrl._maybe_apply_raw_axes(d, _prepared(hv=21.2))
        assert "axes_raw_view" not in d["metadata"]

    def test_stale_flag_cleared_on_calibrated_load(self):
        ctrl = _make_ctrl(phi=4.4, a=4.1)
        d = _d()
        d["metadata"]["axes_raw_view"] = True  # stale from previous load
        ctrl._maybe_apply_raw_axes(d, _prepared(hv=21.2))
        assert "axes_raw_view" not in d["metadata"]

    def test_missing_loader_meta_raises_loud(self):
        ctrl = _make_ctrl(phi=0.0, a=None)
        d = _d(with_meta=False)
        with pytest.raises(ValueError, match="raw axes unavailable"):
            ctrl._maybe_apply_raw_axes(d, _prepared())

    def test_shape_mismatch_raises_loud(self):
        ctrl = _make_ctrl(phi=0.0, a=None)
        d = _d()
        d["metadata"]["theta_par_deg"] = np.linspace(-7, 7, 5)  # wrong length
        with pytest.raises(ValueError, match="shape mismatch"):
            ctrl._maybe_apply_raw_axes(d, _prepared())

    def test_hv_placeholder_stripped(self):
        ctrl = _make_ctrl(phi=0.0, a=None)
        d = _d()
        d["hv"] = 21.2
        d["metadata"]["hv_warning"] = "hν=21.2 eV inconsistent..."
        d["metadata"]["loader_warnings"] = [
            "hν=21.2 eV inconsistent with the file energy window",
            "unrelated warning kept",
        ]
        ctrl._maybe_apply_raw_axes(d, _prepared(placeholder=True))
        assert d["hv"] is None
        assert d["metadata"]["hv_warning"] is None
        assert d["metadata"]["loader_warnings"] == ["unrelated warning kept"]

    def test_real_hv_kept_without_placeholder(self):
        ctrl = _make_ctrl(phi=0.0, a=None)
        d = _d()
        d["hv"] = 48.0  # genuine, e.g. from a BESSY file
        ctrl._maybe_apply_raw_axes(d, _prepared(hv=0.0))
        assert d["hv"] == 48.0


# --------------------------------------------------------- cache key
def test_cache_key_distinguishes_browse_only(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("data")
    keys = {}
    for browse in (False, True):
        params = SimpleNamespace(
            sp_ef=SimpleNamespace(value=lambda: 0.0),
            sp_phi=SimpleNamespace(value=lambda: 0.0),
        )
        parent = SimpleNamespace(
            _session=SimpleNamespace(browse_only=browse),
            _params=params,
            _bessy_energy_reference_mode=lambda: "auto",
        )
        ctrl = LoadController(parent)
        ctrl._work_function_for_load = lambda path, prepared: 0.0
        ctrl._lattice_a_for_load = lambda entry: None
        prepared = SimpleNamespace(
            entry=SimpleNamespace(meta=SimpleNamespace(
                temperature=0.0, azi=0.0, polarization="")),
            fmt_guess="cls_txt",
            hv_for_load=0.0,
            angle_offsets=None,
        )
        keys[browse] = ctrl._load_cache_key(str(f), prepared)
    assert keys[False] != keys[True]


# ------------------------------------------------------------- guards
def _raw_d():
    d = _d()
    d["metadata"]["axes_raw_view"] = True
    return d


class TestGuards:
    def test_get_work_data_refuses(self):
        from arpes.ui.controllers.fit_runner_controller import FitRunnerController
        statuses = []
        parent = SimpleNamespace(_raw_data=_raw_d(), _status=statuses.append)
        ctrl = FitRunnerController(parent)
        assert ctrl._get_work_data() == (None, None, None)
        assert any("browse-only" in s for s in statuses)

    def test_ef_calibrate_refuses(self):
        from arpes.ui.controllers.fit_runner_controller import FitRunnerController
        statuses = []
        parent = SimpleNamespace(_raw_data=_raw_d(), _status=statuses.append)
        ctrl = FitRunnerController(parent)
        ctrl._ef_calibrate()
        assert any("EF calibration not available" in s for s in statuses)

    def test_fs_tab_refuses(self):
        from arpes.ui.controllers.fs_controller import FSController
        statuses = []
        parent = SimpleNamespace(
            _raw_data=_raw_d(),
            _status=statuses.append,
            _fs_canvas=object(),
            _fs_controls=object(),
        )
        ctrl = FSController(parent)
        ctrl._draw_fs_tab()
        assert any("Fermi surface not available" in s for s in statuses)


# ----------------------------------------------- browse-only saves φ/a
class TestBrowseOnlySavesConfigs:
    def _session(self):
        return SimpleNamespace(
            browse_only=False,
            sample_configs={},
            current_sample={},
            saved=[],
            save=lambda self=None: None,
        )

    def test_save_dialog_configs_multi(self):
        from arpes.ui.controllers.sample_setup_controller import SampleSetupController
        session = self._session()
        parent = SimpleNamespace(_session=session, _status=lambda m: None)
        ctrl = SampleSetupController(parent)
        dialog = SimpleNamespace(result_configs=lambda: {
            "BNO": {"work_function_eV": 4.3, "a_angstrom": 4.14},
        })
        n = ctrl._save_dialog_configs(dialog)
        assert n == 1
        assert session.sample_configs["BNO"]["work_function_eV"] == pytest.approx(4.3)

    def test_open_dialog_clears_browse_only(self, monkeypatch):
        from arpes.ui.controllers.sample_setup_controller import SampleSetupController
        statuses = []
        session = self._session()
        session.browse_only = True
        parent = SimpleNamespace(_session=session, _status=statuses.append)
        ctrl = SampleSetupController(parent)
        monkeypatch.setattr(ctrl, "_open_sample_setup",
                            lambda *, auto: None)
        ctrl._sample_setup_action("open_dialog")
        assert session.browse_only is False
        assert any("Browse only disabled" in s for s in statuses)

    def test_folder_opened_skipped_in_browse_only(self, monkeypatch):
        from arpes.ui.controllers.sample_setup_controller import SampleSetupController
        session = self._session()
        session.browse_only = True
        parent = SimpleNamespace(_session=session, _status=lambda m: None)
        ctrl = SampleSetupController(parent)
        opened = []
        monkeypatch.setattr(ctrl, "_open_sample_setup",
                            lambda *, auto: opened.append(auto))
        ctrl._sample_setup_action("folder_opened")
        assert opened == []
