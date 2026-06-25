from types import SimpleNamespace

import numpy as np

from arpes.ui.widgets.results_bands import (
    band_name,
    band_style,
    high_uncertainty_mask,
)


def _entry(*pairs):
    return SimpleNamespace(fit_params=SimpleNamespace(pairs=list(pairs)))


def test_band_name_defaults_and_accepts_scientific_label():
    assert band_name(_entry({}), 0) == "P1"
    assert band_name(_entry({"label": "alpha"}), 0) == "alpha"


def test_pair_styles_are_visually_distinct():
    p1 = band_style((0.2, 0.5, 0.8), 0)
    p2 = band_style((0.2, 0.5, 0.8), 1)
    assert p1["color"] != p2["color"]
    assert p1["marker_minus"] != p2["marker_minus"]
    assert p1["linestyle"] != p2["linestyle"]


def test_uncertainty_mask_flags_only_large_sigma():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    sigma = np.array([0.004, 0.005, 0.006, 0.09])
    assert high_uncertainty_mask(values, sigma).tolist() == [False, False, False, True]


def test_fit_pair_edit_preserves_results_name_and_visibility():
    from arpes.ui.widgets.params import FitParamsPanel

    class _Spin:
        def __init__(self, value):
            self._value = value
        def value(self):
            return self._value

    panel = SimpleNamespace(
        _current_pair=0,
        _pair_params=[{
            "kF_init": 0.2, "gamma_init": 0.03, "gamma_max": 0.1,
            "label": "alpha", "results_visible": False,
        }],
        sp_kfi=_Spin(0.25),
        sp_gi=_Spin(0.04),
        sp_gm=_Spin(0.2),
    )
    FitParamsPanel._save_pair(panel)
    assert panel._pair_params[0]["label"] == "alpha"
    assert panel._pair_params[0]["results_visible"] is False
    assert panel._pair_params[0]["kF_init"] == 0.25


def test_new_pair_copies_numerics_not_previous_band_identity():
    from arpes.ui.widgets.params import FitParamsPanel

    class _Signal:
        def emit(self):
            pass

    panel = SimpleNamespace(
        _current_pair=0,
        _pair_params=[{
            "kF_init": 0.2, "gamma_init": 0.03, "gamma_max": 0.1,
            "label": "alpha", "results_visible": False,
        }],
        _save_pair=lambda: None,
        _pair_lbl=SimpleNamespace(setup=lambda *_args: None),
        _load_pair=lambda *_args: None,
        params_changed=_Signal(),
    )
    FitParamsPanel._on_n_pairs_changed(panel, 2)
    assert panel._pair_params[1]["kF_init"] == 0.2
    assert "label" not in panel._pair_params[1]
    assert "results_visible" not in panel._pair_params[1]


def test_band_registry_rename_and_visibility_round_trip(tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    from arpes.core.session import FileEntry, FitParams, Session
    from arpes.ui.widgets.results import ResultsPanel

    app = QApplication.instance() or QApplication([])
    session = Session(tmp_path)
    params = FitParams(n_pairs=2, pairs=[
        {"kF_init": 0.2, "gamma_init": 0.03, "gamma_max": 0.1},
        {"kF_init": 0.4, "gamma_init": 0.06, "gamma_max": 0.2},
    ])
    fit_result = {
        "n_pairs": 2, "e_fitted": [-0.1, 0.0],
        "kF_minus": [[-0.2, -0.1], [-0.4, -0.3]],
        "kF_plus": [[0.2, 0.1], [0.4, 0.3]],
        "gamma_corrige": [[0.03, 0.04], [0.06, 0.07]],
    }
    session.files["BM1"] = FileEntry(fit_params=params, fit_result=fit_result)
    panel = ResultsPanel(session)
    panel.refresh()
    panel._tree_bands.topLevelItem(0).child(1).setText(0, "beta")
    app.processEvents()
    panel._tree_bands.topLevelItem(0).child(1).setCheckState(
        0, Qt.CheckState.Unchecked,
    )
    app.processEvents()
    session.flush_save()

    restored = Session()
    restored.load(tmp_path / ".arpes_session.json")
    pair = restored.files["BM1"].fit_params.pairs[1]
    assert pair["label"] == "beta"
    assert pair["results_visible"] is False
