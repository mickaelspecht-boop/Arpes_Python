"""Tests for the per-subfolder SampleSetupDialog (Qt offscreen)."""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from arpes.ui.widgets.dialogs.sample_setup_dialog import SampleSetupDialog

app = QApplication.instance() or QApplication([])


def _dlg(**kw):
    defaults = dict(
        folder_name="run42",
        subfolders=[("BNO", 12), ("Au_ref", 3)],
        n_root_files=0,
        detected_mode="multi",
    )
    defaults.update(kw)
    return SampleSetupDialog(**defaults)


class TestSampleSetupDialog:
    def test_multi_mode_one_row_per_subfolder(self):
        dlg = _dlg()
        assert dlg.table.rowCount() == 2
        assert dlg.rb_multi.isChecked()

    def test_single_mode_detected_hides_table(self):
        dlg = _dlg(subfolders=[], detected_mode="single", n_root_files=4)
        assert dlg.rb_single.isChecked()
        assert not dlg.table.isVisibleTo(dlg)
        assert dlg._single_box.isVisibleTo(dlg)

    def test_result_configs_only_nonzero(self):
        dlg = _dlg()
        name, sp_phi, sp_a = dlg._row_spins[0]
        sp_phi.setValue(4.3)
        out = dlg.result_configs()
        assert out == {name: {"work_function_eV": 4.3}}

    def test_same_for_all(self):
        dlg = _dlg()
        _, sp_phi, sp_a = dlg._row_spins[0]
        sp_phi.setValue(4.4)
        sp_a.setValue(4.14)
        dlg._apply_same_for_all()
        out = dlg.result_configs()
        assert out["BNO"]["a_angstrom"] == pytest.approx(4.14)
        assert out["Au_ref"]["work_function_eV"] == pytest.approx(4.4)

    def test_single_mode_returns_root_key(self):
        dlg = _dlg(subfolders=[], detected_mode="single", n_root_files=2)
        dlg.sp_phi_single.setValue(4.5)
        assert dlg.result_configs() == {"": {"work_function_eV": 4.5}}

    def test_skip_returns_nothing_when_blank(self):
        dlg = _dlg()
        assert dlg.result_configs() == {}

    def test_prefill_from_existing_config(self):
        dlg = _dlg(existing={"BNO": {"work_function_eV": 4.3, "a_angstrom": 4.14}})
        name, sp_phi, sp_a = dlg._row_spins[0]
        assert name == "BNO"
        assert sp_phi.value() == pytest.approx(4.3)
        assert "saved sample setup" in sp_phi.toolTip()

    def test_prefill_from_session_default(self):
        dlg = _dlg(session_default={"work_function_eV": 4.5})
        _, sp_phi, _ = dlg._row_spins[0]
        assert sp_phi.value() == pytest.approx(4.5)
        assert "session/logbook" in sp_phi.toolTip()

    def test_unusual_phi_highlighted(self):
        dlg = _dlg()
        _, sp_phi, _ = dlg._row_spins[0]
        sp_phi.setValue(12.0)
        assert "background" in sp_phi.styleSheet()
        sp_phi.setValue(4.4)
        assert sp_phi.styleSheet() == ""

    def test_mode_switch_carries_values(self):
        dlg = _dlg()
        _, sp_phi, _ = dlg._row_spins[0]
        sp_phi.setValue(4.2)
        dlg.rb_single.setChecked(True)
        assert dlg.sp_phi_single.value() == pytest.approx(4.2)
