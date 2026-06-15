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
        name, sp_phi, sp_a, sp_b = dlg._row_spins[0]
        sp_phi.setValue(4.3)
        out = dlg.result_configs()
        assert out == {name: {"work_function_eV": 4.3}}

    def test_same_for_all(self):
        dlg = _dlg()
        _, sp_phi, sp_a, sp_b = dlg._row_spins[0]
        sp_phi.setValue(4.4)
        sp_a.setValue(4.14)
        sp_b.setValue(4.20)
        dlg._apply_same_for_all()
        out = dlg.result_configs()
        assert out["BNO"]["a_angstrom"] == pytest.approx(4.14)
        assert out["BNO"]["b_angstrom"] == pytest.approx(4.20)
        assert out["Au_ref"]["work_function_eV"] == pytest.approx(4.4)
        assert out["Au_ref"]["b_angstrom"] == pytest.approx(4.20)

    def test_result_configs_includes_b(self):
        dlg = _dlg()
        name, sp_phi, sp_a, sp_b = dlg._row_spins[0]
        sp_a.setValue(3.96)
        sp_b.setValue(4.10)
        out = dlg.result_configs()
        assert out[name] == {"a_angstrom": pytest.approx(3.96),
                             "b_angstrom": pytest.approx(4.10)}

    def test_prefill_b_from_existing(self):
        dlg = _dlg(existing={"BNO": {"a_angstrom": 3.96, "b_angstrom": 4.10}})
        _name, _phi, sp_a, sp_b = dlg._row_spins[0]
        assert sp_a.value() == pytest.approx(3.96)
        assert sp_b.value() == pytest.approx(4.10)

    def test_single_mode_returns_root_key(self):
        dlg = _dlg(subfolders=[], detected_mode="single", n_root_files=2)
        dlg.sp_phi_single.setValue(4.5)
        assert dlg.result_configs() == {"": {"work_function_eV": 4.5}}

    def test_skip_returns_nothing_when_blank(self):
        dlg = _dlg()
        assert dlg.result_configs() == {}

    def test_prefill_from_existing_config(self):
        dlg = _dlg(existing={"BNO": {"work_function_eV": 4.3, "a_angstrom": 4.14}})
        name, sp_phi, sp_a, sp_b = dlg._row_spins[0]
        assert name == "BNO"
        assert sp_phi.value() == pytest.approx(4.3)
        assert "saved sample setup" in sp_phi.toolTip()

    def test_prefill_from_session_default(self):
        dlg = _dlg(session_default={"work_function_eV": 4.5})
        _, sp_phi, _, _ = dlg._row_spins[0]
        assert sp_phi.value() == pytest.approx(4.5)
        assert "session/logbook" in sp_phi.toolTip()

    def test_unusual_phi_highlighted(self):
        dlg = _dlg()
        _, sp_phi, _, _ = dlg._row_spins[0]
        sp_phi.setValue(12.0)
        assert "background" in sp_phi.styleSheet()
        sp_phi.setValue(4.4)
        assert sp_phi.styleSheet() == ""

    def test_mode_switch_carries_values(self):
        dlg = _dlg()
        _, sp_phi, _, _ = dlg._row_spins[0]
        sp_phi.setValue(4.2)
        dlg.rb_single.setChecked(True)
        assert dlg.sp_phi_single.value() == pytest.approx(4.2)


def _xlsx(tmp_path, name="logbook.xlsx", sheets=("CA041", "CA046")):
    import pandas as pd
    p = tmp_path / name
    with pd.ExcelWriter(p) as xw:
        for sh in sheets:
            pd.DataFrame({"File": ["f1", "f2"], "hv": [21.2, 48.0]}).to_excel(
                xw, sheet_name=sh, index=False)
    return p


class TestLogbookColumn:
    def test_autodetect_lone_xlsx(self, tmp_path):
        _xlsx(tmp_path)
        dlg = _dlg(subfolders=[("CA041", 5), ("CA046", 4)],
                   folder_path=str(tmp_path))
        assert dlg.ed_logbook.text().endswith("logbook.xlsx")
        assert dlg._global_sheets == ["CA041", "CA046"]

    def test_sheets_matched_to_samples_by_name(self, tmp_path):
        _xlsx(tmp_path)
        dlg = _dlg(subfolders=[("CA041", 5), ("CA046", 4)],
                   folder_path=str(tmp_path))
        out = {d["rel"]: d["sheet"] for d in dlg.result_logbooks()}
        assert out == {"CA041": "CA041", "CA046": "CA046"}

    def test_unmatched_sample_defaults_none(self, tmp_path):
        _xlsx(tmp_path)
        dlg = _dlg(subfolders=[("Au_ref", 3)], folder_path=str(tmp_path))
        assert dlg.result_logbooks() == []

    def test_mode_b_per_row_file(self, tmp_path):
        glob = _xlsx(tmp_path)  # global
        other = _xlsx(tmp_path, name="CA041_custom.xlsx", sheets=("Data",))
        dlg = _dlg(subfolders=[("CA041", 5), ("CA046", 4)],
                   folder_path=str(tmp_path))
        # two xlsx in the folder: auto-detection abstains -> set explicitly
        dlg.ed_logbook.setText(str(glob))
        dlg._reload_global_sheets()
        # simulate "Other file…" outcome for CA041
        dlg._row_files["CA041"] = (str(other), "Data")
        dlg._refresh_logbook_combos()
        out = {d["rel"]: (d["path"], d["sheet"]) for d in dlg.result_logbooks()}
        assert out["CA041"] == (str(other), "Data")
        assert out["CA046"][1] == "CA046"  # still global sheet

    def test_saved_sheet_missing_warns_not_silent(self, tmp_path):
        p = _xlsx(tmp_path, sheets=("CA041_v2",))
        dlg = _dlg(subfolders=[("CA041", 5)], folder_path=str(tmp_path),
                   existing_logbooks={"CA041": {"path": str(p), "sheet": "CA041_v1"}})
        rel, cmb = dlg._row_logbooks[0]
        assert cmb.currentText() == "None"
        assert "not found" in cmb.toolTip()
        assert dlg.result_logbooks() == []

    def test_corrupt_xlsx_loud_not_crash(self, tmp_path):
        bad = tmp_path / "bad.xlsx"
        bad.write_bytes(b"not an excel file")
        dlg = _dlg(subfolders=[("CA041", 5)], folder_path=str(tmp_path))
        assert dlg._global_sheets == []
        assert "Cannot read" in dlg.ed_logbook.toolTip()
        assert dlg.result_logbooks() == []

    def test_single_mode_logbook_rel_empty(self, tmp_path):
        _xlsx(tmp_path, sheets=("Global",))
        dlg = _dlg(subfolders=[], detected_mode="single", n_root_files=3,
                   folder_path=str(tmp_path))
        out = dlg.result_logbooks()
        assert out == [{"rel": "", "path": dlg.ed_logbook.text(), "sheet": "Global"}]

    def test_browse_only_flag(self):
        dlg = _dlg()
        assert dlg.browse_only_requested is False
        dlg._on_browse_only()
        assert dlg.browse_only_requested is True
        assert dlg.result() == 0  # rejected
