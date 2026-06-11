"""Tests for the BZ label convention dialog (Qt offscreen)."""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from arpes.ui.widgets.dialogs.bz_labels_dialog import BZLabelsDialog

app = QApplication.instance() or QApplication([])


def _ok_button(dlg):
    return dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)


class TestBZLabelsDialog:
    def test_canonical_labels_square(self):
        dlg = BZLabelsDialog(shape="square")
        assert set(dlg._edits) == {"X", "M"}

    def test_identity_gives_empty_overrides(self):
        dlg = BZLabelsDialog(shape="square")
        assert dlg.overrides() == {}
        assert _ok_button(dlg).isEnabled()

    def test_preset_fills_fields(self):
        dlg = BZLabelsDialog(shape="square")
        idx = dlg.cmb_preset.findData("i4mmm_sigma_diagonal")
        dlg.cmb_preset.setCurrentIndex(idx)
        assert dlg._edits["M"].text() == "Σ"
        assert dlg.overrides() == {"M": "Σ"}
        assert dlg.preset_key() == "i4mmm_sigma_diagonal"

    def test_duplicate_labels_block_ok(self):
        dlg = BZLabelsDialog(shape="square")
        dlg._edits["X"].setText("Σ")
        dlg._edits["M"].setText("Σ")
        assert not _ok_button(dlg).isEnabled()
        assert "unique" in dlg._lbl_error.text()

    def test_empty_label_blocks_ok(self):
        dlg = BZLabelsDialog(shape="square")
        dlg._edits["X"].setText("")
        assert not _ok_button(dlg).isEnabled()

    def test_existing_overrides_prefill(self):
        dlg = BZLabelsDialog(shape="square", current_overrides={"M": "Σ"},
                             current_preset="i4mmm_sigma_diagonal")
        assert dlg._edits["M"].text() == "Σ"
        assert dlg.preset_key() == "i4mmm_sigma_diagonal"

    def test_rectangle_has_three_labels(self):
        dlg = BZLabelsDialog(shape="rectangle")
        assert set(dlg._edits) == {"X", "Y", "S"}
