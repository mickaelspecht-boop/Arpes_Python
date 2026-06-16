"""Tests for the compact multi-zone table widget (Qt offscreen)."""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from arpes.ui.widgets.zones_strip import ZonesStrip

app = QApplication.instance() or QApplication([])


def _zones():
    return [
        {
            "id": "a1", "label": "Z1", "color_idx": 0, "active": True,
            "fit_params": {"k_min": -0.4, "k_max": 0.4,
                           "ev_start": -0.5, "ev_end": -0.01},
            "fit_result": {"e_fitted": [0.0, -0.1, -0.2]},
        },
        {
            "id": "b2", "label": "Z2", "color_idx": 1, "active": False,
            "fit_params": {"k_min": 0.1, "k_max": 0.6,
                           "ev_start": -0.5, "ev_end": -0.01},
            "fit_result": None,
        },
    ]


class TestSetZones:
    def test_populates_rows_and_selects_active(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        assert s.tbl.rowCount() == 2
        assert s.current_zone_id() == "a1"

    def test_window_and_fit_columns(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        assert s.tbl.item(0, 2).text().startswith("k[")
        assert s.tbl.item(0, 3).text() == "✓ 3 pts"
        assert s.tbl.item(1, 3).text() == "—"

    def test_active_checkbox_reflects_state(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        assert s.tbl.item(0, 0).checkState() == Qt.CheckState.Checked
        assert s.tbl.item(1, 0).checkState() == Qt.CheckState.Unchecked

    def test_empty_disables_buttons(self):
        s = ZonesStrip()
        s.set_zones([], None)
        assert not s.btn_remove.isEnabled()
        assert not s.btn_run_all.isEnabled()
        assert s.current_zone_id() is None


class TestSignals:
    def test_selecting_row_emits_active_changed(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        seen = []
        s.active_zone_changed.connect(seen.append)
        s.tbl.selectRow(1)
        assert seen == ["b2"]
        assert s.current_zone_id() == "b2"

    def test_toggle_checkbox_emits_toggle(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        seen = []
        s.toggle_zone_active.connect(lambda zid, on: seen.append((zid, on)))
        s.tbl.item(1, 0).setCheckState(Qt.CheckState.Checked)
        assert seen == [("b2", True)]

    def test_edit_name_emits_rename(self):
        s = ZonesStrip()
        s.set_zones(_zones(), "a1")
        seen = []
        s.rename_zone_requested.connect(lambda zid, label: seen.append((zid, label)))
        s.tbl.item(0, 1).setText("alpha")
        assert seen == [("a1", "alpha")]

    def test_set_zones_does_not_emit(self):
        """Rebuilding the table must not emit selection/toggle/rename signals."""
        s = ZonesStrip()
        seen = []
        s.active_zone_changed.connect(seen.append)
        s.toggle_zone_active.connect(lambda *a: seen.append(a))
        s.rename_zone_requested.connect(lambda *a: seen.append(a))
        s.set_zones(_zones(), "a1")
        assert seen == []
