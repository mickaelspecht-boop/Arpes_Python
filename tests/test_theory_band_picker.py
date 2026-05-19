from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from arpes.theory.band_picker import (
    picker_band_curves,
    picker_k_axis,
    picker_segment_span,
    picker_ticks,
    path_label,
    validate_picker_data,
)
from arpes.theory.models import TheoryBandData, TheoryOverlayConfig

try:
    from PyQt6.QtWidgets import QApplication
    from arpes.ui.widgets.dialogs.theory_band_picker import TheoryBandPickerDialog
    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False


def _data() -> TheoryBandData:
    return TheoryBandData(
        source="materials_project",
        material_id="mp-test",
        k_distance=[0.0, 0.5, 1.0],
        bands=[
            [-1.0, -0.8, -0.6],
            [-0.2, 0.0, 0.2],
            [0.7, 0.8, 0.9],
        ],
        branches=[{"name": "\\Gamma-X", "start": 0, "end": 2}],
    )


class TestPickerPure:
    def test_validate_rejects_shape_mismatch(self):
        data = TheoryBandData(
            source="local",
            material_id="bad",
            k_distance=[0.0, 1.0, 2.0],
            bands=[[0.0, 1.0]],
        )
        assert "incompatible" in validate_picker_data(data)

    def test_validate_rejects_non_finite_k(self):
        data = TheoryBandData(
            source="local",
            material_id="bad",
            k_distance=[0.0, float("nan"), 2.0],
            bands=[[0.0, 1.0, 2.0]],
        )
        assert "axe k non fini" in validate_picker_data(data)

    def test_legacy_without_band_meta_still_builds_curves(self):
        legacy = {
            "source": "materials_project",
            "material_id": "mp-old",
            "k_distance": [0.0, 1.0],
            "bands": [[-1.0, 0.0], [1.0, 2.0]],
        }
        curves = picker_band_curves(legacy, TheoryOverlayConfig(enabled=True))
        assert [c.band_index for c in curves] == [0, 1]

    def test_picker_prefers_absolute_mp_axis(self):
        data = TheoryBandData(
            source="materials_project",
            material_id="mp-axis",
            k_distance=[-1.0, 0.0, 1.0],
            k_distance_abs=[0.0, 0.8, 2.4],
            bands=[[0.0, 0.1, 0.2]],
            branches=[{"name": "\\Gamma-X", "start": 0, "end": 2}],
        )
        assert picker_k_axis(data).tolist() == [0.0, 0.8, 2.4]
        curves = picker_band_curves(data, TheoryOverlayConfig(enabled=True, segment="Γ-X"))
        assert curves[0].k.tolist() == [0.0, 0.8, 2.4]

    def test_picker_ticks_and_segment_span_use_global_path(self):
        data = TheoryBandData(
            source="materials_project",
            material_id="mp-axis",
            k_distance=[-1.0, -0.3, 0.2, 1.0],
            k_distance_abs=[0.0, 1.0, 1.8, 3.0],
            bands=[[0.0, 0.1, 0.2, 0.3]],
            branches=[
                {"name": "\\Gamma-X", "start": 0, "end": 1},
                {"name": "X-M", "start": 1, "end": 3},
            ],
        )
        assert [(t.x, t.label) for t in picker_ticks(data)] == [
            (0.0, "Γ"),
            (1.0, "X"),
            (3.0, "M"),
        ]
        assert picker_segment_span(data, "X-M") == (1.0, 3.0)

    def test_arpes_pnictide_convention_keeps_mp_label_with_alias(self):
        assert path_label("Y", "arpes_pnictides") == "Y\n(M)"
        assert path_label("P", "arpes_pnictides") == "P\n(M/S)"
        assert path_label("Y", "mp_bulk") == "Y"


@pytest.mark.skipif(not UI_AVAILABLE, reason="PyQt6 / Qt offscreen indisponible")
class TestTheoryBandPickerDialog(unittest.TestCase):
    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def test_instantiates_with_fake_band_data(self):
        dlg = TheoryBandPickerDialog(
            _data(),
            TheoryOverlayConfig(enabled=True, segment="Γ-X", band_indices="1"),
            segments=["Γ-X"],
            selected=[1],
        )
        self.assertIsNotNone(dlg.canvas)
        self.assertEqual(dlg.selected_band_indices(), [1])
        self.assertIn(1, dlg._lines)

    def test_toggle_band_updates_selection(self):
        dlg = TheoryBandPickerDialog(_data(), TheoryOverlayConfig(enabled=True))
        dlg._toggle_band(1)
        self.assertEqual(dlg.selected_band_indices(), [1])
        dlg._toggle_band(1)
        self.assertEqual(dlg.selected_band_indices(), [])

    def test_apply_emits_indices_and_segment(self):
        dlg = TheoryBandPickerDialog(
            _data(),
            TheoryOverlayConfig(enabled=True, segment="Γ-X"),
            segments=["Γ-X"],
            selected=[1, 2],
        )
        got = []
        dlg.selection_applied.connect(lambda indices, segment: got.append((indices, segment)))
        dlg._apply()
        self.assertEqual(got, [([1, 2], "Γ-X")])

    def test_deep_bands_default_to_mp_like_energy_window(self):
        data = TheoryBandData(
            source="materials_project",
            material_id="mp-deep",
            k_distance=[0.0, 1.0],
            bands=[[-70.0, -69.0], [-2.0, 2.0], [30.0, 35.0]],
        )
        dlg = TheoryBandPickerDialog(data, TheoryOverlayConfig(enabled=True))
        y0, y1 = dlg.canvas.ax.get_ylim()
        self.assertEqual((y0, y1), (-5.0, 9.0))

    def test_scroll_zoom_changes_limits(self):
        dlg = TheoryBandPickerDialog(_data(), TheoryOverlayConfig(enabled=True))
        before = dlg.canvas.ax.get_xlim(), dlg.canvas.ax.get_ylim()
        dlg._zoom_axis(dlg.canvas.ax, 0.5, 0.0, 0.8)
        after = dlg.canvas.ax.get_xlim(), dlg.canvas.ax.get_ylim()
        self.assertNotEqual(before, after)
