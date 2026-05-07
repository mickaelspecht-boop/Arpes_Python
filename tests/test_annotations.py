from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from arpes.core.session import FileEntry, Session
from arpes.ui.controllers.interaction_controller import InteractionController


class TestFitAnnotations(unittest.TestCase):
    def test_annotations_round_trip_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = FileEntry()
            entry.annotations = {
                "kF_minus": [
                    {"pair": 0, "index": 1, "text": "Bande split", "ts": "2026-05-07T00:00:00+00:00"}
                ],
                "kF_plus": [],
            }
            session.files["BM1"] = entry
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertEqual(restored.files["BM1"].annotations["kF_minus"][0]["text"], "Bande split")
            self.assertEqual(restored.files["BM1"].annotations["kF_minus"][0]["pair"], 0)
            self.assertEqual(restored.files["BM1"].annotations["kF_minus"][0]["index"], 1)

    def test_annotations_can_be_removed_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Session(root)
            entry = session.get_or_create("BM1")
            entry.annotations = {
                "kF_minus": [{"pair": 0, "index": 0, "text": "obsolete", "ts": "t"}],
            }
            entry.annotations["kF_minus"] = []
            session.save()

            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            self.assertEqual(restored.files["BM1"].annotations["kF_minus"], [])

    def test_right_click_adds_annotation_to_nearest_fit_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "BM1"
            data_path.write_text("dummy")

            fig = Figure()
            ax = fig.add_subplot(111)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 0.1)
            fig.canvas.draw()

            parent = SimpleNamespace(
                _fit_roi_active=False,
                _bm_canvas=SimpleNamespace(ax=ax),
                _mdc_map_canvas=SimpleNamespace(ax=None),
                _fit_res={
                    "e_fitted": np.array([-0.2, -0.1]),
                    "kF_minus": [np.array([0.1, 0.2])],
                    "kF_plus": [],
                },
                _current_path=str(data_path),
                _session=Session(root),
                _draw_bm=lambda: None,
                _draw_mdc_edc=lambda: None,
                _tabs=None,
            )
            statuses: list[str] = []
            parent._status = statuses.append
            event = SimpleNamespace(
                inaxes=ax,
                button=3,
                xdata=0.2,
                ydata=-0.1,
            )

            with patch(
                "arpes.ui.controllers.interaction_controller.QInputDialog.getMultiLineText",
                return_value=("Point important", True),
            ):
                InteractionController(parent)._on_fit_annotate_press(event)

            entry = parent._session.files["BM1"]
            self.assertEqual(entry.annotations["kF_minus"][0]["index"], 1)
            self.assertEqual(entry.annotations["kF_minus"][0]["text"], "Point important")
            self.assertEqual(statuses[-1], "Annotation enregistrée.")


if __name__ == "__main__":
    unittest.main()
