"""E2E workflow Γ : drift, idempotence, persistance, gardes.

Plan d'audit Phase 1 — fixe `GAMMA_AUDIT_PLAN.md`.

Deux familles :
- Sans Qt : drift au reload (persistance meta_gamma_state), idempotence pure de
  `apply_bm_gamma_axis_shift` + `_shift_fit_result_in_place`.
- Avec Qt (skipées si PyQt6 absent) : gardes des handlers controller.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arpes.core.session import FileEntry, Session
from arpes.physics.gamma import apply_bm_gamma_axis_shift

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    from arpes.ui.controllers.gamma_controller import (
        GammaController,
        _shift_fit_result_in_place,
    )

    QT_AVAILABLE = True
except Exception:
    QT_AVAILABLE = False
    # Re-import the pure helper for the no-Qt subset (defined in gamma_controller
    # mais utilisable sans Qt si on stubbe). Pour rester simple en headless on
    # le redéfinit ; le vrai test du remap se fait dans le bloc Qt.
    def _shift_fit_result_in_place(fr, delta):
        if not fr or abs(delta) < 1e-12:
            return
        for key in ("kF_minus", "kF_plus", "gamma_corrige"):
            if not fr.get(key):
                continue
            fr[key] = [
                [(float(v) - delta) if v is not None else v for v in row]
                for row in fr[key]
            ]


# --------------------------------------------------------------------------
# Bloc A — sans Qt — bugs de persistance + drift au reload
# --------------------------------------------------------------------------

class TestGammaMetaPersistRoundTrip(unittest.TestCase):
    """Audit P1.1 : `meta.bm_gamma_axis_*` doit survivre save/load JSON.

    Actuellement les flags vivent dans `raw_data["metadata"]` (transient) et
    ne sont jamais sérialisés dans FileEntry → au reload, l'app croit l'axe
    brut et ré-applique le shift → drift.
    """

    def test_meta_gamma_state_persists_in_file_entry(self):
        session = Session(Path("/tmp/audit_gamma_e2e"))
        entry = session.get_or_create("bm04")

        # Simule ce qu'un shift d'axe doit déposer sur l'entry pour survivre
        # un save/load. P1.1 ajoutera `entry.meta_gamma_state`.
        entry.meta_gamma_state = {
            "bm_gamma_axis_centered": True,
            "bm_gamma_axis_shift": 0.07,
            "bm_gamma_axis_note": "kpar_display = kpar_raw - gamma_bm",
            "bm_gamma_reference_source": "fs_auto",
            "bm_gamma_reference_path": "/tmp/audit_gamma_e2e/fs1",
        }

        payload = session.to_payload()
        roundtrip = Session(Path("/tmp/audit_gamma_e2e"))
        roundtrip.load_from_payload(json.loads(json.dumps(payload)))

        restored = roundtrip.files["bm04"]
        state = getattr(restored, "meta_gamma_state", None)
        self.assertIsInstance(state, dict, "FileEntry.meta_gamma_state perdu au reload")
        self.assertTrue(state.get("bm_gamma_axis_centered"))
        self.assertAlmostEqual(state.get("bm_gamma_axis_shift"), 0.07)


class TestNoDriftOnReload(unittest.TestCase):
    """Audit P1.1 : reload après fit + Γ ne doit PAS re-décaler les kF.

    Reproduit : kpar raw → shift Γ delta=0.1 → kF remappés une fois →
    save/load → restore meta → ré-application stored gamma. Le delta calculé
    doit valoir 0 (déjà appliqué d'après le state restauré), donc kF
    inchangé. Sans persistance du `bm_gamma_axis_shift`, delta=0.1 sera
    réappliqué → kF re-décalés.
    """

    def _make_raw(self, kpar_raw, *, shift=0.0, centered=False):
        meta = {"scan_kind": "BM"}
        if centered:
            meta["bm_gamma_axis_centered"] = True
            meta["bm_gamma_axis_shift"] = shift
        return {
            "path": "/tmp/audit_gamma_e2e/bm04",
            "hv": 60.0,
            "kpar": np.array(kpar_raw, dtype=float),
            "metadata": meta,
        }

    def test_axis_shift_idempotent_when_meta_state_restored(self):
        kpar_raw = [-1.0, 0.0, 1.0]
        delta_initial = 0.1
        fit_result = {
            "kF_minus": [[-0.50, -0.55]],
            "kF_plus": [[0.50, 0.55]],
            "gamma_corrige": [[0.0, 0.0]],
        }
        # Initial: shift l'axe + remap kF
        raw = self._make_raw(kpar_raw)
        applied = apply_bm_gamma_axis_shift(raw, delta_initial, ref={"source": "fs"})
        self.assertTrue(applied)
        _shift_fit_result_in_place(fit_result, delta_initial)
        self.assertAlmostEqual(fit_result["kF_minus"][0][0], -0.60)

        # Save: meta state que l'app DOIT persister (P1.1)
        meta_state = {
            k: raw["metadata"][k]
            for k in raw["metadata"]
            if k.startswith("bm_gamma_") or k.startswith("fs_gamma_")
        }

        # Reload : nouveau raw_data brut + restauration du meta state via
        # le mécanisme P1.1 (restore_meta_gamma_state)
        raw2 = self._make_raw(kpar_raw)
        raw2["metadata"].update(meta_state)
        # Simulation de l'opération "appliquer le shift stocké"
        # (post-fix : delta=0 puisque previous == new)
        # On vérifie que apply ne re-shifte pas
        kpar_before = raw2["kpar"].copy()
        ok = apply_bm_gamma_axis_shift(raw2, delta_initial, ref={"source": "fs"})
        np.testing.assert_allclose(
            raw2["kpar"], kpar_before,
            err_msg="kpar a été re-décalé alors que le state était restauré"
        )
        # Pas de delta supplémentaire à propager à fit_result
        _shift_fit_result_in_place(fit_result, 0.0 if not ok else delta_initial)
        self.assertAlmostEqual(
            fit_result["kF_minus"][0][0], -0.60,
            msg="fit_result re-décalé au reload : drift cumulatif"
        )


# --------------------------------------------------------------------------
# Bloc B — avec Qt — gardes idempotence sur les détecteurs
# --------------------------------------------------------------------------

@unittest.skipUnless(QT_AVAILABLE, "PyQt6 indisponible — bloc B skip")
class TestGammaDetectorsGuard(unittest.TestCase):
    """Audit P1.2 : `_detect_fs_gamma` / `_estimate_gamma_bm` / `_on_fs_map_click`
    doivent refuser si l'axe est déjà recentré ou si le loader a appliqué un
    offset angulaire.
    """

    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def _make_parent_fs(self, *, centered=False, loader_offset=False):
        class FakeControls:
            def __init__(self):
                self.center = (0.0, 0.0)
                self.lbl_info = SimpleNamespace(setText=lambda text: None)

            def params(self):
                return SimpleNamespace(kx_center=self.center[0], ky_center=self.center[1])

            def set_center(self, kx, ky):
                self.center = (float(kx), float(ky))

        class FakeCanvas:
            ax = object()

            def detect_gamma(self, raw, params):
                return {
                    "kx": 0.30, "ky": 0.10, "quality": "ok",
                    "gamma_kx_list": [0.30], "gamma_ky_list": [0.10],
                    "gamma_delta_kx": 0.0,
                }

        class FakeSpin:
            def __init__(self):
                self.value_seen = None

            def setValue(self, v):
                self.value_seen = float(v)

        meta = {
            "fs_data": object(),
            "fs_kx": np.array([-1.0, 0.0, 1.0]),
            "fs_ky": np.array([-1.0, 0.0, 1.0]),
            "fs_kind": "kxky",
        }
        if centered:
            meta["bm_gamma_axis_centered"] = True
            meta["bm_gamma_axis_shift"] = 0.1
            meta["fs_gamma_axis_centered"] = True
            meta["fs_gamma_axis_shift_kx"] = 0.1
            meta["fs_gamma_axis_shift_ky"] = 0.0
        if loader_offset:
            meta["angle_offsets_applied"] = {"theta0_deg": 0.5, "candidate": "loader_auto"}

        class Parent:
            def __init__(self):
                self._raw_data = {
                    "path": "/tmp/fs1",
                    "hv": 60.0,
                    "kpar": np.array([-1.0, 0.0, 1.0]),
                    "metadata": meta,
                }
                self._current_path = "/tmp/fs1"
                self._session = Session(Path("/tmp"))
                self._fs_controls = FakeControls()
                self._fs_canvas = FakeCanvas()
                self._sp_cx = FakeSpin()
                self._params = SimpleNamespace(
                    sp_hv=SimpleNamespace(value=lambda: 60.0),
                    sp_phi=SimpleNamespace(value=lambda: 4.5),
                    sp_cx=self._sp_cx,
                    mark_action_done=lambda text: None,
                )
                self.status_messages = []

            def _current_entry(self):
                return self._session.get_or_create(self._session.key_for_path(self._current_path))

            def _same_path(self, a, b):
                return a == b

            def _current_is_fs(self):
                return True

            def _draw_fs_tab(self):
                pass

            def _status(self, text):
                self.status_messages.append(text)

        return Parent()

    def test_detect_fs_gamma_refused_when_axis_centered(self):
        parent = self._make_parent_fs(centered=True)
        ctrl = GammaController(parent)
        ref_before = dict(parent._session.gamma_reference)

        ctrl._detect_fs_gamma()

        # Garde P1.2 : ne touche ni la réf ni le center
        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "réf Γ écrasée malgré axe déjà recentré")
        self.assertEqual(parent._fs_controls.center, (0.0, 0.0),
                         "centre FS muté malgré garde")
        joined = " ".join(parent.status_messages).lower()
        self.assertTrue(
            "déjà appliqué" in joined or "axe recentré" in joined or "offset" in joined,
            f"pas de warning explicite ; messages: {parent.status_messages}"
        )

    def test_detect_fs_gamma_refused_when_loader_offset_applied(self):
        parent = self._make_parent_fs(loader_offset=True)
        ctrl = GammaController(parent)
        ref_before = dict(parent._session.gamma_reference)

        ctrl._detect_fs_gamma()

        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "réf Γ écrasée malgré loader-offset actif")
        joined = " ".join(parent.status_messages).lower()
        self.assertTrue(
            "déjà appliqué" in joined or "loader" in joined or "offset" in joined,
            f"pas de warning loader ; messages: {parent.status_messages}"
        )

    def test_on_fs_map_click_refused_when_axis_centered(self):
        parent = self._make_parent_fs(centered=True)
        parent._fs_pick_center_active = True
        ctrl = GammaController(parent)
        event = SimpleNamespace(inaxes=parent._fs_canvas.ax, xdata=0.20, ydata=0.10)
        ref_before = dict(parent._session.gamma_reference)

        ctrl._on_fs_map_click(event)

        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "click pick a muté la réf malgré axe centré")


if __name__ == "__main__":
    unittest.main()
