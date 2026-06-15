"""Γ E2E workflow: drift, idempotence, persistence, guards.

Phase 1 audit plan — fixes `GAMMA_AUDIT_PLAN.md`.

Two families:
- Without Qt: reload drift (meta_gamma_state persistence), pure idempotence of
  `apply_bm_gamma_axis_shift` + `_shift_fit_result_in_place`.
- With Qt (skipped if PyQt6 is missing): controller handler guards.
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
    # but usable without Qt when stubbed). To keep headless simple, redefine it;
    # the real remap test is in the Qt block.
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
# Block A — without Qt — persistence bugs + reload drift
# --------------------------------------------------------------------------

class TestGammaMetaPersistRoundTrip(unittest.TestCase):
    """Audit P1.1: `meta.bm_gamma_axis_*` must survive JSON save/load.

    The flags currently live in `raw_data["metadata"]` (transient) and are never
    serialized into FileEntry → on reload, the app thinks the axis is raw and
    reapplies the shift → drift.
    """

    def test_meta_gamma_state_persists_in_file_entry(self):
        session = Session(Path("/tmp/audit_gamma_e2e"))
        entry = session.get_or_create("bm04")

        # Simulates what an axis shift must store on the entry to survive
        # save/load. P1.1 will add `entry.meta_gamma_state`.
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
        self.assertIsInstance(state, dict, "FileEntry.meta_gamma_state lost on reload")
        self.assertTrue(state.get("bm_gamma_axis_centered"))
        self.assertAlmostEqual(state.get("bm_gamma_axis_shift"), 0.07)


class TestNoDriftOnReload(unittest.TestCase):
    """Audit P1.1: reload after fit + Γ must NOT re-shift kF.

    Reproduces: raw kpar → Γ shift delta=0.1 → kF remapped once → save/load →
    restore meta → re-apply stored gamma. The computed delta must be 0 (already
    applied according to restored state), so kF is unchanged. Without
    `bm_gamma_axis_shift` persistence, delta=0.1 would be reapplied → kF
    re-shifted.
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
        # Initial: shift the axis + remap kF.
        raw = self._make_raw(kpar_raw)
        applied = apply_bm_gamma_axis_shift(raw, delta_initial, ref={"source": "fs"})
        self.assertTrue(applied)
        _shift_fit_result_in_place(fit_result, delta_initial)
        self.assertAlmostEqual(fit_result["kF_minus"][0][0], -0.60)

        # Save: meta state that the app MUST persist (P1.1).
        meta_state = {
            k: raw["metadata"][k]
            for k in raw["metadata"]
            if k.startswith("bm_gamma_") or k.startswith("fs_gamma_")
        }

        # Reload: new raw raw_data + meta state restoration through the P1.1
        # mechanism (restore_meta_gamma_state).
        raw2 = self._make_raw(kpar_raw)
        raw2["metadata"].update(meta_state)
        # Simulation of the "apply stored shift" operation.
        # (post-fix: delta=0 because previous == new)
        # Verify that apply does not re-shift.
        kpar_before = raw2["kpar"].copy()
        ok = apply_bm_gamma_axis_shift(raw2, delta_initial, ref={"source": "fs"})
        np.testing.assert_allclose(
            raw2["kpar"], kpar_before,
            err_msg="kpar was re-shifted even though state was restored"
        )
        # No extra delta to propagate to fit_result.
        _shift_fit_result_in_place(fit_result, 0.0 if not ok else delta_initial)
        self.assertAlmostEqual(
            fit_result["kF_minus"][0][0], -0.60,
            msg="fit_result re-shifted on reload: cumulative drift"
        )


# --------------------------------------------------------------------------
# Block B — with Qt — detector idempotence guards
# --------------------------------------------------------------------------

@unittest.skipUnless(QT_AVAILABLE, "PyQt6 unavailable — skipping block B")
class TestGammaDetectorsGuard(unittest.TestCase):
    """Audit P1.2 : `_detect_fs_gamma` / `_estimate_gamma_bm` / `_on_fs_map_click`
    must refuse if the axis is already recentered or if the loader applied an
    angular offset.
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

        # P1.2 guard: touches neither the reference nor the center.
        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "Γ ref overwritten despite already recentered axis")
        self.assertEqual(parent._fs_controls.center, (0.0, 0.0),
                         "FS center mutated despite guard")
        joined = " ".join(parent.status_messages).lower()
        self.assertTrue(
            "already applied" in joined or "axis recentered" in joined or "offset" in joined,
            f"no explicit warning; messages: {parent.status_messages}"
        )

    def test_detect_fs_gamma_refused_when_loader_offset_applied(self):
        parent = self._make_parent_fs(loader_offset=True)
        ctrl = GammaController(parent)
        ref_before = dict(parent._session.gamma_reference)

        ctrl._detect_fs_gamma()

        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "Γ ref overwritten despite active loader offset")
        joined = " ".join(parent.status_messages).lower()
        self.assertTrue(
            "already applied" in joined or "loader" in joined or "offset" in joined,
            f"no loader warning; messages: {parent.status_messages}"
        )

    def test_on_fs_map_click_refused_when_axis_centered(self):
        parent = self._make_parent_fs(centered=True)
        parent._fs_pick_center_active = True
        ctrl = GammaController(parent)
        event = SimpleNamespace(inaxes=parent._fs_canvas.ax, xdata=0.20, ydata=0.10)
        ref_before = dict(parent._session.gamma_reference)

        ctrl._on_fs_map_click(event)

        self.assertEqual(parent._session.gamma_reference, ref_before,
                         "click pick mutated the ref despite centered axis")


@unittest.skipUnless(QT_AVAILABLE, "PyQt6 unavailable — skipping block C")
class TestForgetGamma(unittest.TestCase):
    """Audit P2.bis: `_forget_gamma` must reverse the axis shift and clear all
    flags, allowing Γ to be re-detected after a guard.
    """

    _qt_app = None

    @classmethod
    def setUpClass(cls):
        cls._qt_app = QApplication.instance() or QApplication([])

    def test_forget_after_shift_restores_kpar_and_clears_state(self):
        class FakeSpin:
            def __init__(self, v=0.0):
                self.value_seen = v

            def setValue(self, v):
                self.value_seen = float(v)

            def blockSignals(self, b):
                return False

        class FakeControls:
            def __init__(self):
                self.center = (0.3, 0.0)

            def set_center(self, kx, ky):
                self.center = (float(kx), float(ky))

        class Parent:
            def __init__(self):
                # kpar already shifted by 0.3 (post auto-Γ-BM)
                self._raw_data = {
                    "path": "/tmp/bm04",
                    "hv": 60.0,
                    "kpar": np.array([-1.3, -0.3, 0.7]),
                    "metadata": {
                        "scan_kind": "BM",
                        "bm_gamma_axis_centered": True,
                        "bm_gamma_axis_shift": 0.3,
                    },
                }
                self._current_path = "/tmp/bm04"
                self._session = Session(Path("/tmp"))
                self._session.gamma_reference = {"kx": 0.3, "ky": 0.0, "path": "/tmp/bm04"}
                self._session.angle_offsets = {"theta0_deg": 1.0}
                self._fs_controls = FakeControls()
                self._sp_cx = FakeSpin(0.0)
                self._params = SimpleNamespace(sp_cx=self._sp_cx)

            def _current_entry(self):
                e = self._session.get_or_create(self._session.key_for_path(self._current_path))
                if not getattr(e, "_test_initialized", False):
                    e.meta_gamma_state = {"bm_gamma_axis_centered": True, "bm_gamma_axis_shift": 0.3}
                    e.fs_center_kx = 0.3
                    e.fs_center_ky = 0.0
                    e.fit_params.center_init = 0.0
                    # fit_result already shifted
                    e.fit_result = {"kF_minus": [[-0.8]], "kF_plus": [[0.2]], "gamma_corrige": [[-0.3]]}
                    e._test_initialized = True
                return e

            def _status(self, text):
                pass

            def _draw_current_view(self):
                pass

        parent = Parent()
        # warm up entry
        parent._current_entry()
        ctrl = GammaController(parent)

        ctrl._forget_gamma()

        # kpar restored.
        np.testing.assert_allclose(parent._raw_data["kpar"], [-1.0, 0.0, 1.0])
        # Empty session.
        self.assertEqual(parent._session.gamma_reference, {})
        self.assertEqual(parent._session.angle_offsets, {})
        # Empty entry.
        e = parent._current_entry()
        self.assertEqual(e.meta_gamma_state, {})
        self.assertIsNone(e.fs_center_kx)
        # fit_result remapped (delta -0.3 → kF += 0.3).
        self.assertAlmostEqual(e.fit_result["kF_minus"][0][0], -0.5, places=10)
        self.assertAlmostEqual(e.fit_result["kF_plus"][0][0], 0.5, places=10)
        # Meta flags cleared.
        self.assertFalse(parent._raw_data["metadata"].get("bm_gamma_axis_centered"))

    def test_forget_loader_baked_reloads_or_aborts(self):
        """Loader-baked Γ cannot be undone by axis arithmetic; forget() must
        clear the session refs and reload to rebuild a clean raw axis. Without a
        path it must abort with a visible message (not lie about clearing)."""
        reload_calls = []

        class FakeSpin:
            def setValue(self, v):
                pass

            def blockSignals(self, b):
                return False

        class FakeControls:
            def set_center(self, kx, ky):
                pass

        class Parent:
            def __init__(self, path):
                self._raw_data = {
                    "path": path,
                    "hv": 60.0,
                    "kpar": np.array([-1.0, 0.0, 1.0]),
                    "metadata": {
                        "scan_kind": "BM",
                        "angle_offsets_applied": {"theta0_deg": 0.5},
                    },
                }
                self._current_path = path
                self._session = Session(Path("/tmp"))
                self._session.gamma_reference = {"kx": 0.2, "ky": 0.0, "path": path}
                self._session.angle_offsets = {"theta0_deg": 0.5}
                self._fs_controls = FakeControls()
                self._params = SimpleNamespace(sp_cx=FakeSpin())
                self._status_msgs = []

            def _current_entry(self):
                if not self._current_path:
                    return None
                return self._session.get_or_create(
                    self._session.key_for_path(self._current_path))

            def _status(self, text):
                self._status_msgs.append(text)

            def _draw_current_view(self):
                pass

            def _reload_current_no_cache(self):
                reload_calls.append(self._current_path)

        # With a path: reload triggered, session refs cleared.
        parent = Parent("/tmp/bm04")
        GammaController(parent)._forget_gamma()
        self.assertEqual(reload_calls, ["/tmp/bm04"])
        self.assertEqual(parent._session.gamma_reference, {})
        self.assertEqual(parent._session.angle_offsets, {})

        # Without a path: no reload, explicit "cannot reload" status (no silent lie).
        reload_calls.clear()
        parent2 = Parent(None)
        GammaController(parent2)._forget_gamma()
        self.assertEqual(reload_calls, [])
        self.assertTrue(any("cannot reload" in m.lower() for m in parent2._status_msgs))


if __name__ == "__main__":
    unittest.main()
