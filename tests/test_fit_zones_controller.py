"""Phase 1 tests : CRUD + persistence + asymmetric warning for fit zones."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arpes.core.session import FileEntry, FitParams, Session
from arpes.ui.controllers.fit_zones_controller import (
    ZONE_PALETTE,
    FitZonesController,
)


class _StubParams:
    def __init__(self, fp: FitParams | None = None):
        self._fp = fp or FitParams()
        self.sp_cx = SimpleNamespace(value=lambda: 0.0)

    def get_fit_params(self) -> FitParams:
        return self._fp


class _StubParent:
    def __init__(self, session: Session, key: str):
        self._session = session
        self._current_path = Path(key)
        self._params = _StubParams()

    def _status(self, msg: str) -> None:
        pass


@pytest.fixture
def parent_and_ctrl(tmp_path):
    sess = Session(folder=tmp_path)
    key = "synthetic.h5"
    # ensure key_for_path resolves to a relative key
    p = SimpleNamespace(_session=sess, _current_path=tmp_path / key, _params=_StubParams())
    p._status = lambda m: None
    sess.get_or_create(key)
    ctrl = FitZonesController(p)
    return p, ctrl, key


class TestCRUD:
    def test_add_creates_zone_with_uuid_and_color(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        res = ctrl.fit_zone_action("add", {"label": "Z1"})
        assert res["ok"] and res["zone_id"]
        entry = p._session.files[key]
        assert len(entry.fit_zones) == 1
        assert entry.fit_zones[0]["id"] == res["zone_id"]
        assert entry.fit_zones[0]["color_idx"] == 0
        assert entry.active_zone_id == res["zone_id"]

    def test_add_two_zones_distinct_colors(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        ctrl.fit_zone_action("add", {})
        ctrl.fit_zone_action("add", {})
        entry = p._session.files[key]
        assert {z["color_idx"] for z in entry.fit_zones} == {0, 1}

    def test_remove_drops_zone_and_reassigns_active(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        z1 = ctrl.fit_zone_action("add", {})["zone_id"]
        z2 = ctrl.fit_zone_action("add", {})["zone_id"]
        ctrl.fit_zone_action("set_active", {"zone_id": z1})
        ctrl.fit_zone_action("remove", {"zone_id": z1})
        entry = p._session.files[key]
        assert len(entry.fit_zones) == 1
        assert entry.active_zone_id == z2

    def test_toggle_active(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        zid = ctrl.fit_zone_action("add", {})["zone_id"]
        ctrl.fit_zone_action("toggle_active", {"zone_id": zid, "value": False})
        entry = p._session.files[key]
        assert entry.fit_zones[0]["active"] is False

    def test_clear_results(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        zid = ctrl.fit_zone_action("add", {})["zone_id"]
        entry = p._session.files[key]
        entry.fit_zones[0]["fit_result"] = {"e_fitted": [0.0]}
        entry.fit_result = entry.fit_zones[0]["fit_result"]
        ctrl.fit_zone_action("clear_results", {})
        assert entry.fit_zones[0]["fit_result"] is None
        assert entry.fit_result is None

    def test_unknown_verb_returns_error(self, parent_and_ctrl):
        _, ctrl, _ = parent_and_ctrl
        res = ctrl.fit_zone_action("nuke", {})
        assert res["ok"] is False
        assert "unknown_verb" in res["error"]


class TestPersistence:
    def test_session_roundtrip(self, tmp_path):
        sess = Session(folder=tmp_path)
        entry = sess.get_or_create("foo.h5")
        entry.fit_zones = [
            {
                "id": "abc12345",
                "label": "Z1",
                "color_idx": 0,
                "active": True,
                "fit_params": {"k_min": -0.5, "k_max": 0.5, "n_pairs": 2},
                "fit_result": None,
            }
        ]
        entry.active_zone_id = "abc12345"
        sess.save()
        sess2 = Session(folder=tmp_path)
        sess2.load(sess.json_path)
        e2 = sess2.files["foo.h5"]
        assert e2.fit_zones[0]["id"] == "abc12345"
        assert e2.active_zone_id == "abc12345"
        assert e2.fit_zones[0]["fit_params"]["n_pairs"] == 2

    def test_legacy_session_loads_with_empty_zones(self, tmp_path):
        # Simulate a pre-feature session JSON missing fit_zones / active_zone_id.
        sess = Session(folder=tmp_path)
        payload = {
            "version": 1,
            "folder": str(tmp_path),
            "work_func": 4.0,
            "files": {
                "bar.h5": {
                    "ef_offset": 0.05,
                    "fit_params": {"n_pairs": 1},
                    "fit_result": {"e_fitted": [0.0], "kF_minus": [[0.1]],
                                   "kF_plus": [[-0.1]], "gamma_corrige": [[0.05]]},
                }
            },
        }
        (tmp_path / ".arpes_session.json").write_text(json.dumps(payload))
        sess.load(tmp_path / ".arpes_session.json")
        e = sess.files["bar.h5"]
        assert e.fit_zones == []
        assert e.active_zone_id is None
        assert e.fit_result is not None


class TestAsymmetricWarning:
    def test_zone_left_of_gamma_warns(self):
        ctrl = FitZonesController(SimpleNamespace())
        zone = {"label": "Z1", "fit_params": {"k_min": -0.5, "k_max": -0.1}}
        assert ctrl.asymmetric_warning(zone, gamma_center=0.0) is not None

    def test_zone_right_of_gamma_warns(self):
        ctrl = FitZonesController(SimpleNamespace())
        zone = {"label": "Z2", "fit_params": {"k_min": 0.1, "k_max": 0.5}}
        assert ctrl.asymmetric_warning(zone, gamma_center=0.0) is not None

    def test_zone_crossing_gamma_silent(self):
        ctrl = FitZonesController(SimpleNamespace())
        zone = {"label": "Z3", "fit_params": {"k_min": -0.3, "k_max": 0.3}}
        assert ctrl.asymmetric_warning(zone, gamma_center=0.0) is None


class TestPalette:
    def test_palette_has_distinct_colors(self):
        assert len(set(ZONE_PALETTE)) == len(ZONE_PALETTE)
        assert len(ZONE_PALETTE) >= 10


class TestStoreResultSyncsFitParams:
    """HIGH-4: store_result must snapshot current FitParams into the zone."""

    def test_store_result_snapshots_fit_params(self, tmp_path):
        sess = Session(folder=tmp_path)
        key = "file.h5"
        sess.get_or_create(key)
        captured = FitParams(k_min=-0.42, k_max=0.42, n_pairs=2)

        class P:
            def __init__(self):
                self._session = sess
                self._current_path = tmp_path / key
                self._params = _StubParams(captured)

            def _status(self, m):
                pass

        ctrl = FitZonesController(P())
        zid = ctrl.fit_zone_action(
            "add", {"fit_params": FitParams(k_min=0.0, k_max=0.0, n_pairs=1)},
        )["zone_id"]
        new_fr = {"e_fitted": [0.0, -0.1], "kF_minus": [[0.05, 0.06]],
                  "kF_plus": [[-0.05, -0.06]]}
        ctrl.store_result(zid, new_fr)
        entry = sess.files[key]
        assert entry.fit_zones[0]["fit_result"] is new_fr
        # fit_params snapshot from current spinboxes (captured) overwrote the
        # original add-time params.
        assert entry.fit_zones[0]["fit_params"]["k_min"] == -0.42
        assert entry.fit_zones[0]["fit_params"]["n_pairs"] == 2


class TestSaveErrorSurface:
    """HIGH-3: save failures must reach the status bar instead of being swallowed."""

    def test_save_failure_calls_status(self, tmp_path):
        from types import SimpleNamespace

        class FailingSession:
            def __init__(self):
                self.folder = tmp_path
                self.files = {"f.h5": SimpleNamespace(
                    fit_zones=[], active_zone_id=None,
                )}

            def get_or_create(self, k):
                return self.files[k]

            def key_for_path(self, p):
                return "f.h5"

            def save(self):
                raise OSError("disk full")

        sess = FailingSession()
        messages: list[str] = []

        class P:
            def __init__(self):
                self._session = sess
                self._current_path = tmp_path / "f.h5"
                self._params = _StubParams()

            def _status(self, m):
                messages.append(m)

        ctrl = FitZonesController(P())
        ctrl._save()
        assert any("session save failed" in m for m in messages)
        assert any("disk full" in m for m in messages)


class TestUpdateActiveFromParams:
    """Auto-bind: editing the panel rewrites the active zone's fit_params."""

    def test_update_active_snapshots_current_params(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        ctrl.fit_zone_action("add", {"fit_params": FitParams(k_min=0.0, k_max=0.0)})
        # Simulate the user editing the panel after the zone was created.
        p._params._fp = FitParams(k_min=-0.7, k_max=0.7, n_pairs=2)
        res = ctrl.fit_zone_action("update_active_from_params", {})
        assert res["ok"]
        entry = p._session.files[key]
        assert entry.fit_zones[0]["fit_params"]["k_min"] == -0.7
        assert entry.fit_zones[0]["fit_params"]["n_pairs"] == 2
        # Legacy mirror updated; fit_result must stay untouched.
        assert entry.fit_params.k_min == -0.7
        assert entry.fit_zones[0]["fit_result"] is None

    def test_update_active_noop_without_zone(self, parent_and_ctrl):
        _, ctrl, _ = parent_and_ctrl
        res = ctrl.fit_zone_action("update_active_from_params", {})
        assert res["ok"] is False
        assert res["error"] == "no_active_zone"


class TestLabelNoCollision:
    """D6: removing then re-adding must not duplicate a "Z<n>" label."""

    def test_label_recycled_after_remove(self, parent_and_ctrl):
        p, ctrl, key = parent_and_ctrl
        z1 = ctrl.fit_zone_action("add", {})["zone_id"]   # Z1
        ctrl.fit_zone_action("add", {})                   # Z2
        ctrl.fit_zone_action("remove", {"zone_id": z1})   # drop Z1
        ctrl.fit_zone_action("add", {})                   # should reuse Z1
        entry = p._session.files[key]
        assert sorted(z["label"] for z in entry.fit_zones) == ["Z1", "Z2"]
