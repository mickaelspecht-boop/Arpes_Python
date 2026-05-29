"""Tests for the single-setter wrapper around entry.fit_result."""
from __future__ import annotations

from types import SimpleNamespace

from arpes.core.fit_result_store import clear_fit_result, set_fit_result


def _entry(active=None, zones=None):
    return SimpleNamespace(
        fit_result=None,
        fit_zones=list(zones or []),
        active_zone_id=active,
    )


class TestSetFitResult:
    def test_no_zones_writes_legacy_slot(self):
        e = _entry()
        set_fit_result(e, {"e_fitted": [0.0]})
        assert e.fit_result == {"e_fitted": [0.0]}

    def test_named_active_zone_updates_both(self):
        z = {"id": "a", "fit_result": None}
        e = _entry(active="a", zones=[z])
        set_fit_result(e, {"e": 1})
        assert e.fit_result == {"e": 1}
        assert z["fit_result"] == {"e": 1}

    def test_named_non_active_zone_updates_only_that_zone(self):
        z1 = {"id": "a", "fit_result": None}
        z2 = {"id": "b", "fit_result": None}
        e = _entry(active="a", zones=[z1, z2])
        set_fit_result(e, {"e": 2}, zone_id="b")
        assert z2["fit_result"] == {"e": 2}
        assert z1["fit_result"] is None
        assert e.fit_result is None  # legacy slot NOT touched

    def test_default_zone_id_uses_active(self):
        z = {"id": "a", "fit_result": None}
        e = _entry(active="a", zones=[z])
        set_fit_result(e, {"x": 9})
        assert z["fit_result"] == {"x": 9}
        assert e.fit_result == {"x": 9}

    def test_clear_resets_all(self):
        z1 = {"id": "a", "fit_result": {"e": 1}}
        z2 = {"id": "b", "fit_result": {"e": 2}}
        e = _entry(active="a", zones=[z1, z2])
        e.fit_result = {"e": 1}
        clear_fit_result(e)
        assert e.fit_result is None
        assert z1["fit_result"] is None
        assert z2["fit_result"] is None
