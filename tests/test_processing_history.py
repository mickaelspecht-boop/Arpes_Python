"""Tests for the append-only processing-history provenance journal."""
from __future__ import annotations

import numpy as np

from arpes.core import processing_history as ph
from arpes.core.experience_log import (
    build_full_report,
    build_timeline,
)
from arpes.core.session import FileEntry, Session


def test_log_event_appends_with_fields():
    e = FileEntry()
    ev = ph.log_event(e, ph.CAT_FIT, "MDC fit", summary="41 slices", params={"pairs": 2})
    assert ev is not None
    assert len(e.processing_history) == 1
    assert ev["category"] == "fit"
    assert ev["action"] == "MDC fit"
    assert ev["summary"] == "41 slices"
    assert ev["params"] == {"pairs": 2}
    assert ev["ts"].endswith("Z")


def test_coalesce_replaces_last_same_action():
    e = FileEntry()
    ph.log_event(e, ph.CAT_KF, "drag kF", params={"slice": 1}, coalesce=True)
    ph.log_event(e, ph.CAT_KF, "drag kF", params={"slice": 5}, coalesce=True)
    ph.log_event(e, ph.CAT_KF, "drag kF", params={"slice": 9}, coalesce=True)
    assert len(e.processing_history) == 1
    assert e.processing_history[-1]["params"] == {"slice": 9}


def test_coalesce_does_not_merge_across_different_actions():
    e = FileEntry()
    ph.log_event(e, ph.CAT_KF, "drag kF", coalesce=True)
    ph.log_event(e, ph.CAT_NORM, "EDCnorm on", coalesce=True)
    ph.log_event(e, ph.CAT_KF, "drag kF", coalesce=True)
    assert len(e.processing_history) == 3


def test_clean_params_rounds_floats_and_summarizes_arrays():
    e = FileEntry()
    ev = ph.log_event(
        e, ph.CAT_DISTORT, "trapezoid",
        params={
            "slope": 0.123456789,
            "flag": True,
            "small": [1.111111, 2.0],
            "big": list(np.arange(100, dtype=float)),
            "skip": None,
        },
    )
    assert ev["params"]["slope"] == round(0.123456789, 6)
    assert ev["params"]["flag"] is True
    assert ev["params"]["small"] == [round(1.111111, 6), 2.0]
    assert ev["params"]["big"] == "[100 values]"
    assert "skip" not in ev["params"]


def test_cap_drops_oldest():
    e = FileEntry()
    for i in range(ph._MAX_EVENTS + 25):
        ph.log_event(e, ph.CAT_EDIT, "tick", params={"i": i})
    assert len(e.processing_history) == ph._MAX_EVENTS
    # Oldest dropped, newest kept.
    assert e.processing_history[-1]["params"]["i"] == ph._MAX_EVENTS + 24


def test_clear_history():
    e = FileEntry()
    ph.log_event(e, ph.CAT_LOAD, "loaded")
    ph.clear_history(e)
    assert e.processing_history == []
    assert ph.event_count(e) == 0


def test_log_event_none_entry_is_noop():
    assert ph.log_event(None, ph.CAT_FIT, "x") is None


def test_build_timeline_empty():
    e = FileEntry()
    md = build_timeline(e, name="sig")
    assert "No recorded operations yet" in md


def test_build_timeline_lists_events_chronologically():
    e = FileEntry()
    ph.log_event(e, ph.CAT_LOAD, "loaded", summary="bessy", ts="2026-01-01T10:00:00Z")
    ph.log_event(e, ph.CAT_ENERGY, "EF offset", summary="+0.012 eV", ts="2026-01-01T10:01:00Z")
    md = build_timeline(e, name="sig")
    assert "LOAD" in md and "ENERGY" in md
    assert "bessy" in md and "+0.012 eV" in md
    # loaded appears before EF offset
    assert md.index("loaded") < md.index("EF offset")


def test_build_timeline_limit():
    e = FileEntry()
    for i in range(10):
        ph.log_event(e, ph.CAT_EDIT, "tick", summary=f"n{i}")
    md = build_timeline(e, name="sig", limit=3)
    assert "Showing last 3 of 10" in md
    assert "n9" in md and "n0" not in md


def test_build_full_report_has_timeline_and_state():
    e = FileEntry()
    ph.log_event(e, ph.CAT_FIT, "MDC fit", summary="ok")
    rep = build_full_report(e, name="sig")
    assert "Processing timeline" in rep
    assert "Processing log" in rep  # state snapshot header


def test_history_survives_session_save_load(tmp_path):
    sess = Session(folder=tmp_path)
    entry = sess.get_or_create("sub/sig.h5")
    ph.log_event(entry, ph.CAT_LOAD, "loaded", summary="bessy")
    ph.log_event(entry, ph.CAT_FIT, "MDC fit", summary="41 slices", params={"pairs": 2})
    sess.save()

    reloaded = Session(folder=tmp_path)
    reloaded.load(sess.json_path)
    e2 = reloaded.files["sub/sig.h5"]
    assert len(e2.processing_history) == 2
    assert e2.processing_history[0]["action"] == "loaded"
    assert e2.processing_history[1]["params"] == {"pairs": 2}


def test_v3_payload_without_history_migrates_to_empty_list():
    # Simulate an older (v3) session payload that predates processing_history.
    payload = {
        "version": 3,
        "files": {"sig.h5": {"ef_offset": 0.0, "view_mode": "Raw"}},
    }
    sess = Session()
    sess.load_from_payload(payload)
    entry = sess.files["sig.h5"]
    assert entry.processing_history == []


def test_version_is_at_least_4():
    assert Session.VERSION >= 4
