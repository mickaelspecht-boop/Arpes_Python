"""Append-only provenance journal for one ARPES signal (FileEntry).

Single-setter for ``FileEntry.processing_history`` — the same discipline as
``fit_result_store``. Every data transformation or fit operation is logged here
at the moment it is applied, so the journal is a faithful, timestamped audit
trail of "what we did, and when".

The journal is *never* read back as the source of truth for the current
parameters (those live in the typed ``FileEntry`` fields and are summarised by
``experience_log.build_experience_log``). It is purely an append-only record,
which is exactly why it cannot drift away from the real state.

Pure module: no PyQt, JSON-safe, importable headless and from tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Category tags — stable identifiers used by the renderer and the dock filter.
# Keep these short and lowercase; the UI uppercases them for display.
CAT_LOAD = "load"        # file loaded / re-loaded
CAT_ENERGY = "energy"    # EF offset, EF calibration
CAT_GAMMA = "gamma"      # Γ (k=0) recentering
CAT_NORM = "norm"        # EDC / above-EF normalization, view mode
CAT_GRID = "grid"        # detector-grid Fourier correction
CAT_DISTORT = "distort"  # BM trapezoid/parabola distortion correction
CAT_FS = "fs"            # Fermi-surface center / rotation
CAT_KZ = "kz"            # kz / inner-potential settings
CAT_FIT = "fit"          # MDC peak-pair fit run
CAT_ZONE = "zone"        # multi-zone fit add/remove/activate
CAT_KF = "kf"            # manual kF drag / snap / mark-bad
CAT_BAND = "band"        # TB fit / kink / gap analysis
CAT_THEORY = "theory"    # DFT theory overlay
CAT_EDIT = "edit"        # undo / redo / reset

# All categories, in a stable display order (used to build filter menus).
CATEGORIES = (
    CAT_LOAD, CAT_ENERGY, CAT_GAMMA, CAT_NORM, CAT_GRID, CAT_DISTORT,
    CAT_FS, CAT_KZ, CAT_FIT, CAT_ZONE, CAT_KF, CAT_BAND, CAT_THEORY, CAT_EDIT,
)

# Hard cap so a long working session cannot grow the JSON without bound.
# When exceeded, the oldest events are dropped (provenance stays bounded but the
# recent, actionable history is always preserved).
_MAX_EVENTS = 5000


def _now_iso() -> str:
    """UTC timestamp, second precision, trailing 'Z' (sortable, JSON-safe)."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _history(entry: Any) -> list:
    """Return entry.processing_history, creating it if absent/corrupt."""
    h = getattr(entry, "processing_history", None)
    if not isinstance(h, list):
        h = []
        try:
            entry.processing_history = h
        except Exception:
            pass
    return h


def _clean_params(params: dict | None) -> dict:
    """Coerce params to a compact, JSON-safe dict (no large arrays/objects).

    Floats are rounded for readability; long sequences are summarised by their
    length so a stray numpy array can never bloat the session file.
    """
    if not params:
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        name = str(key)
        if value is None:
            continue
        if isinstance(value, bool):
            out[name] = value
        elif isinstance(value, int):
            out[name] = value
        elif isinstance(value, float):
            out[name] = round(value, 6)
        elif isinstance(value, str):
            out[name] = value
        elif isinstance(value, (list, tuple)):
            seq = list(value)
            if len(seq) <= 8 and all(
                isinstance(x, (int, float, bool, str)) for x in seq
            ):
                out[name] = [
                    round(x, 6) if isinstance(x, float) and not isinstance(x, bool)
                    else x
                    for x in seq
                ]
            else:
                out[name] = f"[{len(seq)} values]"
        else:
            out[name] = str(value)
    return out


def log_event(
    entry: Any,
    category: str,
    action: str,
    *,
    summary: str = "",
    params: dict | None = None,
    coalesce: bool = False,
    ts: str | None = None,
) -> dict | None:
    """Append one provenance event to ``entry.processing_history``.

    Parameters
    ----------
    category : one of the ``CAT_*`` constants.
    action   : short verb phrase, e.g. ``"MDC fit"`` or ``"EDCnorm on"``.
    summary  : human-readable one-liner shown as the event detail.
    params   : compact key/value metadata (rounded, JSON-safe).
    coalesce : when True, replace the immediately-preceding event if it shares
               the same ``(category, action)``. Used for continuous adjustments
               (e.g. dragging a kF handle) so only the final value is kept,
               instead of one event per intermediate step.

    Returns the stored event dict (or ``None`` if ``entry`` is falsy).
    """
    if entry is None:
        return None
    hist = _history(entry)
    event = {
        "ts": ts or _now_iso(),
        "category": str(category),
        "action": str(action),
        "summary": str(summary or ""),
        "params": _clean_params(params),
    }
    if coalesce and hist:
        last = hist[-1]
        if (
            last.get("category") == event["category"]
            and last.get("action") == event["action"]
        ):
            hist[-1] = event
            return event
    hist.append(event)
    if len(hist) > _MAX_EVENTS:
        del hist[: len(hist) - _MAX_EVENTS]
    return event


def clear_history(entry: Any) -> None:
    """Wipe the journal for one entry (user-initiated 'Clear log')."""
    try:
        entry.processing_history = []
    except Exception:
        pass


def event_count(entry: Any) -> int:
    """Number of recorded events for one entry (cheap, for live refresh polls)."""
    h = getattr(entry, "processing_history", None)
    return len(h) if isinstance(h, list) else 0


def log_action(
    window: Any,
    category: str,
    action: str,
    *,
    entry: Any = None,
    summary: str = "",
    params: dict | None = None,
    coalesce: bool = False,
) -> None:
    """UI-facing safe logger used by every instrumentation site.

    Resolves ``window._experience_log_ctrl`` and forwards the event (the
    controller also refreshes the live dock). Lives in this pure module so any
    controller that already imports ``processing_history`` can log without
    importing a UI module, and so it no-ops cleanly on headless windows / test
    doubles that lack the controller. If no controller exists but a concrete
    ``entry`` was supplied, the event is still recorded directly so provenance
    is never silently lost in headless runs.
    """
    try:
        ctrl = getattr(window, "_experience_log_ctrl", None)
        if ctrl is not None:
            ctrl.log(
                category, action, entry=entry, summary=summary,
                params=params, coalesce=coalesce,
            )
            return
        if entry is not None:
            log_event(
                entry, category, action,
                summary=summary, params=params, coalesce=coalesce,
            )
    except Exception:
        pass
