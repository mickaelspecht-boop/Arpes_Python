"""User-level app settings via QSettings.

Per-user, OS-native store living outside the (public) repository and outside the
shareable ``.arpes_session.json``. Used for secrets such as the Materials Project
API key: a user enters their own key in the UI; it is persisted here only — never
written to the repo or to a session file that travels with data.

A developer's local key can stay in the ``MP_API_KEY`` environment variable:
``resolve_mp_api_key`` returns None when nothing is stored here, letting
``arpes.theory.materials_project`` fall back to the env var.
"""
from __future__ import annotations

from PyQt6.QtCore import QSettings

_ORG = "ArpesExplorer"
_APP = "ArpesExplorer"
_MP_KEY = "materials_project/api_key"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_mp_api_key() -> str:
    """Stored Materials Project API key, or '' if none."""
    return str(_settings().value(_MP_KEY, "") or "")


def set_mp_api_key(key: str) -> None:
    """Persist (or clear, when empty) the MP API key for this user."""
    s = _settings()
    key = (key or "").strip()
    if key:
        s.setValue(_MP_KEY, key)
    else:
        s.remove(_MP_KEY)


def resolve_mp_api_key() -> str | None:
    """QSettings key if set, else None (so materials_project can fall back to the
    MP_API_KEY env var — e.g. a developer's local key)."""
    return get_mp_api_key() or None
