"""Liste persistante des sessions ouvertes/sauvegardées récemment.

Stocke `~/.arpes/recent_sessions.json` (max ``MAX_RECENT`` entrées). Les
entrées pointant vers un fichier inexistant sont filtrées au listing mais
conservées tant que l'utilisateur ne les retire pas explicitement.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

MAX_RECENT = 10


def _registry_path() -> Path:
    base = Path.home() / ".arpes"
    base.mkdir(parents=True, exist_ok=True)
    return base / "recent_sessions.json"


def _read_all() -> list[dict[str, Any]]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("path"):
            out.append(item)
    return out


def _write_all(items: list[dict[str, Any]]) -> None:
    _registry_path().write_text(json.dumps(items[:MAX_RECENT], indent=2))


def add_recent(session_path: Path | str, name: str = "", folder_hint: str = "") -> None:
    """Promote ``session_path`` to most-recent and persist."""
    abs_path = str(Path(session_path).expanduser().resolve())
    items = [it for it in _read_all() if it.get("path") != abs_path]
    items.insert(0, {
        "path": abs_path,
        "name": name or Path(abs_path).stem,
        "folder_hint": folder_hint,
        "last_used": datetime.now().isoformat(timespec="seconds"),
    })
    _write_all(items)


def list_recent(*, only_existing: bool = True) -> list[dict[str, Any]]:
    items = _read_all()
    if not only_existing:
        return items
    return [it for it in items if Path(it["path"]).exists()]


def remove_recent(session_path: Path | str) -> None:
    abs_path = str(Path(session_path).expanduser().resolve())
    _write_all([it for it in _read_all() if it.get("path") != abs_path])
