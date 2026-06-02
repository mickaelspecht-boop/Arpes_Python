"""Path matching helpers for ARPES logbooks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _path_match_tokens(path: str | Path, session_folder: Path | None) -> list[str]:
    p = Path(path)
    tokens = [p.name, p.stem]
    if session_folder is not None:
        try:
            rel = p.resolve().relative_to(session_folder.resolve())
            tokens.extend([str(rel), rel.name, rel.stem])
        except Exception:
            pass
    return sorted({t for t in tokens if t}, key=len, reverse=True)


def _extract_measurement_numbers(text: str) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for pat in (r"(\d{3,4})w", r"_(\d{1,4})(?:\D|$)"):
        for m in re.finditer(pat, text):
            try:
                out.add(int(m.group(1)))
            except ValueError:
                pass
    return out


def _path_measurement_numbers(path: str | Path) -> set[int]:
    p = Path(path)
    out: set[int] = set()
    for text in (p.name, p.stem):
        out |= _extract_measurement_numbers(text)
    return out


def _record_in_subfolder_scope(record: Any, path: str | Path, session_folder: Path | None) -> bool:
    """Check `_subfolder_rel` scoping when a logbook is tied to a subfolder."""
    if not isinstance(record, dict):
        return True
    subfolder = str(record.get("_subfolder_rel") or "").strip()
    if not subfolder:
        return True
    if session_folder is None:
        return True
    try:
        p_resolved = Path(path).resolve()
        scope_resolved = (Path(session_folder) / subfolder).resolve()
    except Exception:
        return True
    try:
        p_resolved.relative_to(scope_resolved)
        return True
    except ValueError:
        return False


_ALNUM_KEY_RE = re.compile(r"^\s*([A-Za-z]+)\s*0*(\d+)\s*$")


def _alnum_label_key(text: Any) -> tuple[str, int] | None:
    """('BM3' | 'BM03' | 'bm 3') -> ('bm', 3). Sinon None."""
    m = _ALNUM_KEY_RE.match(str(text or ""))
    if not m:
        return None
    try:
        return m.group(1).lower(), int(m.group(2))
    except ValueError:
        return None


def _record_matches_path(
    record_value: Any,
    path: str | Path,
    session_folder: Path | None,
    *,
    cell_text,
    cell_float,
) -> bool:
    value = cell_text(record_value)
    if not value:
        return False
    path_nums = _path_measurement_numbers(path)
    num = cell_float(record_value)
    if num is not None and abs(num - round(num)) < 1e-9:
        if int(round(num)) in path_nums:
            return True
    cell_nums = _extract_measurement_numbers(value)
    if cell_nums and path_nums and (cell_nums & path_nums):
        return True
    value_norm = value.lower()
    value_key = _alnum_label_key(value)
    for token in _path_match_tokens(path, session_folder):
        token_norm = token.lower()
        if value_norm == token_norm:
            return True
        if value_key is not None and _alnum_label_key(token) == value_key:
            return True
        pat = r"(?<![A-Za-z0-9])" + re.escape(token_norm) + r"(?![A-Za-z0-9])"
        if re.search(pat, value_norm):
            return True
    return False
