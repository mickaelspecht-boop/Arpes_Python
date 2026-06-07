"""BM ↔ FS pairing by metadata auto-discovery plus manual override.

A.2 of the BM↔FS plan (see BM_FS_ORGANIZATION_PLAN.md). Hybrid M4 model:
- Manual override through `entry.parent_fs_path` (absolute priority).
- Otherwise filter by folder + hv (±5%) + azi (±2°) + polarization.

Auto-discovery also iterates logbook records (candidate BMs not loaded) and
synthesizes a minimal FileEntry from each record so they can pass through the
same filters. See `build_pseudo_entries_from_logbook`.

Pure: no Qt, optional I/O only for `detect_scan_kind`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import math


@dataclass(frozen=True)
class PairingCriteria:
    """BM↔FS compatibility criteria for auto-discovery.

    Defaults are documented in BM_FS_ORGANIZATION_PLAN.md Q2.
    """
    same_folder: bool = True
    folder_depth: int = 0            # 0 = strict same folder (avoids cross-sample BNA_S1↔BNA_S2)
    hv_tolerance_rel: float = 0.05   # ±5 %
    azi_tolerance_deg: float = 2.0
    require_polarization: bool = True
    require_sample: bool = False     # opt-in via formula / mp_id


@dataclass(frozen=True)
class PairingMatch:
    """Pairing result: path, compatible entry, reason + distance."""
    path: str
    entry: object                # FileEntry - typed as Any to avoid circular import
    reason: str                  # "manual" | "auto"
    distance: float              # 0.0 = perfect; +∞ = incompatible (filtered before)


def _same_folder(path_a: str, path_b: str, *, depth: int) -> bool:
    """True if both paths share their parent folder within `depth`.

    depth=0: exact same direct folder.
    depth=1: same folder OR same grandparent (one subfolder apart).
    depth>=2: same folder OR up to N levels apart.
    """
    try:
        a = Path(path_a).resolve(strict=False)
        b = Path(path_b).resolve(strict=False)
    except Exception:
        return str(path_a) == str(path_b)
    pa = a.parent
    pb = b.parent
    if pa == pb:
        return True
    if depth <= 0:
        return False
    # Walk up to `depth` levels on both sides and compare.
    for da in range(depth + 1):
        for db in range(depth + 1):
            anc_a = pa
            anc_b = pb
            for _ in range(da):
                anc_a = anc_a.parent
            for _ in range(db):
                anc_b = anc_b.parent
            if anc_a == anc_b and anc_a != anc_a.parent:  # pas racine "/"
                return True
    return False


def _relative_hv_diff(hv_a, hv_b) -> float:
    """|Δhv|/max(hv_a, hv_b). Return +inf if either value is missing."""
    try:
        ha = float(hv_a)
        hb = float(hv_b)
    except (TypeError, ValueError):
        return math.inf
    if ha <= 0 or hb <= 0 or not math.isfinite(ha) or not math.isfinite(hb):
        return math.inf
    return abs(ha - hb) / max(ha, hb)


def _azi_diff_deg(azi_a, azi_b) -> float:
    """|Δazi| in deg. Return 0.0 if both azi values are None (compatible).
    Return +inf if only one is None."""
    if azi_a is None and azi_b is None:
        return 0.0
    if azi_a is None or azi_b is None:
        return math.inf
    try:
        diff = (float(azi_a) - float(azi_b) + 180.0) % 360.0 - 180.0
        return abs(diff)
    except (TypeError, ValueError):
        return math.inf


def _polarization_compatible(pol_a, pol_b) -> bool:
    """Compare polarizations (LH/LV/...). Tolerate an empty value on either side."""
    a = str(pol_a or "").strip().lower()
    b = str(pol_b or "").strip().lower()
    if not a or not b:
        return True
    return a == b


def _sample_compatible(entry_a, entry_b) -> bool:
    """Compare formula / mp_id for two entries. Tolerate an empty value on either side."""
    for attr in ("formula", "mp_id"):
        va = str(getattr(entry_a.meta, attr, "") or "").strip().lower()
        vb = str(getattr(entry_b.meta, attr, "") or "").strip().lower()
        if va and vb and va != vb:
            return False
    return True


def _is_compatible_auto(
    bm_entry, bm_path: str,
    fs_entry, fs_path: str,
    criteria: PairingCriteria,
) -> tuple[bool, float]:
    """Filter auto-discovery and compute a distance score for sorting.

    Returns:
        (compatible, distance). distance ∈ [0, +∞), 0 = perfect.
    """
    if criteria.same_folder and not _same_folder(
        bm_path, fs_path, depth=criteria.folder_depth
    ):
        return False, math.inf

    hv_rel = _relative_hv_diff(bm_entry.meta.hv, fs_entry.meta.hv)
    if hv_rel > criteria.hv_tolerance_rel:
        return False, math.inf

    azi_diff = _azi_diff_deg(bm_entry.meta.azi, fs_entry.meta.azi)
    if azi_diff > criteria.azi_tolerance_deg:
        return False, math.inf

    if criteria.require_polarization and not _polarization_compatible(
        bm_entry.meta.polarization, fs_entry.meta.polarization
    ):
        return False, math.inf

    if criteria.require_sample and not _sample_compatible(bm_entry, fs_entry):
        return False, math.inf

    # Score: normalized root-mean-square of differences (0 = perfect).
    # Tolerances are used as denominators (avoids div/0 via max(1e-6, ...)).
    azi_norm = azi_diff / max(criteria.azi_tolerance_deg, 1e-6)
    hv_norm = hv_rel / max(criteria.hv_tolerance_rel, 1e-6)
    distance = math.sqrt(azi_norm ** 2 + hv_norm ** 2)
    return True, distance


def find_bms_for_fs(
    fs_entry,
    fs_path: str,
    all_files: dict,
    criteria: PairingCriteria | None = None,
) -> list[PairingMatch]:
    """Return BMs compatible with a given FS.

    Order: manual overrides (entry.parent_fs_path == fs_path) first, then
    auto-discovered matches sorted by increasing distance.

    Args:
        fs_entry: FS FileEntry.
        fs_path: FS key/path in session.files.
        all_files: dict {path → FileEntry} (usually session.files).
        criteria: PairingCriteria (defaults if None).

    Returns:
        list[PairingMatch].
    """
    criteria = criteria or PairingCriteria()
    if getattr(fs_entry.meta, "scan_kind", None) not in {"FS", None, ""}:
        return []

    manual: list[PairingMatch] = []
    auto: list[PairingMatch] = []
    for bm_path, bm_entry in all_files.items():
        if bm_path == fs_path:
            continue
        if getattr(bm_entry.meta, "scan_kind", "") != "BM":
            continue
        # Manual override: absolute priority.
        if getattr(bm_entry, "parent_fs_path", None) == fs_path:
            manual.append(PairingMatch(bm_path, bm_entry, "manual", 0.0))
            continue
        compatible, distance = _is_compatible_auto(
            bm_entry, bm_path, fs_entry, fs_path, criteria,
        )
        if compatible:
            auto.append(PairingMatch(bm_path, bm_entry, "auto", distance))

    manual.sort(key=lambda m: m.path)
    auto.sort(key=lambda m: (m.distance, m.path))
    return manual + auto


def find_fs_for_bm(
    bm_entry,
    bm_path: str,
    all_files: dict,
    criteria: PairingCriteria | None = None,
) -> list[PairingMatch]:
    """Symmetric helper: return FS entries compatible with a given BM.

    Manual override: if `bm_entry.parent_fs_path` is set and points to an FS in
    `all_files`, it appears first with
    reason="manual".
    """
    criteria = criteria or PairingCriteria()
    if getattr(bm_entry.meta, "scan_kind", None) not in {"BM", None, ""}:
        return []

    manual: list[PairingMatch] = []
    auto: list[PairingMatch] = []
    pinned = getattr(bm_entry, "parent_fs_path", None)
    for fs_path, fs_entry in all_files.items():
        if fs_path == bm_path:
            continue
        if getattr(fs_entry.meta, "scan_kind", "") != "FS":
            continue
        if pinned and fs_path == pinned:
            manual.append(PairingMatch(fs_path, fs_entry, "manual", 0.0))
            continue
        compatible, distance = _is_compatible_auto(
            bm_entry, bm_path, fs_entry, fs_path, criteria,
        )
        if compatible:
            auto.append(PairingMatch(fs_path, fs_entry, "auto", distance))

    manual.sort(key=lambda m: m.path)
    auto.sort(key=lambda m: (m.distance, m.path))
    return manual + auto


def group_files_by_fs(
    all_files: dict,
    criteria: PairingCriteria | None = None,
) -> tuple[list[tuple[str, object, list[PairingMatch]]], list[tuple[str, object]]]:
    """Build a tree view (FS → linked BMs) plus an orphan list.

    For the O3 file browser (Phase A.5).

    Returns:
        (tree, orphans), where:
        - tree: [(fs_path, fs_entry, [PairingMatch BM]), ...] sorted by fs_path
        - orphans: [(path, entry), ...] = unattached BMs + other scan_kind values
    """
    criteria = criteria or PairingCriteria()
    tree: list[tuple[str, object, list[PairingMatch]]] = []
    attached_bms: set[str] = set()
    fs_items = [
        (p, e) for p, e in all_files.items()
        if getattr(e.meta, "scan_kind", "") == "FS"
    ]
    fs_items.sort(key=lambda kv: kv[0])
    for fs_path, fs_entry in fs_items:
        bms = find_bms_for_fs(fs_entry, fs_path, all_files, criteria)
        tree.append((fs_path, fs_entry, bms))
        attached_bms.update(m.path for m in bms)
    orphans: list[tuple[str, object]] = []
    for path, entry in all_files.items():
        if getattr(entry.meta, "scan_kind", "") == "FS":
            continue
        if getattr(entry.meta, "scan_kind", "") == "BM" and path in attached_bms:
            continue
        orphans.append((path, entry))
    orphans.sort(key=lambda kv: kv[0])
    return tree, orphans


def _iter_data_candidates(folder: Path, *, max_depth: int = 1) -> Iterable[Path]:
    """Yield data-like files/folders under `folder`.

    For CLS, BM = an extensionless file named `BMxx`; FS = a folder named
    `FSxx` containing _Cycle_Step files. Other formats (.pxt, .ibw, .zip
    Solaris) are treated as files.

    Skip: hidden files, `*_param.txt` (CLS sidecars), `.arpes_*` folders.
    """
    if not folder or not folder.exists() or not folder.is_dir():
        return
    SKIP_DIRS = {".arpes_cache", ".arpes_theory_cache", ".git"}
    DATA_SUFFIXES = {".pxt", ".ibw", ".zip", ".h5", ".hdf5"}
    for entry in folder.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_file():
            if entry.name.endswith("_param.txt"):
                continue
            # Extensionless file OR data extension -> candidate.
            if not entry.suffix or entry.suffix.lower() in DATA_SUFFIXES:
                yield entry
            continue
        if entry.is_dir():
            # CLS-FS or similar folder -> direct candidate.
            yield entry
            if max_depth > 1:
                yield from _iter_data_candidates(entry, max_depth=max_depth - 1)


def build_pseudo_entries_from_logbook(
    session,
    *,
    scan_kind_resolver: Callable[[str], str] | None = None,
) -> dict:
    """Build minimal FileEntry objects for session-folder files that appear in
    the logbook but are not loaded yet.

    Strategy: scan `session.folder` for data-like paths, then call
    `LogbookManager.values_for_path` on each path (it handles fuzzy BM1↔BM01
    matching through internal heuristics). If values are present and the path is
    not loaded, create a pseudo entry.

    This fills BM↔FS pairing when the user loaded only the reference FS but the
    candidate BMs exist in the folder (see BM_FS_ORGANIZATION_PLAN.md, user
    feedback 2026-06-01).

    Returns:
        dict {key → pseudo FileEntry}.
    """
    from arpes.core.session import FileEntry, FileMeta
    from arpes.io.logbook import LogbookManager
    if scan_kind_resolver is None:
        try:
            from arpes.io.loaders import detect_scan_kind as _detect
            scan_kind_resolver = lambda p: _detect(p)
        except Exception:
            scan_kind_resolver = lambda _p: "unknown"

    records = getattr(session, "logbook_records", None) or []
    mapping = getattr(session, "logbook_mapping", None) or {}
    scoped = {}
    raw_scoped = getattr(session, "scoped_logbooks", None) or {}
    for rel, meta in raw_scoped.items():
        if isinstance(meta, dict) and meta.get("mapping"):
            scoped[str(rel)] = meta["mapping"]
    folder = getattr(session, "folder", None)
    if not folder:
        return {}
    manager = LogbookManager(records, mapping, folder, scoped)
    loaded_keys = set((getattr(session, "files", {}) or {}).keys())

    out: dict = {}
    for path in _iter_data_candidates(Path(folder), max_depth=1):
        abs_path = str(path)
        try:
            key = session.key_for_path(abs_path)
        except Exception:
            key = abs_path
        if key in loaded_keys or key in out:
            continue
        try:
            values = manager.values_for_path(abs_path)
        except Exception:
            continue
        if not values.has_any():
            continue
        try:
            sk = str(scan_kind_resolver(abs_path) or "unknown")
        except Exception:
            sk = "unknown"
        meta = FileMeta(
            hv=float(values.hv) if values.hv is not None else 0.0,
            azi=values.azi,
            polar=values.polar,
            tilt=values.tilt,
            polarization=str(values.polarization or ""),
            formula=str(values.formula or ""),
            mp_id=str(values.mp_id or ""),
            scan_kind=sk,
        )
        out[key] = FileEntry(meta=meta)
    return out
