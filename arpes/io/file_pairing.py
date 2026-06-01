"""Pairing BM ↔ FS — auto-discovery par métadonnées + override manuel.

A.2 du plan BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md). Modèle M4 hybride :
- Override manuel via `entry.parent_fs_path` (priorité absolue).
- Sinon filtrage par dossier + hv (±5%) + azi (±2°) + polarization.

L'auto-discovery itère aussi les records du logbook (BMs candidates non
chargées) — synthétise un FileEntry minimal depuis chaque record pour
les passer aux mêmes filtres. Cf `build_pseudo_entries_from_logbook`.

Pure : sans Qt, I/O optionnel uniquement pour `detect_scan_kind`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import math


@dataclass(frozen=True)
class PairingCriteria:
    """Critères de compatibilité BM↔FS pour l'auto-discovery.

    Defaults documentés dans BM_FS_ORGANIZATION_PLAN.md Q2.
    """
    same_folder: bool = True
    folder_depth: int = 1            # 0 = même dossier strict, 1 = même parent direct
    hv_tolerance_rel: float = 0.05   # ±5 %
    azi_tolerance_deg: float = 2.0
    require_polarization: bool = True
    require_sample: bool = False     # opt-in via formula / mp_id


@dataclass(frozen=True)
class PairingMatch:
    """Résultat d'un pairing : path, entry compatible, raison + distance."""
    path: str
    entry: object                # FileEntry — typé Any pour éviter import circulaire
    reason: str                  # "manual" | "auto"
    distance: float              # 0.0 = parfait ; +∞ = incompatible (filtré avant)


def _same_folder(path_a: str, path_b: str, *, depth: int) -> bool:
    """True si les deux paths partagent leur dossier parent à `depth` près.

    depth=0 : même dossier direct exact.
    depth=1 : même dossier OU même grand-parent (un sous-dossier d'écart).
    depth>=2 : même dossier OU jusqu'à N niveaux d'écart.
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
    # Remonte jusqu'à `depth` niveaux des deux côtés et compare
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
    """|Δhv|/max(hv_a, hv_b). Retourne +inf si l'une des deux valeurs absente."""
    try:
        ha = float(hv_a)
        hb = float(hv_b)
    except (TypeError, ValueError):
        return math.inf
    if ha <= 0 or hb <= 0 or not math.isfinite(ha) or not math.isfinite(hb):
        return math.inf
    return abs(ha - hb) / max(ha, hb)


def _azi_diff_deg(azi_a, azi_b) -> float:
    """|Δazi| en deg. Retourne 0.0 si les deux azi sont None (compatible).
    Retourne +inf si une seule est None."""
    if azi_a is None and azi_b is None:
        return 0.0
    if azi_a is None or azi_b is None:
        return math.inf
    try:
        return abs(float(azi_a) - float(azi_b))
    except (TypeError, ValueError):
        return math.inf


def _polarization_compatible(pol_a, pol_b) -> bool:
    """Compare polarizations (LH/LV/...). Tolère vide d'un côté."""
    a = str(pol_a or "").strip().lower()
    b = str(pol_b or "").strip().lower()
    if not a or not b:
        return True
    return a == b


def _sample_compatible(entry_a, entry_b) -> bool:
    """Compare formula / mp_id de deux entries. Tolère vide d'un côté."""
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
    """Filtre auto-discovery + score distance pour tri.

    Returns:
        (compatible, distance). distance ∈ [0, +∞), 0 = parfait.
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

    # Score : moyenne quadratique normalisée des écarts (0 = parfait).
    # Tolérances utilisées comme dénominateurs (évite div/0 par max(1e-6, …)).
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
    """Retourne les BMs compatibles avec une FS donnée.

    Order : manual overrides (entry.parent_fs_path == fs_path) en premier,
    puis auto-discovered triées par distance croissante.

    Args:
        fs_entry: FileEntry de la FS.
        fs_path: clé/path de la FS dans session.files.
        all_files: dict {path → FileEntry} (souvent session.files).
        criteria: PairingCriteria (defaults si None).

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
        # Override manuel : priorité absolue
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
    """Symétrique : retourne les FS compatibles avec une BM donnée.

    Override manuel : si `bm_entry.parent_fs_path` est défini et qu'il
    pointe vers une FS de `all_files`, elle apparaît en premier avec
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
    """Construit une vue arborescente (FS → BMs liées) + liste orphelins.

    Pour le file browser O3 (Phase A.5).

    Returns:
        (tree, orphans) où :
        - tree : [(fs_path, fs_entry, [PairingMatch BM]), ...] trié par fs_path
        - orphans : [(path, entry), ...] = BMs non rattachées + autres scan_kind
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
    """Yield fichiers/dossiers data-like sous `folder`.

    Pour chaque CLS BM = un fichier sans extension nommé `BMxx`, pour CLS FS
    = un dossier nommé `FSxx` contenant des fichiers _Cycle_Step. Autres
    formats (.pxt, .ibw, .zip Solaris) traités comme fichiers.

    Skip : fichiers cachés, `*_param.txt` (sidecars CLS), dossiers `.arpes_*`.
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
            # Fichier sans extension OU avec extension data → candidat
            if not entry.suffix or entry.suffix.lower() in DATA_SUFFIXES:
                yield entry
            continue
        if entry.is_dir():
            # Dossier CLS-FS ou similaire → candidat direct
            yield entry
            if max_depth > 1:
                yield from _iter_data_candidates(entry, max_depth=max_depth - 1)


def build_pseudo_entries_from_logbook(
    session,
    *,
    scan_kind_resolver: Callable[[str], str] | None = None,
) -> dict:
    """Construit des FileEntry minimaux pour les fichiers du dossier session
    qui apparaissent dans le logbook mais ne sont pas encore chargés.

    Stratégie : scanne `session.folder` pour les paths data-like, puis pour
    chaque path appelle `LogbookManager.values_for_path` (qui gère le
    matching fuzzy BM1↔BM01 via les heuristiques internes). Si values
    présentes et path non chargé → pseudo entry.

    Permet de combler le pairing BM↔FS quand l'utilisateur n'a chargé que
    la FS de référence mais que les BMs candidates existent dans le dossier
    (cf BM_FS_ORGANIZATION_PLAN.md, feedback user 2026-06-01).

    Returns:
        dict {key → FileEntry pseudo}.
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
