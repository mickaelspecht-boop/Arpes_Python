"""Découverte et chargement de séries hν pour cartes kz."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from arpes.io.logbook import LogbookManager
from arpes.physics.kz import KzScanInput, scan_from_legacy_dict

try:
    from arpes.io.loaders import detect_format, load_arpes_file
except Exception:  # pragma: no cover - environnement sans loaders complets
    detect_format = None
    load_arpes_file = None


LoadFileFunc = Callable[..., dict | None]


_KZ_EXTS = {".ibw", ".pxt", ".txt"}
_REJECT_NAME_BITS = (
    "[fs]",
    "_fs",
    " fs",
    "fixed cut",
    "align",
    "duplicate",
    "copy",
)


def _valid_hv(value: Any) -> float | None:
    try:
        hv = float(value)
    except (TypeError, ValueError):
        return None
    return hv if np.isfinite(hv) and hv > 0 else None


@dataclass(frozen=True)
class KzDataset:
    folder: Path
    scans: list[KzScanInput]
    warnings: list[str] = field(default_factory=list)

    @property
    def hv_values(self) -> np.ndarray:
        return np.asarray([scan.hv for scan in self.scans], dtype=float)


def discover_kz_inputs(folder: str | Path) -> list[Path]:
    """Retourne candidats BM d'un dossier KZ, sans charger les données."""
    paths, _ignored = _discover_kz_inputs_with_ignored(folder)
    return paths


def _looks_like_non_kz(path: Path) -> str | None:
    name = path.name.lower()
    if path.suffix.lower() == ".zip":
        return "zip/FS ignoré"
    if path.suffix.lower() not in _KZ_EXTS:
        return "extension ignorée"
    for bit in _REJECT_NAME_BITS:
        if bit in name:
            return f"nom non-KZ ({bit.strip()})"
    return None


def _discover_kz_inputs_with_ignored(folder: str | Path) -> tuple[list[Path], list[str]]:
    """Retourne candidats KZ et raisons d'exclusion non fatales."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"dossier KZ invalide: {root}")
    all_files = sorted(p for p in root.iterdir() if p.is_file())
    candidates: list[Path] = []
    ignored: list[str] = []
    for path in all_files:
        reason = _looks_like_non_kz(path)
        if reason:
            ignored.append(f"{path.name}: {reason}")
            continue
        candidates.append(path)
    if detect_format is None:
        return sorted(candidates), ignored
    valid: list[Path] = []
    for path in sorted(candidates):
        try:
            fmt = detect_format(path)
        except Exception:
            fmt = ""
        if fmt:
            valid.append(path)
        else:
            ignored.append(f"{path.name}: format non reconnu")
    return valid, ignored


def _hv_from_logbook(
    path: Path,
    records: list[dict] | None,
    mapping: dict[str, str] | None,
    session_folder: str | Path | None,
) -> float | None:
    if not records or not mapping:
        return None
    values = LogbookManager(records, mapping, session_folder).values_for_path(path)
    return _valid_hv(values.hv)


def _resolve_hv_fallback(
    path: Path,
    *,
    kz_logbook_records: list[dict] | None,
    kz_logbook_mapping: dict[str, str] | None,
    main_logbook_records: list[dict] | None,
    main_logbook_mapping: dict[str, str] | None,
    session_folder: str | Path | None,
    hv_fallback: float | None,
) -> tuple[float | None, str]:
    kz_hv = _hv_from_logbook(path, kz_logbook_records, kz_logbook_mapping, session_folder)
    if kz_hv is not None:
        return kz_hv, "kz_logbook"
    main_hv = _hv_from_logbook(path, main_logbook_records, main_logbook_mapping, session_folder)
    if main_hv is not None:
        return main_hv, "main_logbook"
    manual_hv = _valid_hv(hv_fallback)
    if manual_hv is not None:
        return manual_hv, "manual"
    return None, "unknown"


def _load_one_kz_scan(
    load: LoadFileFunc,
    path: Path,
    *,
    work_func: float,
    ef_offset: float,
    hv_for_load: float | None,
    hv_source_for_load: str,
) -> tuple[KzScanInput, list[str]]:
    warnings: list[str] = []
    first_error: Exception | None = None
    if hv_for_load is not None:
        try:
            d_file = load(str(path), float(work_func), float(ef_offset), hv=None)
            if d_file is not None:
                scan_file = scan_from_legacy_dict(d_file)
                hv_file = _valid_hv(scan_file.hv)
                if hv_file is not None and abs(hv_file - hv_for_load) > 1e-6:
                    warnings.append(
                        f"{path.name}: hν fichier={hv_file:.3f} eV remplace "
                        f"hν {hv_source_for_load}={hv_for_load:.3f} eV"
                    )
                scan_file = KzScanInput(
                    data=scan_file.data,
                    kpar=scan_file.kpar,
                    energy=scan_file.energy,
                    hv=scan_file.hv,
                    path=scan_file.path,
                    metadata={**dict(scan_file.metadata or {}), "hv_source": "file"},
                )
                return scan_file, warnings
            first_error = ValueError("loader indisponible sans hν")
        except Exception as exc:
            first_error = exc

    try:
        d = load(str(path), float(work_func), float(ef_offset), hv=hv_for_load)
    except Exception as exc:
        if first_error is not None:
            warnings.append(f"{path.name}: hν fichier absent/inutilisable ({first_error})")
        raise exc
    if d is None:
        raise ValueError("loader indisponible")
    scan = scan_from_legacy_dict(d)
    hv_loaded = _valid_hv(scan.hv)
    hv_source = hv_source_for_load if hv_for_load is not None else "file"
    if hv_for_load is not None and hv_loaded is not None and abs(hv_loaded - hv_for_load) > 1e-6:
        hv_source = "file"
        warnings.append(
            f"{path.name}: hν fichier={hv_loaded:.3f} eV remplace "
            f"hν {hv_source_for_load}={hv_for_load:.3f} eV"
        )
    elif first_error is not None and hv_for_load is not None:
        warnings.append(f"{path.name}: hν fichier absent/inutilisable; hν {hv_source_for_load} utilisé")
    return KzScanInput(
        data=scan.data,
        kpar=scan.kpar,
        energy=scan.energy,
        hv=scan.hv,
        path=scan.path,
        metadata={**dict(scan.metadata or {}), "hv_source": hv_source},
    ), warnings


def load_kz_stack(
    folder: str | Path,
    *,
    work_func: float,
    ef_offset: float,
    hv_fallback: float | None = None,
    kz_logbook_records: list[dict] | None = None,
    kz_logbook_mapping: dict[str, str] | None = None,
    main_logbook_records: list[dict] | None = None,
    main_logbook_mapping: dict[str, str] | None = None,
    session_folder: str | Path | None = None,
    load_func: LoadFileFunc | None = None,
) -> KzDataset:
    """Charge une série hν via les loaders BM existants."""
    load = load_func or load_arpes_file
    if load is None:
        raise RuntimeError("loaders ARPES indisponibles")
    paths, ignored = _discover_kz_inputs_with_ignored(folder)
    scans: list[KzScanInput] = []
    warnings: list[str] = list(ignored)
    for path in paths:
        hv_for_load, hv_source_for_load = _resolve_hv_fallback(
            path,
            kz_logbook_records=kz_logbook_records,
            kz_logbook_mapping=kz_logbook_mapping,
            main_logbook_records=main_logbook_records,
            main_logbook_mapping=main_logbook_mapping,
            session_folder=session_folder,
            hv_fallback=hv_fallback,
        )
        try:
            scan, scan_warnings = _load_one_kz_scan(
                load,
                path,
                work_func=work_func,
                ef_offset=ef_offset,
                hv_for_load=hv_for_load,
                hv_source_for_load=hv_source_for_load,
            )
        except Exception as exc:
            warnings.append(f"{path.name}: chargement impossible ({exc})")
            continue
        warnings.extend(scan_warnings)
        scans.append(scan)
    scans.sort(key=lambda s: s.hv)
    hv = np.asarray([s.hv for s in scans], dtype=float)
    if hv.size < 2:
        raise ValueError("kz: au moins deux fichiers avec hν valide requis")
    if np.unique(np.round(hv, 6)).size < 2:
        raise ValueError("kz: hν doit varier entre les scans")
    return KzDataset(folder=Path(folder), scans=scans, warnings=warnings)


def dataset_summary(dataset: KzDataset) -> dict[str, Any]:
    hv = dataset.hv_values
    hv_sources: dict[str, int] = {}
    for scan in dataset.scans:
        source = str((scan.metadata or {}).get("hv_source") or "unknown")
        hv_sources[source] = hv_sources.get(source, 0) + 1
    return {
        "folder": str(dataset.folder),
        "n_scans": len(dataset.scans),
        "hv_min": float(np.nanmin(hv)) if hv.size else None,
        "hv_max": float(np.nanmax(hv)) if hv.size else None,
        "hv_sources": hv_sources,
        "warnings": list(dataset.warnings),
    }
