"""Discovery and loading of hν series for kz maps."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import re

import numpy as np

from arpes.io.logbook import LogbookManager
from arpes.physics.kz import KzScanInput, scan_from_legacy_dict

try:
    from arpes.io.loaders import detect_format, load_arpes_file
    from arpes.io.loaders import _cls_angle_to_k_pi_over_a, _load_cls_fs_volume, _parse_cls_param
except Exception:  # pragma: no cover - environment without full loaders
    detect_format = None
    load_arpes_file = None
    _cls_angle_to_k_pi_over_a = None
    _load_cls_fs_volume = None
    _parse_cls_param = None


LoadFileFunc = Callable[..., dict | None]


_KZ_EXTS = {".ibw", ".pxt", ".txt"}
_CLS_PHOTON_LOG_RE = re.compile(
    r"Step:\s*(\d+)\s+Mono\s+PE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)
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
    """Return BM candidates from a KZ folder without loading the data."""
    paths, _ignored = _discover_kz_inputs_with_ignored(folder)
    return paths


def _looks_like_non_kz(path: Path) -> str | None:
    name = path.name.lower()
    if path.suffix.lower() == ".zip":
        return "zip/FS ignored"
    if path.suffix.lower() not in _KZ_EXTS:
        return "extension ignored"
    for bit in _REJECT_NAME_BITS:
        if bit in name:
            return f"non-KZ name ({bit.strip()})"
    return None


def _discover_kz_inputs_with_ignored(folder: str | Path) -> tuple[list[Path], list[str]]:
    """Return KZ candidates and non-fatal exclusion reasons."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"invalid KZ folder: {root}")
    photon_dirs = _discover_cls_photon_scan_dirs(root)
    if photon_dirs:
        return photon_dirs, []
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
            ignored.append(f"{path.name}: unrecognized format")
    return valid, ignored


def _cls_prefixes_with_steps(folder: Path) -> list[str]:
    prefixes: list[str] = []
    for param_file in sorted(folder.glob("*_param.txt")):
        prefix = param_file.name.removesuffix("_param.txt")
        if any(folder.glob(f"{prefix}_Cycle_*_Step_*.txt")):
            prefixes.append(prefix)
    return prefixes


def _is_cls_photon_scan_dir(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for prefix in _cls_prefixes_with_steps(folder):
        if (folder / f"{prefix}_log.txt").exists():
            return True
        if prefix.lower().startswith(("pe", "photon", "hv", "ps")):
            return True
    return False


def _discover_cls_photon_scan_dirs(root: Path) -> list[Path]:
    if _is_cls_photon_scan_dir(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and _is_cls_photon_scan_dir(p))


def _hv_by_step_from_cls_log(folder: Path, prefix: str) -> dict[int, float]:
    log_path = folder / f"{prefix}_log.txt"
    if log_path.exists():
        out: dict[int, float] = {}
        for step_s, hv_s in _CLS_PHOTON_LOG_RE.findall(log_path.read_text(errors="replace")):
            hv = _valid_hv(hv_s)
            if hv is not None:
                out[int(step_s)] = hv
        if out:
            return out

    param_path = folder / f"{prefix}_param.txt"
    if not param_path.exists():
        return {}
    txt = param_path.read_text(errors="replace")
    m = re.search(r"Energies:\s*(.*?)(?:\n\s*[A-Za-z][^\n:]*:|\Z)", txt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return {}
    vals = [
        hv for hv in (_valid_hv(x) for x in re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", m.group(1)))
        if hv is not None
    ]
    return {idx: hv for idx, hv in enumerate(vals)}


def _load_cls_photon_scan_folder(
    folder: Path,
    *,
    work_func: float,
    ef_offset: float,
    a_lattice: float = 0.0,
) -> tuple[list[KzScanInput], list[str]]:
    if _parse_cls_param is None or _load_cls_fs_volume is None or _cls_angle_to_k_pi_over_a is None:
        raise RuntimeError("CLS loader unavailable")
    prefixes = _cls_prefixes_with_steps(folder)
    if not prefixes:
        raise ValueError(f"no CLS Cycle/Step scan in {folder}")
    prefix = prefixes[0]
    p = _parse_cls_param(folder, prefix)
    volume, step_ids, n_cycles = _load_cls_fs_volume(folder, prefix)
    hv_by_step = _hv_by_step_from_cls_log(folder, prefix)
    warnings: list[str] = []
    if not hv_by_step:
        warnings.append(f"{folder.name}: hν missing from {prefix}_log.txt/{prefix}_param.txt")

    n_e, n_theta = volume.shape[1], volume.shape[2]
    energy_raw = p["energy_min"] + np.arange(n_e) * p["energy_delta"]
    central = p.get("central_energy")
    if central is not None and np.isfinite(float(central)):
        ef_kin_nominal = float(central)
        energy_reference = "central_energy_assumed_EF_for_photon_scan"
    else:
        first_hv = next(iter(hv_by_step.values()), None)
        ef_kin_nominal = float(first_hv) - float(work_func) if first_hv is not None else 0.5 * (
            float(energy_raw[0]) + float(energy_raw[-1])
        )
        energy_reference = "first_hv_minus_work_function"
        warnings.append(f"{folder.name}: Central Energy missing; energy axis based on first hν")
    energy = energy_raw - ef_kin_nominal + float(ef_offset)
    theta = p["angle_min"] + np.arange(n_theta) * p["angle_delta"]
    angular_offset = float(p.get("polar", 0.0) or 0.0)
    kpar = _cls_angle_to_k_pi_over_a(theta, ef_kin_nominal, a_lattice, angular_offset)

    scans: list[KzScanInput] = []
    for idx, step in enumerate(step_ids):
        hv = hv_by_step.get(int(step))
        if hv is None:
            warnings.append(f"{folder.name}: step {step} ignored, hν missing")
            continue
        scans.append(KzScanInput(
            data=np.asarray(volume[idx], dtype=float).T,
            kpar=np.asarray(kpar, dtype=float),
            energy=np.asarray(energy, dtype=float),
            hv=float(hv),
            path=str(folder / f"{prefix}_Cycle_*_Step_{step}.txt"),
            metadata={
                "source_format": "cls_txt",
                "scan_kind": "KZ",
                "kz_source": "cls_photon_scan",
                "hv_source": f"{prefix}_log",
                "cls_prefix": prefix,
                "step": int(step),
                "n_cycles": int(n_cycles),
                "energy_axis_original": "kinetic",
                "energy_axis": "E-EF",
                "energy_reference": energy_reference,
                "energy_raw_min": float(energy_raw[0]),
                "energy_raw_max": float(energy_raw[-1]),
                "ef_kinetic_nominal_from_hv": float(ef_kin_nominal),
                "polar": float(p.get("polar", 0.0) or 0.0),
                "theta_par_deg": theta,
                "x_axis_unit": "pi/a",
                "kx_unit": "pi/a",
            },
        ))
    return scans, warnings


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
                        f"{path.name}: file hν={hv_file:.3f} eV replaces "
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
            first_error = ValueError("loader unavailable without hν")
        except Exception as exc:
            first_error = exc

    try:
        d = load(str(path), float(work_func), float(ef_offset), hv=hv_for_load)
    except Exception as exc:
        if first_error is not None:
            warnings.append(f"{path.name}: file hν missing/unusable ({first_error})")
        raise exc
    if d is None:
        raise ValueError("loader unavailable")
    scan = scan_from_legacy_dict(d)
    hv_loaded = _valid_hv(scan.hv)
    hv_source = hv_source_for_load if hv_for_load is not None else "file"
    if hv_for_load is not None and hv_loaded is not None and abs(hv_loaded - hv_for_load) > 1e-6:
        hv_source = "file"
        warnings.append(
            f"{path.name}: file hν={hv_loaded:.3f} eV replaces "
            f"hν {hv_source_for_load}={hv_for_load:.3f} eV"
        )
    elif first_error is not None and hv_for_load is not None:
        warnings.append(f"{path.name}: file hν missing/unusable; hν {hv_source_for_load} used")
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
    a_lattice: float = 0.0,
    hv_fallback: float | None = None,
    kz_logbook_records: list[dict] | None = None,
    kz_logbook_mapping: dict[str, str] | None = None,
    main_logbook_records: list[dict] | None = None,
    main_logbook_mapping: dict[str, str] | None = None,
    session_folder: str | Path | None = None,
    load_func: LoadFileFunc | None = None,
) -> KzDataset:
    """Load an hν series through the existing BM loaders."""
    load = load_func or load_arpes_file
    if load is None:
        raise RuntimeError("ARPES loaders unavailable")
    paths, ignored = _discover_kz_inputs_with_ignored(folder)
    scans: list[KzScanInput] = []
    warnings: list[str] = list(ignored)
    for path in paths:
        if path.is_dir() and _is_cls_photon_scan_dir(path):
            try:
                ps_scans, ps_warnings = _load_cls_photon_scan_folder(
                    path,
                    work_func=work_func,
                    ef_offset=ef_offset,
                    a_lattice=a_lattice,
                )
            except Exception as exc:
                warnings.append(f"{path.name}: photon scan loading failed ({exc})")
                continue
            scans.extend(ps_scans)
            warnings.extend(ps_warnings)
            continue
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
            warnings.append(f"{path.name}: loading failed ({exc})")
            continue
        warnings.extend(scan_warnings)
        scans.append(scan)
    scans.sort(key=lambda s: s.hv)
    hv = np.asarray([s.hv for s in scans], dtype=float)
    if hv.size < 2:
        raise ValueError("kz: at least two files with valid hν are required")
    if np.unique(np.round(hv, 6)).size < 2:
        raise ValueError("kz: hν must vary between scans")
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
