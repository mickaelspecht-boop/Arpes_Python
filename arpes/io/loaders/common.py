"""Helpers communs aux loaders ARPES (Solaris/BESSY/CLS).

Contient :
- modèles `ARPESData`, `LoaderSpec`, `ARPESDataValidationError` ;
- registre `_LOADER_REGISTRY` + `register_loader`/`registered_loaders` ;
- validation `assert_arpes_data_valid` ;
- helpers numériques (`_valid_positive_float`, `_valid_float`,
  `_first_present`, `_loadtxt_float32`, `_transpose_to_axes`) ;
- helpers métadonnées (`_add_loader_diagnostics`,
  `_add_instrument_resolution_metadata`) ;
- conversion d'angle commune `_cls_angle_to_k_pi_over_a`.

Voir `__init__.py` pour la convention interne ARPESData (E−EF en eV, etc.).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


SUPPORTED_SOLARIS_EXTENSIONS = {".ibw", ".pxt", ".zip"}
_C_ARPES = 0.51233


@dataclass
class ARPESData:
    data: np.ndarray
    energy: np.ndarray
    kx: Optional[np.ndarray] = None
    ky: Optional[np.ndarray] = None
    hv: Optional[float] = None
    path: Optional[Path] = None
    source_format: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    xarray: Any = None
    @property
    def ev_arr(self): return self.energy
    @property
    def kpar(self): return self.kx
    def as_legacy_bandmap_dict(self) -> dict[str, Any]:
        return {"kpar": self.kx, "ev_arr": self.energy, "data": self.data,
                "hv": self.hv, "path": str(self.path) if self.path else None,
                "source_format": self.source_format, "metadata": self.metadata}


@dataclass(frozen=True)
class LoaderSpec:
    name: str
    detector_fn: Callable[[Path], bool]
    loader_fn: Callable[..., ARPESData]
    description: str = ""


class ARPESDataValidationError(ValueError):
    """Erreur levée quand un loader ne respecte pas la convention interne."""


_LOADER_REGISTRY: dict[str, LoaderSpec] = {}


def register_loader(name: str, detector_fn: Callable[[Path], bool],
                    loader_fn: Callable[..., ARPESData], description: str = "") -> None:
    """Enregistre un loader ARPES.

    `detector_fn` doit être rapide et sans effet de bord. `loader_fn` doit
    retourner un `ARPESData` conforme à la convention interne.
    """
    key = str(name).strip()
    if not key:
        raise ValueError("Nom de loader vide")
    if key in _LOADER_REGISTRY:
        raise ValueError(f"Loader déjà enregistré: {key}")
    _LOADER_REGISTRY[key] = LoaderSpec(key, detector_fn, loader_fn, description)


def registered_loaders() -> dict[str, LoaderSpec]:
    """Retourne une copie du registre pour inspection/tests."""
    return dict(_LOADER_REGISTRY)


def _is_monotonic_axis(axis: np.ndarray) -> bool:
    if axis.ndim != 1:
        return False
    if axis.size < 2:
        return True
    finite = axis[np.isfinite(axis)]
    if finite.size < 2:
        return True
    d = np.diff(finite)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _validation_issue(path: Path | None, source_format: str, message: str) -> str:
    name = path.name if path is not None else "<memory>"
    return f"{source_format}:{name}: {message}"


def _append_unique_list(meta: dict[str, Any], key: str, values: list[str]) -> None:
    cur = meta.setdefault(key, [])
    if not isinstance(cur, list):
        cur = [str(cur)]
        meta[key] = cur
    for value in values:
        if value and value not in cur:
            cur.append(value)


def _add_loader_diagnostics(meta: dict[str, Any], *, capability: str,
                            assumptions: list[str] | None = None,
                            warnings_: list[str] | None = None,
                            geometry_confidence: str = "medium",
                            axis_sources: dict[str, str] | None = None) -> None:
    """Ajoute un diagnostic explicite sur ce que le loader suppose.

    Ces champs ne changent pas la donnée, ils rendent visibles les conventions
    automatiques pour éviter de confondre "chargé sans crash" et "validé
    physiquement pour tout le labo".
    """
    meta["loader_capability"] = capability
    meta["geometry_confidence"] = geometry_confidence
    if axis_sources:
        meta["axis_sources"] = dict(axis_sources)
    _append_unique_list(meta, "loader_assumptions", assumptions or [])
    _append_unique_list(meta, "loader_warnings", warnings_ or [])


def assert_arpes_data_valid(ds: ARPESData) -> ARPESData:
    """Valide strictement la sortie d'un loader.

    Les incohérences de shape/axes lèvent `ARPESDataValidationError`. Les points
    plausibles mais suspects sont stockés dans `metadata["validation_warnings"]`
    et émis via `warnings.warn`.
    """
    errors: list[str] = []
    warnings_list: list[str] = []
    if not isinstance(ds, ARPESData):
        raise ARPESDataValidationError(f"Objet retourné non ARPESData: {type(ds)!r}")

    fmt = str(ds.source_format or "unknown")
    path = ds.path if isinstance(ds.path, Path) else Path(ds.path) if ds.path else None
    data = np.asarray(ds.data)
    energy = np.asarray(ds.energy)

    if data.ndim != 2:
        errors.append(_validation_issue(path, fmt, f"data doit être 2D `(n_k,n_E)`, reçu {data.shape}"))
    if energy.ndim != 1:
        errors.append(_validation_issue(path, fmt, f"energy doit être 1D, reçu {energy.shape}"))
    elif data.ndim == 2 and data.shape[1] != energy.size:
        errors.append(_validation_issue(
            path, fmt, f"shape incohérente: data.shape[1]={data.shape[1]} mais len(energy)={energy.size}"
        ))
    if energy.ndim == 1 and not _is_monotonic_axis(energy):
        errors.append(_validation_issue(path, fmt, "energy doit être monotone"))

    if ds.kx is None:
        errors.append(_validation_issue(path, fmt, "kx/kpar manquant pour la band map"))
    else:
        kx = np.asarray(ds.kx)
        if kx.ndim != 1:
            errors.append(_validation_issue(path, fmt, f"kx doit être 1D, reçu {kx.shape}"))
        elif data.ndim == 2 and kx.size != data.shape[0]:
            errors.append(_validation_issue(
                path, fmt, f"shape incohérente: len(kx)={kx.size} mais data.shape[0]={data.shape[0]}"
            ))
        if kx.ndim == 1 and not _is_monotonic_axis(kx):
            errors.append(_validation_issue(path, fmt, "kx doit être monotone"))

    if ds.ky is not None:
        ky = np.asarray(ds.ky)
        if ky.ndim != 1:
            errors.append(_validation_issue(path, fmt, f"ky doit être 1D, reçu {ky.shape}"))
        elif not _is_monotonic_axis(ky):
            errors.append(_validation_issue(path, fmt, "ky doit être monotone"))

    if not isinstance(ds.metadata, dict):
        errors.append(_validation_issue(path, fmt, "metadata doit être un dict"))
    else:
        fs_data = ds.metadata.get("fs_data")
        if fs_data is not None:
            fs = np.asarray(fs_data)
            fs_kx = np.asarray(ds.metadata.get("fs_kx", []))
            fs_ky = np.asarray(ds.metadata.get("fs_ky", []))
            fs_energy = np.asarray(ds.metadata.get("fs_energy", energy))
            if fs.ndim != 3:
                errors.append(_validation_issue(path, fmt, f"fs_data doit être 3D `(n_ky,n_kx,n_E)`, reçu {fs.shape}"))
            elif fs.shape != (fs_ky.size, fs_kx.size, fs_energy.size):
                errors.append(_validation_issue(
                    path, fmt,
                    "shape FS incohérente: "
                    f"fs_data={fs.shape}, len(fs_ky)={fs_ky.size}, len(fs_kx)={fs_kx.size}, len(fs_energy)={fs_energy.size}"
                ))
        hv = ds.hv if ds.hv is not None else ds.metadata.get("hv")
        try:
            hvf = float(hv)
            if not np.isfinite(hvf) or hvf <= 0:
                warnings_list.append("hν absent ou non positif")
        except (TypeError, ValueError):
            warnings_list.append("hν absent ou non numérique")

    if data.size == 0:
        errors.append(_validation_issue(path, fmt, "data vide"))
    elif not np.any(np.isfinite(data)):
        errors.append(_validation_issue(path, fmt, "data ne contient aucune valeur finie"))

    if errors:
        raise ARPESDataValidationError("\n".join(errors))

    if warnings_list:
        ds.metadata.setdefault("validation_warnings", [])
        for msg in warnings_list:
            if msg not in ds.metadata["validation_warnings"]:
                ds.metadata["validation_warnings"].append(msg)
            _append_unique_list(ds.metadata, "loader_warnings", [msg])
            warnings.warn(_validation_issue(path, fmt, msg), RuntimeWarning, stacklevel=2)
    return ds


def _require_erlab():
    try:
        import erlab.io  # type: ignore
    except Exception as exc:
        raise ImportError("erlab n'est pas disponible. Active ton environnement peaks/conda.") from exc
    return erlab.io


def _set_da30_loader(erlab_io) -> None:
    try: erlab_io.set_loader("da30")
    except Exception: pass


def _transpose_to_axes(arr: np.ndarray, dims: list[str], order: tuple[str, ...]) -> np.ndarray:
    return np.transpose(arr, [dims.index(name) for name in order])


def _loadtxt_float32(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def static_polar_for_kx(
    polar: float | None,
    scan_values,
    *,
    is_fs: bool,
    motor_present: bool = True,
) -> tuple[float, float, bool]:
    """Return `(polar_for_kx, raw_polar, ignored_scan_polar)`.

    In FS scans, the manipulator axis used as the second momentum coordinate
    may be recorded both as a scan list and as the instantaneous motor
    position at loop start. If that raw motor value belongs to the scan list,
    it is a scanned coordinate, not a static analyzer polar offset for
    theta→kx.
    """
    raw = float(polar or 0.0)
    if not is_fs or not motor_present or scan_values is None:
        return raw, raw, False
    vals = np.asarray(scan_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(raw):
        return raw, raw, False
    unique = np.unique(vals)
    step = abs(float(np.nanmedian(np.diff(unique)))) if unique.size > 1 else 0.0
    tol = max(1e-6, 0.25 * step)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if (lo - tol) <= raw <= (hi + tol) or np.nanmin(np.abs(vals - raw)) <= tol:
        return 0.0, raw, True
    return raw, raw, False


def scan_axis_summary(scan_values) -> dict[str, float | int] | None:
    vals = np.asarray(scan_values, dtype=float) if scan_values is not None else np.asarray([], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    unique = np.unique(vals)
    step = float(np.nanmedian(np.diff(unique))) if unique.size > 1 else 0.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return {
        "min": lo,
        "max": hi,
        "center": 0.5 * (lo + hi),
        "step": step,
        "n": int(vals.size),
    }


def _valid_positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _valid_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def _add_instrument_resolution_metadata(
    meta: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
    pass_energy: Any = None,
    lens_mode: Any = None,
    energy_step: Any = None,
    angle_step: Any = None,
) -> None:
    """Normalise les paramètres analyseur utiles à l'estimation de résolution."""
    src = source or meta
    if pass_energy is None:
        pass_energy = _first_present(src, (
            "pass_energy_eV", "Pass Energy", "Pass energy", "PassEnergy",
            "pass_energy", "Pass energy (eV)",
        ))
    if lens_mode is None:
        lens_mode = _first_present(src, (
            "lens_mode", "Lens Mode", "Lens mode", "LensMode",
        ))
    if energy_step is None:
        energy_step = _first_present(src, (
            "energy_step_eV", "Energy Step", "Energy step", "EnergyStep",
            "Energy delta", "energy_delta", "Energy Delta",
        ))
    if angle_step is None:
        angle_step = _first_present(src, (
            "angle_step_deg", "Angle Step", "Angle step", "AngleStep",
            "Angle delta", "angle_delta", "Angle Delta",
        ))

    pe = _valid_positive_float(pass_energy)
    if pe is not None:
        meta["pass_energy_eV"] = pe
    lm = str(lens_mode).strip() if lens_mode not in (None, "") else ""
    if lm:
        meta["lens_mode"] = lm
    es = _valid_positive_float(energy_step)
    if es is not None:
        meta["energy_step_eV"] = es
    astep = _valid_float(angle_step)
    meta["angle_step_deg"] = astep if astep is not None else None


def _cls_angle_to_k_pi_over_a(angle_deg, ef_kinetic: float, a_lattice: float, angular_offset_deg: float = 0.0):
    """Conversion angle→k (en pi/a) commune CLS/BESSY/Solaris.

    Formule Scienta : k_par = C·√Ek·sin(θ−θ0).
    """
    ek = max(float(ef_kinetic), 1e-9)
    theta = np.radians(np.asarray(angle_deg, dtype=float) - float(angular_offset_deg))
    return (_C_ARPES * np.sqrt(ek) * np.sin(theta)) * float(a_lattice) / np.pi


def detect_format(path: str | Path) -> str:
    """Itère le registre et retourne le nom du premier loader qui détecte `path`."""
    p = Path(path)
    for name, spec in _LOADER_REGISTRY.items():
        try:
            if spec.detector_fn(p):
                return name
        except Exception:
            continue
    return "unknown"


def detect_scan_kind(path: str | Path, format_hint: str | None = None) -> str:
    """Détection légère du type de scan: `BM`, `FS` ou `unknown`.

    Cette fonction est faite pour l'interface de navigation. Elle ne charge pas
    les données ARPES complètes.
    """
    p = Path(path)
    fmt = format_hint or detect_format(p)
    if fmt == "cls_txt":
        from .cls import _is_cls_fs_dir, _is_cls_bm_file
        if _is_cls_fs_dir(p):
            return "FS"
        if _is_cls_bm_file(p):
            return "BM"
    if fmt == "bessy_ses_ibw":
        from .bessy import _read_ibw5_info
        try:
            info = _read_ibw5_info(p)
        except (OSError, ValueError):
            return "unknown"
        if len(info.dims) == 2:
            return "BM"
        if len(info.dims) == 3:
            return "FS"
    if fmt == "solaris_da30":
        if p.suffix.lower() == ".zip":
            return "FS"
        return "unknown"
    return "unknown"


def load_arpes(path, *, work_func: float, ef_offset: float = 0.0, a_lattice: float = 3.96,
               format_hint: str | None = None, hv: float | None = None,
               temperature: float | None = None, azi: float = 0.0, pol: str = "",
               angle_offsets: dict | None = None,
               bessy_energy_reference: str = "auto") -> ARPESData:
    """Dispatch principal : choisit le loader via le registre puis valide la sortie."""
    fmt = format_hint or detect_format(path)
    spec = _LOADER_REGISTRY.get(fmt)
    if spec is None:
        known = ", ".join(_LOADER_REGISTRY) or "aucun"
        raise ValueError(f"Format non supporté pour {Path(path).name!r}: {fmt}. Formats: {known}.")
    ds = spec.loader_fn(
        path,
        work_func=work_func,
        ef_offset=ef_offset,
        a_lattice=a_lattice,
        hv=hv,
        temperature=temperature,
        azi=azi,
        pol=pol,
        angle_offsets=angle_offsets,
        bessy_energy_reference=bessy_energy_reference,
    )
    return assert_arpes_data_valid(ds)


def load_arpes_file(path: str, work_func: float, ef_offset: float,
                    a_lattice: float = 3.96, hv: float | None = None,
                    temperature: float | None = None,
                    azi: float | None = None,
                    pol: str = "",
                    angle_offsets: dict | None = None,
                    bessy_energy_reference: str = "auto") -> dict | None:
    """Wrapper utilitaire renvoyant un dict legacy bandmap (None si erlab absent)."""
    try:
        ds = load_arpes(path, work_func=work_func, ef_offset=ef_offset,
                        a_lattice=a_lattice, hv=hv,
                        temperature=temperature,
                        azi=float(azi) if azi is not None else 0.0,
                        pol=pol,
                        angle_offsets=angle_offsets,
                        bessy_energy_reference=bessy_energy_reference)
    except RuntimeError as exc:
        if "erlab" in str(exc).lower():
            return None
        raise
    return ds.as_legacy_bandmap_dict()


def loader_label(source_format: str | None, metadata: dict | None = None) -> str:
    """Label court et stable pour l'affichage utilisateur (Solaris/BESSY/CLS)."""
    fmt = (source_format or "").strip()
    md = metadata or {}
    explicit = str(md.get("loader_label") or md.get("lab_label") or "").strip()
    if explicit:
        return explicit
    lab = str(md.get("lab") or "").strip().lower()
    if "cls" in lab or "lnls" in lab:
        return "CLS"
    labels = {
        "cls_txt": "CLS",
        "solaris_da30": "Solaris",
        "bessy_ses_ibw": "BESSY",
    }
    if fmt in labels:
        return labels[fmt]
    if not fmt:
        return ""
    return fmt.replace("_", " ").replace("-", " ").title()
