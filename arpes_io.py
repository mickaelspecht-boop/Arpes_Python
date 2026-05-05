#!/usr/bin/env python3
"""Couche IO commune pour données ARPES: Solaris/DA30 + CLS/LNLS txt.

Convention interne ARPES Explorer
=================================

Tous les loaders doivent retourner un :class:`ARPESData` conforme à cette
convention, indépendante du laboratoire source.

- Axe énergie : `energy` est toujours `E - EF` en eV, avec `0` à EF quand la
  calibration est connue. Les énergies cinétiques brutes restent dans
  `metadata` si elles sont utiles au diagnostic.
- Band map : `data` a toujours la shape `(n_k, n_E)`, donc `data[:, i]` est une
  MDC et `data[j, :]` est une EDC.
- FS / volume : quand un volume est disponible, il est stocké dans
  `metadata["fs_data"]` avec la shape `(n_ky, n_kx, n_E)`. Les axes associés
  sont `metadata["fs_ky"]`, `metadata["fs_kx"]`, `metadata["fs_energy"]`.
- Unités k : les axes `kx`/`ky` sont en `pi/a`. Les angles bruts restent dans
  `metadata` (`theta_par_deg`, `fs_ky_angle_deg`, etc.).
- Convention CLS actuelle : `kx` est calculé depuis
  `theta_raw - polar - theta0_deg`; `ky` depuis `tilt_raw - tilt0_deg`.
  Les corrections cristallines `azi`/rotation de ZDB restent des métadonnées
  ou des corrections de visualisation tant qu'elles ne sont pas propagées par
  un modèle géométrique explicite.

Avant d'ajouter un nouveau loader, il doit passer `assert_arpes_data_valid()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json, os, re, struct, warnings
import numpy as np

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

SUPPORTED_SOLARIS_EXTENSIONS = {".ibw", ".pxt", ".zip"}
_C_ARPES = 0.51233
_CLS_CACHE_VERSION = 2
_CYCLE_STEP_RE = re.compile(r"Cycle_(\d+)_Step_(\d+)")
_LOADER_REGISTRY: dict[str, LoaderSpec] = {}
_IBW5_BIN_HEADER_SIZE = 64
_IBW5_WAVE_HEADER_SIZE = 320

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


def _is_cls_fs_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for param_file in path.glob("*_param.txt"):
        prefix = param_file.name.removesuffix("_param.txt")
        if any(path.glob(f"{prefix}_Cycle_*_Step_*.txt")):
            return True
    return False

def _is_cls_bm_file(path: Path) -> bool:
    return path.is_file() and (path.parent / f"{path.name}_param.txt").exists()

def _is_solaris_da30_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SOLARIS_EXTENSIONS

def _is_bessy_ses_ibw(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".ibw":
        return False
    try:
        info = _read_ibw5_info(path)
        note = _read_ibw5_note(path, info).encode("latin1", errors="ignore")
    except (OSError, ValueError):
        return False
    # `Instrument=R8000` est la marque non-ambiguë du R8000 BESSY (les exports
    # testés Ba122 ont `Instrument=R8000-8ES202`). On exige cette signature en
    # plus de `[SES]` pour éviter d'attraper les exports DA30 qui contiennent
    # aussi `Energy Scale=Kinetic`.
    return b"[SES]" in note and b"Instrument=R8000" in note

def detect_format(path: str | Path) -> str:
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
        if _is_cls_fs_dir(p):
            return "FS"
        if _is_cls_bm_file(p):
            return "BM"
    if fmt == "bessy_ses_ibw":
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

def load_solaris_da30_bandmap(path, work_func: float, ef_offset: float = 0.0,
                              a_lattice: float = 3.96, convert_kspace: bool = True) -> ARPESData:
    path = Path(path); erlab_io = _require_erlab(); _set_da30_loader(erlab_io)
    da = erlab_io.load(str(path))
    hv = float(da.attrs.get("hv", np.nan))
    if not np.isfinite(hv) or hv <= 0:
        raise ValueError(
            f"hv absent ou non valide dans {path.name} (lu={hv!r}). "
            f"Vérifie le logbook ou saisis hν manuellement avant de recharger."
        )
    ef_kin = hv - work_func
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        da_be = da.assign_coords(eV=da.eV - ef_kin + ef_offset, hv=hv, xi=0.0)
    da_be.attrs["configuration"] = 1
    if hasattr(da_be, "kspace"): da_be.kspace.work_function = work_func
    da_out = da_be.kspace.convert() if convert_kspace else da_be
    kx = ky = None
    if "kx" in da_out.coords:
        da_out = da_out.assign_coords(kx=da_out.kx * a_lattice / np.pi)
        kx_arr = np.asarray(da_out.kx.values, dtype=float)
        kx = kx_arr if kx_arr.ndim == 1 else None
    if "ky" in da_out.coords:
        da_out = da_out.assign_coords(ky=da_out.ky * a_lattice / np.pi)
        ky_arr = np.asarray(da_out.ky.values, dtype=float)
        ky = ky_arr if ky_arr.ndim == 1 else None
    energy = np.asarray(da_out.eV.values, dtype=float)
    arr = np.asarray(da_out.values, dtype=float).squeeze()
    dims = list(getattr(da_out, "dims", []))
    meta = dict(da.attrs); meta.update({
        "lab": "Solaris",
        "loader_label": "Solaris",
        "x_axis_unit": "pi/a",
        "kx_unit": "pi/a",
        "fs_source": "solaris_da30",
        "energy_reference": "hv_minus_work_function",
        "energy_axis_original": "kinetic",
        "energy_axis": "E-EF",
        "ef_kinetic_from_hv": float(ef_kin),
    })
    _add_instrument_resolution_metadata(meta, source=dict(da.attrs))
    # Température : essayer plusieurs noms d'attribut courants Solaris/erlab
    if "temperature" not in meta or not np.isfinite(float(meta.get("temperature") or np.nan)):
        for key in ("temperature_a", "TempA", "tempA", "T", "sample_temperature", "temperature_b"):
            v = da.attrs.get(key)
            try:
                vf = float(v)
                if np.isfinite(vf) and vf > 0:
                    meta["temperature"] = vf
                    break
            except (TypeError, ValueError):
                continue
    data = arr
    if arr.ndim == 3 and kx is not None and ky is not None:
        if all(d in dims for d in ("ky", "kx", "eV")):
            fs_data = _transpose_to_axes(arr, dims, ("ky", "kx", "eV"))
        elif all(d in dims for d in ("kx", "ky", "eV")):
            fs_data = _transpose_to_axes(arr, dims, ("ky", "kx", "eV"))
        else:
            fs_data = arr
            if fs_data.shape[-1] != len(energy):
                e_axis = next((i for i,s in enumerate(fs_data.shape) if s == len(energy)), 2)
                fs_data = np.moveaxis(fs_data, e_axis, -1)
        if fs_data.shape[0] == len(kx) and fs_data.shape[1] == len(ky):
            fs_data = np.transpose(fs_data, (1,0,2))
        meta.update({"fs_data": fs_data, "fs_kx": kx, "fs_ky": ky, "fs_energy": energy, "fs_kind": "kxky"})
        data = np.nanmean(fs_data, axis=0)
    elif arr.ndim == 2 and kx is not None:
        if arr.shape == (len(energy), len(kx)): data = arr.T
        elif arr.shape == (len(kx), len(energy)): data = arr
    _add_loader_diagnostics(
        meta,
        capability="Solaris/DA30 files readable by erlab da30 loader",
        assumptions=[
            "erlab DA30 metadata provide reliable hv and angular geometry",
            "E-EF is computed from hv - work_function + ef_offset",
            "k-space conversion follows erlab DA30 conventions",
        ],
        geometry_confidence="high",
        axis_sources={
            "energy": "erlab eV axis shifted by hv - work_function",
            "kx": "erlab kspace.convert",
            "ky": "erlab kspace.convert when present",
            "hv": "Solaris/DA30 metadata",
        },
    )
    ds = ARPESData(data=data, energy=energy, kx=kx, ky=ky, hv=hv, path=path,
                   source_format="solaris_da30", metadata=meta, xarray=da_out)
    return assert_arpes_data_valid(ds)

def _parse_cls_param(folder: Path, prefix: str) -> dict[str, Any]:
    param_file = folder / f"{prefix}_param.txt"
    if not param_file.exists(): raise FileNotFoundError(f"Paramètres CLS introuvables : {param_file}")
    txt = param_file.read_text(errors="replace"); lines = txt.strip().splitlines()
    em = re.search(r"Energy min:\s*([-\d.]+);\s*Energy delta:\s*([-\d.]+)", txt)
    am = re.search(r"Angle min:\s*([-\d.]+);\s*Angle delta:\s*([-\d.]+)", txt)
    if not em or not am: raise ValueError(f"Impossible de lire Energy/Angle min-delta dans {param_file}")
    def fmatch(pat):
        m = re.search(pat, txt); return float(m.group(1)) if m else None
    def smatch(pat):
        m = re.search(pat, txt); return m.group(1) if m else None
    phi_line = next((l for l in lines if re.match(r"^\s*-?[\d.]+(\s+-?[\d.]+)+\s*$", l)), None)
    phi_values = np.array([float(x) for x in phi_line.split()]) if phi_line is not None else None
    json_line = next((l for l in lines if l.strip().startswith("{")), None)
    try: motors = json.loads(json_line).get("d", {}) if json_line else {}
    except Exception: motors = {}
    return {"energy_min": float(em.group(1)), "energy_delta": float(em.group(2)),
            "angle_min": float(am.group(1)), "angle_delta": float(am.group(2)),
            "pass_energy": fmatch(r"Pass energy:\s*([\d.]+)"),
            "lens_mode": smatch(r"Lens mode:\s*(\S+)"),
            "central_energy": fmatch(r"Central Energy:\s*([-\d.]+)"),
            "dwell_ms": fmatch(r"Dwell Time:\s*([\d.]+)"),
            "acquisition_mode": int(fmatch(r"Acquisition mode:\s*(\d+)") or 0),
            "phi_values": phi_values,
            "polar": motors.get("P", {}).get("position", 0.0),
            "tilt_ref": motors.get("T", {}).get("position", 0.0),
            "x": motors.get("X", {}).get("position", 0.0),
            "y": motors.get("Y", {}).get("position", 0.0),
            "z": motors.get("Z", {}).get("position", 0.0)}

def _cls_angle_to_k_pi_over_a(angle_deg, ef_kinetic: float, a_lattice: float, angular_offset_deg: float = 0.0):
    ek = max(float(ef_kinetic), 1e-9)
    theta = np.radians(np.asarray(angle_deg, dtype=float) - float(angular_offset_deg))
    return (_C_ARPES * np.sqrt(ek) * np.sin(theta)) * float(a_lattice) / np.pi

@dataclass(frozen=True)
class _IBW5Info:
    dtype: str
    dims: tuple[int, ...]
    npnts: int
    data_offset: int
    note_offset: int
    note_size: int
    sf_a: tuple[float, float, float, float]
    sf_b: tuple[float, float, float, float]
    wave_name: str

def _read_ibw5_info(path: Path) -> _IBW5Info:
    with path.open("rb") as f:
        header = f.read(_IBW5_BIN_HEADER_SIZE + _IBW5_WAVE_HEADER_SIZE)
    if len(header) < _IBW5_BIN_HEADER_SIZE + _IBW5_WAVE_HEADER_SIZE:
        raise ValueError(f"IBW trop court : {path.name}")
    if int.from_bytes(header[:2], "little") != 5:
        raise ValueError(f"Seuls les IBW v5 sont supportés pour BESSY : {path.name}")
    wfm_size = struct.unpack_from("<I", header, 4)[0]
    note_size = struct.unpack_from("<I", header, 12)[0]
    wave0 = _IBW5_BIN_HEADER_SIZE
    npnts = struct.unpack_from("<I", header, wave0 + 12)[0]
    wave_type = struct.unpack_from("<H", header, wave0 + 16)[0]
    name_raw = header[wave0 + 28:wave0 + 60].split(b"\0", 1)[0]
    wave_name = name_raw.decode("latin1", errors="replace")
    dims = tuple(int(x) for x in struct.unpack_from("<4I", header, wave0 + 68) if int(x) > 0)
    sf_a = tuple(float(x) for x in struct.unpack_from("<4d", header, wave0 + 84))
    sf_b = tuple(float(x) for x in struct.unpack_from("<4d", header, wave0 + 116))
    dtype_by_type = {
        0x02: "<f4",
        0x04: "<f8",
        0x08: "<i1",
        0x10: "<i2",
        0x20: "<i4",
        0x40: "<u1",
        0x80: "<u2",
        0x100: "<u4",
    }
    dtype = dtype_by_type.get(wave_type)
    if dtype is None:
        raise ValueError(f"Type IBW BESSY non supporté ({wave_type}) dans {path.name}")
    if not dims or int(np.prod(dims)) != npnts:
        raise ValueError(f"Dimensions IBW incohérentes dans {path.name}: dims={dims}, npnts={npnts}")
    return _IBW5Info(
        dtype=dtype,
        dims=dims,
        npnts=int(npnts),
        data_offset=_IBW5_BIN_HEADER_SIZE + _IBW5_WAVE_HEADER_SIZE,
        note_offset=_IBW5_BIN_HEADER_SIZE + int(wfm_size),
        note_size=int(note_size),
        sf_a=sf_a,
        sf_b=sf_b,
        wave_name=wave_name,
    )

def _read_ibw5_note(path: Path, info: _IBW5Info) -> str:
    with path.open("rb") as f:
        f.seek(info.note_offset)
        raw = f.read(info.note_size)
    if raw.startswith(b"\0"):
        raw = raw[1:]
    return raw.decode("latin1", errors="replace").replace("\r", "\n")

def _parse_ses_note(note: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for line in note.replace("\x0b", "\t").splitlines():
        s = line.strip()
        if not s or s.startswith("[") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            meta[key] = float(val)
        except ValueError:
            meta[key] = val
    p_axis: list[float] = []
    r_axis: list[float] = []
    for line in note.replace("\x0b", "\t").splitlines():
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) < 3:
            continue
        try:
            float(cols[0])
            p_axis.append(float(cols[1]))
            r_axis.append(float(cols[2]))
        except ValueError:
            continue
    if p_axis:
        meta["P-Axis scan"] = np.asarray(p_axis, dtype=float)
    if r_axis:
        meta["R-Axis scan"] = np.asarray(r_axis, dtype=float)
    return meta

def _load_ibw5_numeric(path: Path, info: _IBW5Info) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.dtype(info.dtype), count=info.npnts, offset=info.data_offset)
    if arr.size != info.npnts:
        raise ValueError(f"Lecture IBW incomplète dans {path.name}: {arr.size}/{info.npnts}")
    return arr.reshape(info.dims, order="F")

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

def load_bessy_ses_ibw(path, *, work_func: float = 4.031, ef_offset: float = 0.0,
                       a_lattice: float = 3.96, hv: float | None = None,
                       temperature: float | None = None, azi: float = 0.0,
                       pol: str = "", angle_offsets: dict | None = None,
                       bessy_energy_reference: str = "auto") -> ARPESData:
    """Charge les exports Igor Binary Wave SES/R8000 de BESSY.

    Les fichiers testés contiennent des axes Igor `(E_kin, theta[, P])` et une
    note `@[SES]`. L'énergie photon n'est pas fiable dans la note (`0` dans les
    fichiers Ba122), donc `hv` doit être fourni par l'utilisateur/logbook.
    """
    path = Path(path)
    info = _read_ibw5_info(path)
    note = _read_ibw5_note(path, info)
    ses = _parse_ses_note(note)
    hv_val = _valid_positive_float(hv)
    if hv_val is None:
        hv_val = _valid_positive_float(ses.get("Excitation Energy"))
    if hv_val is None:
        hv_val = _valid_positive_float(ses.get("Monochromator Energy"))

    temp_val = _valid_positive_float(temperature)
    angle_offsets = angle_offsets or {}
    theta0_deg = float(angle_offsets.get("theta0_deg", 0.0) or 0.0)
    tilt0_deg = float(angle_offsets.get("tilt0_deg", 0.0) or 0.0)
    polar = float(ses.get("P-Axis", 0.0) or 0.0)
    raw = _load_ibw5_numeric(path, info).astype(np.float32, copy=False)
    n_e, n_theta = info.dims[0], info.dims[1]
    energy_raw = info.sf_b[0] + np.arange(n_e, dtype=float) * info.sf_a[0]
    theta = info.sf_b[1] + np.arange(n_theta, dtype=float) * info.sf_a[1]
    # Détection de l'échelle énergie SES : Kinetic (cas standard) vs Binding.
    # Sur Binding, l'axe est déjà référencé à EF côté SES, donc on ne soustrait
    # rien et on inverse le signe pour respecter la convention E−EF (positif
    # au-dessus d'EF). Pour kx il faut Ek réel, donc hν est requis.
    energy_scale_raw = str(ses.get("Energy Scale") or ses.get("Energy Unit") or "Kinetic").strip()
    energy_scale = energy_scale_raw.lower()
    is_binding_axis = energy_scale.startswith("bind") or energy_scale in {"be", "e_b", "binding energy"}

    mode = str(bessy_energy_reference or "auto").strip().lower()
    if mode in {"center", "ses", "ses_center"}:
        mode = "ses_center_energy"
    elif mode in {"hv", "hv_phi", "hv-work-function", "hv_minus_phi"}:
        mode = "hv_minus_work_function"
    elif mode not in {"auto", "ses_center_energy", "hv_minus_work_function"}:
        mode = "auto"

    center_energy = _valid_positive_float(ses.get("Center Energy"))
    center_energy_from_fallback = center_energy is None
    if center_energy is None:
        center_energy = float(np.nanmean([energy_raw[0], energy_raw[-1]]))
    # Sur les exports BESSY/SES R8000 testés (Ba122), Excitation/Monochromator
    # Energy = 0 dans la note, donc hν doit venir du logbook ou être passé
    # explicitement. Center Energy est en principe le centre de la fenêtre
    # cinétique enregistrée — fiable comme proxy d'EF SEULEMENT si l'opérateur
    # a effectivement centré la BM sur EF. Sinon (ex: BM à 30 eV de binding),
    # Center Energy place EF à plusieurs eV de zéro et casse la calibration.
    ef_kin_from_hv = float(hv_val - work_func) if hv_val is not None else None
    loader_warnings: list[str] = []
    # Mode Auto = Center Energy (le réglage analyseur reflète l'intention de
    # l'expérimentateur). hν-φ reste un override explicite parce que sur les
    # vieux fichiers BESSY (ex: Ba122) le logbook hν est souvent erroné, donc
    # forcer hν-φ par défaut placerait EF au mauvais endroit.
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "ses_center_energy"
    if resolved_mode == "hv_minus_work_function":
        if ef_kin_from_hv is None:
            raise ValueError(
                "Mode BESSY hν-φ demandé mais hν est absent/non valide. "
                "Charge le logbook ou repasse en mode Auto."
            )
        ef_kin_nominal = float(ef_kin_from_hv)
        energy_reference = "hv_minus_work_function"
        hv_policy = "used_for_EF"
        loader_warnings.append(
            "Mode hν-φ forcé : EF placé via logbook. Vérifier que hν du logbook est correct pour ce fichier."
        )
    else:
        ef_kin_nominal = float(center_energy)
        energy_reference = "ses_center_energy"
        hv_policy = "stored_for_kz_not_used_for_EF"
    if is_binding_axis:
        # Axe SES déjà référencé à EF en convention Binding (positif = sous EF).
        # On convertit en E-EF (positif = au-dessus d'EF) par flip de signe ; on
        # ne soustrait rien. Pour kx il faut Ek réel : on utilise hν-φ si dispo,
        # sinon Center Energy comme proxy.
        energy = -energy_raw + float(ef_offset)
        energy_reference = "ses_binding_axis"
        ef_kin_for_kx = float(ef_kin_from_hv) if ef_kin_from_hv is not None else float(center_energy)
        kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_for_kx, a_lattice, polar + theta0_deg)
        loader_warnings.append(
            f"Échelle SES en Binding ({energy_scale_raw}) : axe converti en E-EF par flip de signe."
        )
        if ef_kin_from_hv is None:
            loader_warnings.append(
                "Binding axis sans hν : kx utilise Center Energy comme proxy pour Ek (imprécis)."
            )
    else:
        energy = energy_raw - ef_kin_nominal + float(ef_offset)
        kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_nominal, a_lattice, polar + theta0_deg)
    if center_energy_from_fallback:
        loader_warnings.append("Center Energy absent/invalide; E-EF estimé depuis le centre de l'axe énergie brut")
    if hv_val is None:
        loader_warnings.append("hν absent dans le fichier/logbook; conservé comme inconnu pour kz/comparaison hv")
    center_minus_hv_phi = float(center_energy - ef_kin_from_hv) if ef_kin_from_hv is not None else None
    if ef_kin_from_hv is not None and abs(float(center_energy) - ef_kin_from_hv) > 1.0:
        loader_warnings.append(
            f"hν-φ={ef_kin_from_hv:.3f} eV diffère de Center Energy={float(center_energy):.3f} eV; "
            f"référence énergie utilisée: {energy_reference}"
        )
    meta: dict[str, Any] = {
        "lab": "BESSY",
        "loader_label": "BESSY",
        "fs_source": "bessy_ses_ibw",
        "scan_kind": "BM" if raw.ndim == 2 else "FS",
        "ibw_wave_name": info.wave_name,
        "ibw_dims": info.dims,
        "ses_note": note,
        "ses": ses,
        "energy_axis_original": "binding" if is_binding_axis else "kinetic",
        "energy_scale_raw": energy_scale_raw,
        "energy_axis": "E-EF",
        "energy_reference": energy_reference,
        "bessy_energy_reference_mode": resolved_mode,
        "bessy_energy_reference_requested": mode,
        "energy_raw": energy_raw,
        "energy_raw_min": float(energy_raw[0]),
        "energy_raw_max": float(energy_raw[-1]),
        "center_energy": float(center_energy),
        "center_energy_from_fallback": bool(center_energy_from_fallback),
        "ef_kinetic_nominal": float(ef_kin_nominal),
        "ef_kinetic_nominal_from_hv": ef_kin_from_hv,
        "ef_kinetic_from_hv": ef_kin_from_hv,
        "center_minus_hv_phi": center_minus_hv_phi,
        "hv_policy": hv_policy,
        "theta_par_deg": theta,
        "x_axis_unit": "pi/a",
        "kx_unit": "pi/a",
        "kx_conversion": "theta_minus_p_axis_minus_theta0",
        "angle_offsets_applied": dict(angle_offsets),
        "theta0_deg": theta0_deg,
        "tilt0_deg": tilt0_deg,
        "hv": hv_val if hv_val is not None else np.nan,
        "temperature": temp_val if temp_val is not None else np.nan,
        "pol": pol,
        "azi": azi,
        "polar": polar,
        "lens_mode": ses.get("Lens Mode"),
        "pass_energy": ses.get("Pass Energy"),
        "acquisition_mode": ses.get("Acquisition Mode"),
        "number_of_sweeps": ses.get("Number of Sweeps"),
        "sample": ses.get("Sample"),
        "region_name": ses.get("Region Name"),
    }
    _add_instrument_resolution_metadata(meta, source=ses)
    ky = None
    if raw.ndim == 2:
        data = raw.T
        n_steps = 1
    elif raw.ndim == 3:
        fs_data = np.transpose(raw, (2, 1, 0))  # (P scan, theta/kx, E)
        data = np.nanmean(fs_data, axis=0)
        p_scan = np.asarray(ses.get("P-Axis scan", []), dtype=float)
        p_scan_from_note = p_scan.size == fs_data.shape[0]
        if p_scan.size != fs_data.shape[0]:
            p_scan = info.sf_b[2] + np.arange(fs_data.shape[0], dtype=float) * info.sf_a[2]
            loader_warnings.append("P-Axis scan absent/incomplet; axe ky reconstruit depuis l'échelle Igor")
        p_center = float(np.nanmean([np.nanmin(p_scan), np.nanmax(p_scan)]))
        if p_scan.size > 2:
            span = float(np.nanmax(p_scan) - np.nanmin(p_scan))
            midpoint = 0.5 * (float(np.nanmax(p_scan)) + float(np.nanmin(p_scan)))
            if span > 0 and abs(midpoint) > 0.25 * span:
                loader_warnings.append(
                    "P-Axis semble off-center; ky est recentré au milieu du scan, vérifier Γ/FS manuellement"
                )
        ky_offset = p_center + tilt0_deg
        ky = _cls_angle_to_k_pi_over_a(p_scan, ef_kin_nominal, a_lattice, ky_offset)
        n_steps = int(fs_data.shape[0])
        meta.update({
            "fs_data": fs_data,
            "fs_kx": kx,
            "fs_ky": ky,
            "fs_energy": energy,
            "fs_kind": "kxky",
            "fs_ky_angle_deg": p_scan,
            "fs_ky_angle_center_deg": p_center,
            "fs_ky_angle_from_note": bool(p_scan_from_note),
            "ky_conversion": "p_axis_scan_minus_scan_center_minus_tilt0",
        })
    else:
        raise ValueError(f"IBW BESSY avec dimension non supportée {raw.shape} dans {path.name}")
    meta.update({"n_steps": n_steps, "n_cycles": 1})
    _add_loader_diagnostics(
        meta,
        capability="BESSY Scienta/SES R8000 Igor Binary Wave v5",
        assumptions=[
            "SES energy axis is kinetic and locally referenced by Center Energy",
            "Auto/SES mode uses Center Energy to place E-EF",
            "hν-φ mode is an explicit diagnostic override",
            "kx uses theta - P-Axis - theta0",
            "FS ky uses P-Axis scan recentered on the scan midpoint",
        ],
        warnings_=loader_warnings,
        geometry_confidence="medium",
        axis_sources={
            "energy": "SES Center Energy",
            "kx": "IBW theta scale and SES P-Axis",
            "ky": "SES P-Axis scan for FS, recentered",
            "hv": "logbook/manual, then SES note fallback",
        },
    )
    ds = ARPESData(data=data, energy=energy, kx=kx, ky=ky, hv=hv_val, path=path,
                   source_format="bessy_ses_ibw", metadata=meta)
    return assert_arpes_data_valid(ds)

def _cls_cycle_step(path: Path) -> tuple[int, int]:
    m = _CYCLE_STEP_RE.search(path.name)
    if not m:
        raise ValueError(f"Nom CLS Cycle/Step invalide : {path.name}")
    return int(m.group(1)), int(m.group(2))

def _cls_fs_cache_path(folder: Path, prefix: str) -> Path:
    return folder / ".arpes_cache" / f"{prefix}_fs_mean_v{_CLS_CACHE_VERSION}.npz"

def _cls_fs_signature(param_file: Path, files: list[Path]) -> str:
    items = [{
        "name": f.name,
        "size": f.stat().st_size,
        "mtime_ns": f.stat().st_mtime_ns,
    } for f in files]
    if param_file.exists():
        items.append({
            "name": param_file.name,
            "size": param_file.stat().st_size,
            "mtime_ns": param_file.stat().st_mtime_ns,
        })
    return json.dumps({"version": _CLS_CACHE_VERSION, "files": items}, sort_keys=True)

def _load_cls_fs_cache(cache_path: Path, signature: str):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            if str(npz["signature"].item()) != signature:
                return None
            return (
                np.asarray(npz["volume"], dtype=np.float32),
                [int(x) for x in np.asarray(npz["step_ids"]).tolist()],
            )
    except Exception:
        return None

def _save_cls_fs_cache(cache_path: Path, signature: str, volume: np.ndarray, step_ids: list[int]) -> None:
    try:
        cache_path.parent.mkdir(exist_ok=True)
        np.savez(
            cache_path,
            signature=np.array(signature),
            volume=np.asarray(volume, dtype=np.float32),
            step_ids=np.asarray(step_ids, dtype=np.int32),
        )
    except Exception:
        pass

def _load_cls_fs_volume(folder: Path, prefix: str) -> tuple[np.ndarray, list[int], int]:
    all_files = sorted(folder.glob(f"{prefix}_Cycle_*_Step_*.txt"), key=_cls_cycle_step)
    if not all_files:
        raise FileNotFoundError(f"Aucun fichier Cycle/Step dans {folder}")

    steps: dict[int, list[Path]] = defaultdict(list)
    for f in all_files:
        _, step = _cls_cycle_step(f)
        steps[step].append(f)
    step_ids = sorted(steps)
    n_cycles = max(len(v) for v in steps.values())

    signature = _cls_fs_signature(folder / f"{prefix}_param.txt", all_files)
    cache_path = _cls_fs_cache_path(folder, prefix)
    cached = _load_cls_fs_cache(cache_path, signature)
    if cached is not None:
        volume, cached_step_ids = cached
        return volume, cached_step_ids, n_cycles

    first = _loadtxt_float32(steps[step_ids[0]][0])
    n_e, n_th = first.shape
    volume = np.zeros((len(step_ids), n_e, n_th), dtype=np.float32)
    volume[0] += first

    tasks: list[tuple[int, Path]] = []
    for i, sid in enumerate(step_ids):
        files = steps[sid]
        start = 1 if i == 0 else 0
        for f in files[start:]:
            tasks.append((i, f))

    max_workers = min(8, max(1, os.cpu_count() or 1), max(1, len(tasks)))
    if tasks:
        sums = [volume[i] for i in range(len(step_ids))]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, arr in pool.map(lambda item: (item[0], _loadtxt_float32(item[1])), tasks):
                if arr.shape != (n_e, n_th):
                    raise ValueError(f"Shape CLS incohérente dans {steps[step_ids[i]][0].parent}: {arr.shape} vs {(n_e, n_th)}")
                sums[i] += arr

    for i, sid in enumerate(step_ids):
        volume[i] /= float(len(steps[sid]))

    _save_cls_fs_cache(cache_path, signature, volume, step_ids)
    return volume, step_ids, n_cycles

def load_cls_txt(path, *, work_func: float = 4.031, ef_offset: float = 0.0,
                 a_lattice: float = 3.96, hv: float | None = None,
                 temperature: float | None = None, azi: float = 0.0, pol: str = "",
                 angle_offsets: dict | None = None,
                 fs_step_mode: str = "mean") -> ARPESData:
    path = Path(path)
    hv_val = float(hv) if hv is not None and np.isfinite(hv) and float(hv) > 0 else np.nan
    if not np.isfinite(hv_val):
        raise ValueError(
            "hν est obligatoire pour les données CLS/LNLS : entre l'énergie photon "
            "du logbook avant de charger le fichier."
        )
    temp_val = float(temperature) if temperature is not None and np.isfinite(temperature) else np.nan
    if path.is_file():
        folder = path.parent; prefix = path.name; p = _parse_cls_param(folder, prefix)
        raw = _loadtxt_float32(path); volume = None; tilt_coords = None
        scan_kind, n_steps, n_cycles = "BM", 1, 1
    else:
        folder = path; param_files = sorted(folder.glob("*_param.txt"))
        if not param_files: raise FileNotFoundError(f"Aucun *_param.txt dans {folder}")
        candidates = []
        for pf in param_files:
            cand = pf.name.removesuffix("_param.txt")
            if any(folder.glob(f"{cand}_Cycle_*_Step_*.txt")):
                candidates.append(cand)
        if not candidates:
            raise ValueError(
                f"{folder} contient des paramètres CLS mais aucun fichier Cycle/Step. "
                "Charge un fichier BM individuel, pas le dossier."
            )
        prefix = candidates[0]; p = _parse_cls_param(folder, prefix)
        volume, step_ids, n_cycles = _load_cls_fs_volume(folder, prefix)
        raw = volume.mean(axis=0) if fs_step_mode == "mean" else volume[len(step_ids)//2]
        phi_vals = p.get("phi_values")
        if phi_vals is not None:
            if all(0 <= s < len(phi_vals) for s in step_ids):
                tilt_coords = np.array([phi_vals[s] for s in step_ids], dtype=float)
            elif all(1 <= s <= len(phi_vals) for s in step_ids):
                tilt_coords = np.array([phi_vals[s - 1] for s in step_ids], dtype=float)
            else:
                tilt_coords = np.array(step_ids, dtype=float)
        else:
            tilt_coords = np.array(step_ids, dtype=float)
        scan_kind, n_steps = "FS", len(step_ids)
    n_e, n_theta = raw.shape
    energy_raw = p["energy_min"] + np.arange(n_e) * p["energy_delta"]
    ef_kin_nominal = hv_val - float(work_func)
    energy = energy_raw - ef_kin_nominal + float(ef_offset)
    theta = p["angle_min"] + np.arange(n_theta) * p["angle_delta"]
    angle_offsets = angle_offsets or {}
    theta0_deg = float(angle_offsets.get("theta0_deg", 0.0) or 0.0)
    tilt0_deg = float(angle_offsets.get("tilt0_deg", 0.0) or 0.0)
    angular_offset = float(p.get("polar", 0.0) or 0.0) + theta0_deg
    kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_nominal, a_lattice, angular_offset)
    data = raw.T
    meta: dict[str, Any] = {"lab": "CLS/LNLS", "scan_kind": scan_kind, "fs_source": "cls_txt",
        "loader_label": "CLS",
        "energy_axis_original": "kinetic", "energy_axis": "E-EF",
        "energy_reference": "hv_minus_work_function",
        "energy_raw": energy_raw,
        "energy_raw_min": float(energy_raw[0]), "energy_raw_max": float(energy_raw[-1]),
        "ef_kinetic_nominal_from_hv": float(ef_kin_nominal) if np.isfinite(ef_kin_nominal) else None,
        "ef_kinetic_from_hv": float(ef_kin_nominal) if np.isfinite(ef_kin_nominal) else None,
        "theta_par_deg": theta, "x_axis_unit": "pi/a", "kx_unit": "pi/a",
        "kx_conversion": "theta_minus_polar_minus_theta0",
        "polar_already_applied_to_kx": True,
        "angle_offsets_applied": dict(angle_offsets),
        "theta0_deg": theta0_deg, "tilt0_deg": tilt0_deg,
        "hv": hv_val if np.isfinite(hv_val) else None, "temperature": temp_val, "pol": pol, "azi": azi,
        "polar": float(p.get("polar",0.0)), "tilt_ref": float(p.get("tilt_ref",0.0)),
        "x": float(p.get("x",0.0)), "y": float(p.get("y",0.0)), "z": float(p.get("z",0.0)),
        "central_energy": p.get("central_energy"), "pass_energy": p.get("pass_energy"),
        "lens_mode": p.get("lens_mode"), "dwell_ms": p.get("dwell_ms"), "acquisition_mode": p.get("acquisition_mode"),
        "n_steps": n_steps, "n_cycles": n_cycles, "tilt_coords": tilt_coords}
    _add_instrument_resolution_metadata(meta, source=p)
    ky = None
    if volume is not None and tilt_coords is not None:
        fs_data = np.transpose(volume, (0, 2, 1))  # (tilt, theta/kx, E)
        tilt_angle = np.asarray(tilt_coords, dtype=float)
        ky = _cls_angle_to_k_pi_over_a(tilt_angle, ef_kin_nominal, a_lattice, tilt0_deg)
        meta.update({"fs_data": fs_data, "fs_kx": kx, "fs_ky": ky, "fs_energy": energy,
                     "fs_kind": "kxky", "fs_ky_angle_deg": tilt_angle,
                     "ky_conversion": "tilt_minus_tilt0"})
    _add_loader_diagnostics(
        meta,
        capability="CLS/LNLS text BM and Cycle/Step FS with matching *_param.txt",
        assumptions=[
            "hν comes from logbook/manual input and is required for E-EF",
            "CLS *_param.txt provides energy/angle scales and manipulator P/T when available",
            "kx uses theta - polar - theta0",
            "FS ky uses tilt/phi values from param file when present, otherwise step ids",
        ],
        warnings_=[] if p.get("phi_values") is not None or volume is None else [
            "FS tilt/phi values absent in *_param.txt; ky built from step ids"
        ],
        geometry_confidence="medium",
        axis_sources={
            "energy": "CLS text energy scale shifted by hν - work_function",
            "kx": "CLS angle scale and param P motor",
            "ky": "CLS phi_values/tilt steps for FS",
            "hv": "logbook/manual input",
        },
    )
    ds = ARPESData(data=data, energy=energy, kx=kx, ky=ky,
                   hv=hv_val if np.isfinite(hv_val) else None, path=path,
                   source_format="cls_txt", metadata=meta)
    return assert_arpes_data_valid(ds)

def load_arpes(path, *, work_func: float, ef_offset: float = 0.0, a_lattice: float = 3.96,
               format_hint: str | None = None, hv: float | None = None,
               temperature: float | None = None, azi: float = 0.0, pol: str = "",
               angle_offsets: dict | None = None,
               bessy_energy_reference: str = "auto") -> ARPESData:
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

def _load_solaris_from_registry(path, *, work_func: float, ef_offset: float = 0.0,
                                a_lattice: float = 3.96, **_: Any) -> ARPESData:
    return load_solaris_da30_bandmap(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice)

def _load_cls_from_registry(path, *, work_func: float, ef_offset: float = 0.0,
                            a_lattice: float = 3.96, hv: float | None = None,
                            temperature: float | None = None, azi: float = 0.0,
                            pol: str = "", angle_offsets: dict | None = None,
                            **_: Any) -> ARPESData:
    return load_cls_txt(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice,
                        hv=hv, temperature=temperature, azi=azi, pol=pol,
                        angle_offsets=angle_offsets)

register_loader("bessy_ses_ibw", _is_bessy_ses_ibw, load_bessy_ses_ibw,
                "BESSY Scienta/SES Igor Binary Wave")
register_loader("solaris_da30", _is_solaris_da30_file, _load_solaris_from_registry,
                "Solaris/DA30 via erlab")
register_loader("cls_txt", lambda p: _is_cls_fs_dir(p) or _is_cls_bm_file(p),
                _load_cls_from_registry, "CLS/LNLS texte + *_param.txt")
