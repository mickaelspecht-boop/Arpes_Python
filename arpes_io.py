#!/usr/bin/env python3
"""Couche IO commune pour données ARPES: Solaris/DA30 + CLS/LNLS txt."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json, os, re, warnings
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

SUPPORTED_SOLARIS_EXTENSIONS = {".ibw", ".pxt", ".zip"}
_C_ARPES = 0.51233
_CLS_CACHE_VERSION = 2
_CYCLE_STEP_RE = re.compile(r"Cycle_(\d+)_Step_(\d+)")

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

def detect_format(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() in SUPPORTED_SOLARIS_EXTENSIONS: return "solaris_da30"
    if _is_cls_fs_dir(p) or _is_cls_bm_file(p): return "cls_txt"
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
    if not np.isfinite(hv): raise ValueError(f"hv absent dans {path.name}")
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
        kx = np.asarray(da_out.kx.values, dtype=float)
    if "ky" in da_out.coords:
        da_out = da_out.assign_coords(ky=da_out.ky * a_lattice / np.pi)
        ky = np.asarray(da_out.ky.values, dtype=float)
    energy = np.asarray(da_out.eV.values, dtype=float)
    arr = np.asarray(da_out.values, dtype=float).squeeze()
    dims = list(getattr(da_out, "dims", []))
    meta = dict(da.attrs); meta.update({"x_axis_unit": "pi/a", "kx_unit": "pi/a", "fs_source": "solaris_da30"})
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
    return ARPESData(data=data, energy=energy, kx=kx, ky=ky, hv=hv, path=path,
                     source_format="solaris_da30", metadata=meta, xarray=da_out)

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
    angular_offset = float(p.get("polar", 0.0) or 0.0)
    kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_nominal, a_lattice, angular_offset)
    data = raw.T
    meta: dict[str, Any] = {"lab": "CLS/LNLS", "scan_kind": scan_kind, "fs_source": "cls_txt",
        "energy_axis_original": "kinetic", "energy_axis": "E-EF",
        "energy_raw": energy_raw,
        "energy_raw_min": float(energy_raw[0]), "energy_raw_max": float(energy_raw[-1]),
        "ef_kinetic_nominal_from_hv": float(ef_kin_nominal) if np.isfinite(ef_kin_nominal) else None,
        "theta_par_deg": theta, "x_axis_unit": "pi/a", "kx_unit": "pi/a",
        "kx_conversion": "theta_minus_polar",
        "polar_already_applied_to_kx": True,
        "hv": hv_val if np.isfinite(hv_val) else None, "temperature": temp_val, "pol": pol, "azi": azi,
        "polar": float(p.get("polar",0.0)), "tilt_ref": float(p.get("tilt_ref",0.0)),
        "x": float(p.get("x",0.0)), "y": float(p.get("y",0.0)), "z": float(p.get("z",0.0)),
        "central_energy": p.get("central_energy"), "pass_energy": p.get("pass_energy"),
        "lens_mode": p.get("lens_mode"), "dwell_ms": p.get("dwell_ms"), "acquisition_mode": p.get("acquisition_mode"),
        "n_steps": n_steps, "n_cycles": n_cycles, "tilt_coords": tilt_coords}
    ky = None
    if volume is not None and tilt_coords is not None:
        fs_data = np.transpose(volume, (0, 2, 1))  # (tilt, theta/kx, E)
        tilt_angle = np.asarray(tilt_coords, dtype=float)
        ky = _cls_angle_to_k_pi_over_a(tilt_angle, ef_kin_nominal, a_lattice, 0.0)
        meta.update({"fs_data": fs_data, "fs_kx": kx, "fs_ky": ky, "fs_energy": energy,
                     "fs_kind": "kxky", "fs_ky_angle_deg": tilt_angle,
                     "ky_conversion": "small_angle_tilt"})
    return ARPESData(data=data, energy=energy, kx=kx, ky=ky,
                     hv=hv_val if np.isfinite(hv_val) else None, path=path,
                     source_format="cls_txt", metadata=meta)

def load_arpes(path, *, work_func: float, ef_offset: float = 0.0, a_lattice: float = 3.96,
               format_hint: str | None = None, hv: float | None = None,
               temperature: float | None = None, azi: float = 0.0, pol: str = "") -> ARPESData:
    fmt = format_hint or detect_format(path)
    if fmt == "solaris_da30":
        return load_solaris_da30_bandmap(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice)
    if fmt == "cls_txt":
        return load_cls_txt(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice,
                            hv=hv, temperature=temperature, azi=azi, pol=pol)
    raise ValueError(f"Format non supporté pour {Path(path).name!r}: {fmt}. Formats: Solaris/DA30 + CLS/LNLS txt.")
