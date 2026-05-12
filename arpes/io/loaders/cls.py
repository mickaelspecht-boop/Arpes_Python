"""Loader CLS/LNLS texte (BM individuel + FS Cycle/Step)."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ARPESData,
    _add_instrument_resolution_metadata,
    _add_loader_diagnostics,
    _cls_angle_to_k_pi_over_a,
    _loadtxt_float32,
    assert_arpes_data_valid,
    register_loader,
)


_CLS_CACHE_VERSION = 2
_CYCLE_STEP_RE = re.compile(r"Cycle_(\d+)_Step_(\d+)")


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
    ef_kin_from_hv = hv_val - float(work_func)
    # --- garde-fou cohérence hν ----------------------------------------------
    # L'échelle d'énergie CLS est cinétique (~ Central Energy). Pour une carte
    # BM, la fenêtre est ~centrée sur EF, donc EF_kin ≈ hν - φ ≈ centre fenêtre.
    # Si le hν fourni place EF_kin très loin de la fenêtre du fichier, hν est
    # presque sûrement faux (mauvaise colonne/ligne du logbook). On bascule
    # alors sur Central Energy comme référence EF et on prévient bruyamment.
    central = p.get("central_energy")
    win_lo, win_hi = float(energy_raw[0]), float(energy_raw[-1])
    win_mid = 0.5 * (win_lo + win_hi)
    hv_gap = abs(ef_kin_from_hv - win_mid)
    hv_warning = None
    energy_reference = "hv_minus_work_function"
    ef_kin_nominal = ef_kin_from_hv
    if hv_gap > 15.0 and central is not None and np.isfinite(float(central)):
        ef_kin_nominal = float(central)
        energy_reference = "central_energy_assumed_EF"
        hv_implied = float(central) + float(work_func)
        hv_warning = (
            f"hν={hv_val:g} eV ⇒ EF_kin={ef_kin_from_hv:g} eV, incohérent avec la "
            f"fenêtre d'énergie du fichier [{win_lo:g}, {win_hi:g}] eV (Central Energy "
            f"{float(central):g}). Référencement basé sur Central Energy (fenêtre supposée "
            f"centrée sur EF). hν réelle probablement ≈ {hv_implied:g} eV — corrige le logbook."
        )
    elif hv_gap > 15.0:
        hv_warning = (
            f"hν={hv_val:g} eV semble incohérent avec la fenêtre d'énergie du fichier "
            f"[{win_lo:g}, {win_hi:g}] eV — vérifie le logbook (carte probablement hors EF)."
        )
    elif hv_gap > 3.0:
        hv_warning = (
            f"hν={hv_val:g} eV ⇒ EF_kin={ef_kin_from_hv:g} eV, à {hv_gap:.1f} eV du centre "
            f"de la fenêtre [{win_lo:g}, {win_hi:g}] eV — fenêtre pas centrée sur EF "
            f"(offset volontaire ou hν imprécis ?). E−EF affiché tel quel."
        )
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
        "energy_reference": energy_reference,
        "hv_warning": hv_warning,
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
        warnings_=(
            ([] if p.get("phi_values") is not None or volume is None else [
                "FS tilt/phi values absent in *_param.txt; ky built from step ids"])
            + ([hv_warning] if hv_warning else [])
        ),
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


def _load_cls_from_registry(path, *, work_func: float, ef_offset: float = 0.0,
                            a_lattice: float = 3.96, hv: float | None = None,
                            temperature: float | None = None, azi: float = 0.0,
                            pol: str = "", angle_offsets: dict | None = None,
                            **_: Any) -> ARPESData:
    return load_cls_txt(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice,
                        hv=hv, temperature=temperature, azi=azi, pol=pol,
                        angle_offsets=angle_offsets)


register_loader("cls_txt", lambda p: _is_cls_fs_dir(p) or _is_cls_bm_file(p),
                _load_cls_from_registry, "CLS/LNLS texte + *_param.txt")
