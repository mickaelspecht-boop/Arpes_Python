"""Metadata and summary helpers for distortion calibration."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def angle_offsets_hash(angle_offsets: dict | None) -> str:
    """Stable hash of angular offsets."""
    payload = {} if not angle_offsets else {
        k: round(float(v), 6) if isinstance(v, (int, float, np.floating)) else str(v)
        for k, v in angle_offsets.items()
        if k in ("theta0_deg", "tilt0_deg", "azi", "polar_already_applied_to_kx")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def gamma_shift_signature(meta: dict | None) -> dict:
    """Snapshot of the gamma shift at calibration time."""
    if not meta:
        return {}
    return {
        "bm_gamma_axis_centered": bool(meta.get("bm_gamma_axis_centered", False)),
        "bm_gamma_axis_shift": float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0),
        "fs_gamma_axis_shift_kx": float(meta.get("fs_gamma_axis_shift_kx", 0.0) or 0.0),
    }


def calib_key_for_meta(meta: dict | None) -> tuple:
    """Calibration key shared by lens mode / E_pass / hν."""
    if not meta:
        return ("?", "?", "?")
    lens = str(meta.get("lens_mode") or "?")
    epass = meta.get("pass_energy")
    epass_key = f"{float(epass):.1f}" if epass is not None else "?"
    hv = meta.get("hv")
    hv_key = f"{float(hv):.1f}" if hv is not None else "?"
    return (lens, epass_key, hv_key)


def is_fs_data(meta: dict | None) -> bool:
    """True if meta carries an FS volume."""
    return bool(meta and meta.get("fs_data") is not None)


def get_cfg_summary(cfg: dict | None, *, is_active) -> str:
    """Short summary for status bar / UI label."""
    if not is_active(cfg):
        return "BM distortion: disabled"
    bits: list[str] = []
    trap = cfg.get("trapezoid") or {}
    para = cfg.get("parabola") or {}
    if trap.get("enabled"):
        sl = float(trap.get("slope_left", 0.0) or 0.0)
        sr = float(trap.get("slope_right", 0.0) or 0.0)
        bits.append(f"trapezoid L={sl:+.3f} R={sr:+.3f}")
    if para.get("enabled"):
        a = float(para.get("a", 0.0) or 0.0)
        k0 = float(para.get("k0", 0.0) or 0.0)
        bits.append(f"parabola a={a:+.3f} k0={k0:+.3f}")
    return "BM distortion active: " + " | ".join(bits)
