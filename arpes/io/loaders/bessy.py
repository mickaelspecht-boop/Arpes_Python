"""Loader BESSY Scienta/SES R8000 (Igor Binary Wave v5)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ARPESData,
    _add_instrument_resolution_metadata,
    _add_loader_diagnostics,
    _cls_angle_to_k_pi_over_a,
    scan_axis_summary,
    static_polar_for_kx,
    _valid_positive_float,
    assert_arpes_data_valid,
    register_loader,
)


_IBW5_BIN_HEADER_SIZE = 64
_IBW5_WAVE_HEADER_SIZE = 320


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
        raise ValueError(f"IBW too short: {path.name}")
    if int.from_bytes(header[:2], "little") != 5:
        raise ValueError(f"Only IBW v5 files are supported for BESSY: {path.name}")
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
        raise ValueError(f"Unsupported BESSY IBW type ({wave_type}) in {path.name}")
    if not dims or int(np.prod(dims)) != npnts:
        raise ValueError(f"Inconsistent IBW dimensions in {path.name}: dims={dims}, npnts={npnts}")
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
        raise ValueError(f"Incomplete IBW read in {path.name}: {arr.size}/{info.npnts}")
    return arr.reshape(info.dims, order="F")


def _is_bessy_ses_ibw(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".ibw":
        return False
    try:
        info = _read_ibw5_info(path)
        note = _read_ibw5_note(path, info).encode("latin1", errors="ignore")
    except (OSError, ValueError):
        return False
    # `Instrument=R8000` is the unambiguous marker for the BESSY R8000 (the
    # tested Ba122 exports have `Instrument=R8000-8ES202`). Require this
    # signature in addition to `[SES]` to avoid catching DA30 exports that also
    # contain `Energy Scale=Kinetic`.
    return b"[SES]" in note and b"Instrument=R8000" in note


def load_bessy_ses_ibw(path, *, work_func: float = 0.0, ef_offset: float = 0.0,
                       a_lattice: float = 0.0, hv: float | None = None,
                       temperature: float | None = None, azi: float = 0.0,
                       pol: str = "", angle_offsets: dict | None = None,
                       bessy_energy_reference: str = "auto") -> ARPESData:
    """Load BESSY SES/R8000 Igor Binary Wave exports.

    The tested files contain Igor axes `(E_kin, theta[, P])` and a `@[SES]`
    note. The photon energy is not reliable in the note (`0` in the Ba122
    files), so `hv` must be provided by the user/logbook.
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
    polar_raw = float(ses.get("P-Axis", 0.0) or 0.0)
    raw = _load_ibw5_numeric(path, info).astype(np.float32, copy=False)
    n_e, n_theta = info.dims[0], info.dims[1]
    energy_raw = info.sf_b[0] + np.arange(n_e, dtype=float) * info.sf_a[0]
    theta = info.sf_b[1] + np.arange(n_theta, dtype=float) * info.sf_a[1]
    # Detect the SES energy scale: Kinetic (standard case) vs Binding. On
    # Binding, the axis is already referenced to EF on the SES side, so nothing
    # is subtracted and the sign is flipped to follow the E−EF convention
    # (positive above EF). kx needs the real Ek, so hν is required.
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
    # In the tested BESSY/SES R8000 exports (Ba122), Excitation/Monochromator
    # Energy = 0 in the note, so hν must come from the logbook or be passed
    # explicitly. Center Energy is normally the center of the recorded kinetic
    # window: reliable as an EF proxy ONLY if the operator actually centered
    # the BM on EF. Otherwise (for example, a BM at 30 eV binding), Center
    # Energy puts EF several eV away from zero and breaks calibration.
    ef_kin_from_hv = float(hv_val - work_func) if hv_val is not None else None
    loader_warnings: list[str] = []
    # Auto mode = Center Energy (the analyzer setting reflects the
    # experimenter's intent). hν-φ remains an explicit override because on old
    # BESSY files (for example Ba122), the logbook hν is often wrong, so using
    # hν-φ by default would place EF incorrectly.
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "ses_center_energy"
    if resolved_mode == "hv_minus_work_function":
        if ef_kin_from_hv is None:
            raise ValueError(
                "BESSY hν-φ mode was requested, but hν is missing/invalid. "
                "Load the logbook or switch back to Auto mode."
            )
        ef_kin_nominal = float(ef_kin_from_hv)
        energy_reference = "hv_minus_work_function"
        hv_policy = "used_for_EF"
        loader_warnings.append(
            "Forced hν-φ mode: EF placed from the logbook. Check that the logbook hν is correct for this file."
        )
    else:
        ef_kin_nominal = float(center_energy)
        energy_reference = "ses_center_energy"
        hv_policy = "stored_for_kz_not_used_for_EF"
    if is_binding_axis:
        # SES axis already referenced to EF in Binding convention (positive =
        # below EF). Convert to E-EF (positive = above EF) by flipping the sign;
        # subtract nothing. kx needs real Ek: use hν-φ if available, otherwise
        # Center Energy as a proxy.
        energy = -energy_raw + float(ef_offset)
        energy_reference = "ses_binding_axis"
        ef_kin_for_kx = float(ef_kin_from_hv) if ef_kin_from_hv is not None else float(center_energy)
        p_scan_for_polar = ses.get("P-Axis scan") if raw.ndim == 3 else None
        polar, polar_raw, ignored_scan_polar = static_polar_for_kx(
            polar_raw, p_scan_for_polar,
            is_fs=raw.ndim == 3,
            motor_present="P-Axis" in ses,
        )
        kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_for_kx, a_lattice, polar + theta0_deg)
        loader_warnings.append(
            f"SES Binding scale ({energy_scale_raw}): axis converted to E-EF by sign flip."
        )
        if ef_kin_from_hv is None:
            loader_warnings.append(
                "Binding axis without hν: kx uses Center Energy as an Ek proxy (imprecise)."
            )
    else:
        energy = energy_raw - ef_kin_nominal + float(ef_offset)
        p_scan_for_polar = ses.get("P-Axis scan") if raw.ndim == 3 else None
        polar, polar_raw, ignored_scan_polar = static_polar_for_kx(
            polar_raw, p_scan_for_polar,
            is_fs=raw.ndim == 3,
            motor_present="P-Axis" in ses,
        )
        kx = _cls_angle_to_k_pi_over_a(theta, ef_kin_nominal, a_lattice, polar + theta0_deg)
    kx_axis_midpoint = float(0.5 * (np.nanmin(kx) + np.nanmax(kx))) if kx.size else np.nan
    kx_center_index = float(kx[len(kx) // 2]) if kx.size else np.nan
    if center_energy_from_fallback:
        loader_warnings.append("Center Energy missing/invalid; E-EF estimated from the raw energy-axis center")
    if hv_val is None:
        loader_warnings.append("hν missing in the file/logbook; kept as unknown for kz/hv comparison")
    if ignored_scan_polar:
        loader_warnings.append("P-Axis motor position matches the FS scan axis; ignored as static kx polar")
    center_minus_hv_phi = float(center_energy - ef_kin_from_hv) if ef_kin_from_hv is not None else None
    if ef_kin_from_hv is not None and abs(float(center_energy) - ef_kin_from_hv) > 1.0:
        loader_warnings.append(
            f"hν-φ={ef_kin_from_hv:.3f} eV differs from Center Energy={float(center_energy):.3f} eV; "
            f"energy reference used: {energy_reference}"
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
        "polar_raw_motor": polar_raw,
        "fs_scan_polar_ignored_for_kx": bool(ignored_scan_polar),
        "fs_static_polar_policy": "ignore_scanned_polar_for_fs_kx",
        "kx_axis_midpoint": kx_axis_midpoint,
        "kx_center_index": kx_center_index,
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
            loader_warnings.append("P-Axis scan missing/incomplete; ky axis rebuilt from the Igor scale")
        p_center = float(np.nanmean([np.nanmin(p_scan), np.nanmax(p_scan)]))
        p_axis_offcenter = None
        if p_scan.size > 2:
            span = float(np.nanmax(p_scan) - np.nanmin(p_scan))
            midpoint = 0.5 * (float(np.nanmax(p_scan)) + float(np.nanmin(p_scan)))
            if span > 0 and abs(midpoint) > 0.25 * span:
                loader_warnings.append(
                    "P-Axis appears off-center; ky is recentered on the scan midpoint, check Γ/FS manually"
                )
                # P4.5: structured flag -> the UI opens a confirmation dialog
                # before accepting the recentering (io cannot import PyQt).
                p_axis_offcenter = {
                    "span_deg": span,
                    "midpoint_deg": midpoint,
                    "recentered_to_deg": p_center,
                }
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
            "fs_scan_axis_deg": scan_axis_summary(p_scan),
            "fs_ky_angle_center_deg": p_center,
            "fs_ky_angle_from_note": bool(p_scan_from_note),
            "fs_p_axis_offcenter": p_axis_offcenter,
            "ky_conversion": "p_axis_scan_minus_scan_center_minus_tilt0",
        })
    else:
        raise ValueError(f"BESSY IBW with unsupported dimension {raw.shape} in {path.name}")
    meta.update({"n_steps": n_steps, "n_cycles": 1})
    _add_loader_diagnostics(
        meta,
        capability="BESSY Scienta/SES R8000 Igor Binary Wave v5",
        assumptions=[
            "SES energy axis is kinetic and locally referenced by Center Energy",
            "Auto/SES mode uses Center Energy to place E-EF",
            "hν-φ mode is an explicit diagnostic override",
            "kx uses theta - static P-Axis - theta0; FS scanned P-Axis is not reused as static polar",
            "FS ky uses P-Axis scan recentered on the scan midpoint",
        ],
        warnings_=loader_warnings,
        geometry_confidence="medium",
        axis_sources={
            "energy": "SES Center Energy",
            "kx": "IBW theta scale and static SES P-Axis, with FS scanned P-Axis ignored when it is the loop coordinate",
            "ky": "SES P-Axis scan for FS, recentered",
            "hv": "logbook/manual, then SES note fallback",
        },
    )
    ds = ARPESData(data=data, energy=energy, kx=kx, ky=ky, hv=hv_val, path=path,
                   source_format="bessy_ses_ibw", metadata=meta)
    return assert_arpes_data_valid(ds)


register_loader("bessy_ses_ibw", _is_bessy_ses_ibw, load_bessy_ses_ibw,
                "BESSY Scienta/SES Igor Binary Wave")
