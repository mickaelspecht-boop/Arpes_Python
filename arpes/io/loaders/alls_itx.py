"""Loader ALLS SpecsLab Prodigy Igor Text exports."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ARPESData,
    _add_instrument_resolution_metadata,
    _add_loader_diagnostics,
    _cls_angle_to_k_pi_over_a,
    _valid_float,
    _valid_positive_float,
    assert_arpes_data_valid,
    register_loader,
    scan_axis_summary,
)


_WAVES_RE = re.compile(r"WAVES/[^\n]*?/N=\(([^)]+)\)\s+'?([^'\n]+)'?")
_SET_SCALE_RE = re.compile(
    r"SetScale/I\s+([xyzd]),\s*([^,]+),\s*([^,]+),\s*\"([^\"]*)\"",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r"^X\s+//([^=]+?)\s*=\s*(.*?)\s*$")


@dataclass(frozen=True)
class ITXScale:
    axis: str
    start: float
    stop: float
    label: str


@dataclass(frozen=True)
class ITXInfo:
    dims: tuple[int, ...]
    wave_name: str
    header: dict[str, Any]
    scales: dict[str, ITXScale]


def _coerce_header_value(value: str) -> Any:
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        return text


def _axis_values(scale: ITXScale, n: int) -> np.ndarray:
    if n <= 1:
        return np.asarray([scale.start], dtype=float)
    return np.linspace(scale.start, scale.stop, n, dtype=float)


def _parse_alls_itx_info(text: str) -> ITXInfo:
    if not text.startswith("IGOR"):
        raise ValueError("Not an Igor Text file")
    wave_match = _WAVES_RE.search(text)
    if wave_match is None:
        raise ValueError("ITX WAVES/N declaration not found")
    dims = tuple(int(part.strip()) for part in wave_match.group(1).split(",") if part.strip())
    if not dims:
        raise ValueError("ITX WAVES/N declaration has no dimensions")
    wave_name = wave_match.group(2).strip()

    header: dict[str, Any] = {}
    for line in text[: wave_match.start()].splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            header[m.group(1).strip()] = _coerce_header_value(m.group(2))

    scales: dict[str, ITXScale] = {}
    for m in _SET_SCALE_RE.finditer(text):
        axis = m.group(1).lower()
        start = _valid_float(m.group(2))
        stop = _valid_float(m.group(3))
        if start is None or stop is None:
            continue
        scales[axis] = ITXScale(axis=axis, start=start, stop=stop, label=m.group(4).strip())
    return ITXInfo(dims=dims, wave_name=wave_name, header=header, scales=scales)


def _read_alls_itx_info(path: Path) -> ITXInfo:
    return _parse_alls_itx_info(path.read_text(errors="replace"))


def _is_alls_itx_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".itx":
        return False
    try:
        with path.open("r", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return False
    return (
        head.startswith("IGOR")
        and "Created by: SpecsLab Prodigy" in head
        and "WAVES/" in head
    )


def _load_alls_itx_array(text: str, dims: tuple[int, ...], path: Path) -> np.ndarray:
    begin = text.find("\nBEGIN")
    if begin < 0:
        raise ValueError(f"ITX BEGIN block not found: {path.name}")
    begin = text.find("\n", begin + 1)
    end = text.find("\nEND", begin)
    if begin < 0 or end < 0:
        raise ValueError(f"ITX END block not found: {path.name}")
    values = np.fromstring(text[begin:end], sep=" ", dtype=np.float32)
    expected = int(np.prod(dims))
    if values.size != expected:
        raise ValueError(
            f"ITX data size mismatch in {path.name}: read {values.size}, expected {expected}"
        )
    return values.reshape(dims, order="F")


def _scale_or_default(info: ITXInfo, axis: str, n: int) -> tuple[np.ndarray, str]:
    scale = info.scales.get(axis)
    if scale is None:
        return np.arange(n, dtype=float), "index"
    return _axis_values(scale, n), scale.label


def _label_lower(info: ITXInfo, axis: str) -> str:
    scale = info.scales.get(axis)
    return (scale.label if scale is not None else "").strip().lower()


def _scan_kind_from_info(info: ITXInfo) -> str:
    labels = " ".join(_label_lower(info, axis) for axis in ("x", "y", "z"))
    if "delay" in labels or "loop" in labels:
        return "TRM"
    if "shiftx" in labels or "shift x" in labels:
        return "FS"
    return "BM" if len(info.dims) == 2 else "unknown"


def load_alls_itx(path, *, work_func: float = 0.0, ef_offset: float = 0.0,
                  a_lattice: float = 0.0, hv: float | None = None,
                  temperature: float | None = None, azi: float = 0.0,
                  pol: str = "", angle_offsets: dict | None = None,
                  **_unused) -> ARPESData:
    path = Path(path)
    text = path.read_text(errors="replace")
    info = _parse_alls_itx_info(text)
    raw = _load_alls_itx_array(text, info.dims, path)

    hv_val = _valid_positive_float(info.header.get("Excitation Energy"))
    if hv_val is None:
        hv_val = _valid_positive_float(hv)
    if hv_val is None:
        raise ValueError(
            f"hν missing or invalid in ALLS ITX file {path.name}. "
            "Enter photon energy manually before loading."
        )

    temp_val = _valid_positive_float(temperature)
    theta, theta_label = _scale_or_default(info, "x", raw.shape[0])
    kinetic, energy_label = _scale_or_default(info, "y", raw.shape[1])
    if "theta" not in theta_label.lower() and "angle" not in theta_label.lower():
        raise ValueError(
            f"Unsupported ALLS ITX axes in {path.name}: x axis is {theta_label!r}, "
            "not analyzer theta. This time/loop scan is not an ARPES band map."
        )
    if "kinetic" not in energy_label.lower() and "ev" not in energy_label.lower():
        raise ValueError(
            f"Unsupported ALLS ITX axes in {path.name}: y axis is {energy_label!r}, "
            "not an energy axis."
        )
    scan_axis = None
    scan_label = ""
    if raw.ndim == 3:
        scan_axis, scan_label = _scale_or_default(info, "z", raw.shape[2])

    energy = kinetic - float(hv_val) + float(work_func) + float(ef_offset)
    angle_offsets = angle_offsets or {}
    theta0_deg = float(angle_offsets.get("theta0_deg", 0.0) or 0.0)
    a_for_k = float(a_lattice) if float(a_lattice or 0.0) > 0 else 1.0
    kx = _cls_angle_to_k_pi_over_a(theta, float(np.nanmean(kinetic)), a_for_k, theta0_deg)

    loader_warnings: list[str] = []
    if float(a_lattice or 0.0) <= 0:
        loader_warnings.append(
            "a_lattice missing or non-positive; kx uses a=1 fallback and must be recalibrated."
        )
    if raw.ndim == 3 and scan_label and "deg" not in scan_label.lower():
        loader_warnings.append(
            f"Third ITX axis is {scan_label!r}; stored as raw scan axis, not calibrated ky."
        )
    if "kinetic" not in energy_label.lower():
        loader_warnings.append(
            f"Energy axis label is {energy_label!r}; loader still applies kinetic-hν+φ convention."
        )
    if _valid_float(info.header.get("WorkFunction")) == 0 and float(work_func or 0.0) == 0.0:
        loader_warnings.append(
            "Work function is 0 in the ITX header and no manual value was provided; E-EF is approximate."
        )
    kinetic_center = _valid_float(info.header.get("Kinetic Energy"))
    if kinetic_center is not None and (kinetic_center < np.nanmin(kinetic) or kinetic_center > np.nanmax(kinetic)):
        loader_warnings.append(
            "Header Kinetic Energy is outside the SetScale energy range; using SetScale y as the energy axis."
        )
    if raw.size and np.mean(raw == 0) > 0.5:
        loader_warnings.append("More than 50% of intensity values are zero; dataset may be sparse or partially empty.")

    if raw.ndim == 2:
        data = np.asarray(raw, dtype=np.float32)
        n_steps = 1
    elif raw.ndim == 3:
        fs_data = np.transpose(raw, (2, 0, 1))  # ITX (theta, E, scan) -> (scan, kx, E)
        data = np.nanmean(fs_data, axis=0)
        n_steps = int(fs_data.shape[0])
    else:
        raise ValueError(f"ALLS ITX with unsupported dimension {raw.shape} in {path.name}")

    meta: dict[str, Any] = {
        "lab": "ALLS",
        "loader_label": "ALLS",
        "source_format": "alls_itx",
        "specslab_wave_name": info.wave_name,
        "igor_wave_name": info.wave_name,
        "igor_dims": tuple(int(x) for x in info.dims),
        "specslab_header": dict(info.header),
        "fs_source": "alls_itx",
        "scan_kind": _scan_kind_from_info(info),
        "raw_axes": {
            "x": {"values": theta, "label": theta_label},
            "y": {"values": kinetic, "label": energy_label},
        },
        "theta_y_deg": theta,
        "theta_par_deg": theta,
        "kinetic_energy_eV": kinetic,
        "energy_raw": kinetic,
        "energy_raw_unit": "eV",
        "energy_raw_min": float(np.nanmin(kinetic)),
        "energy_raw_max": float(np.nanmax(kinetic)),
        "kinetic_energy_center_eV": kinetic_center,
        "binding_energy_header_eV": _valid_float(info.header.get("Binding Energy")),
        "hv": hv_val,
        "hv_source": "itx_header" if _valid_positive_float(info.header.get("Excitation Energy")) is not None else "manual",
        "temperature": temp_val if temp_val is not None else np.nan,
        "pol": pol,
        "azi": azi,
        "x_axis_unit": "pi/a",
        "kx_unit": "pi/a",
        "kx_conversion": "theta_y_minus_theta0",
        "angle_offsets_applied": dict(angle_offsets),
        "theta0_deg": theta0_deg,
        "energy_axis": "E-EF",
        "energy_axis_original": "kinetic",
        "energy_reference": "kinetic_minus_hv_plus_work_function",
        "work_func_eV": float(work_func or 0.0),
        "work_function_eV": float(work_func or 0.0),
        "work_function_source": "manual" if float(work_func or 0.0) > 0 else "missing",
        "intensity_unit": info.scales.get("d").label if info.scales.get("d") else "",
        "intensity_min": float(np.nanmin(raw)) if raw.size else np.nan,
        "intensity_max": float(np.nanmax(raw)) if raw.size else np.nan,
        "intensity_zero_fraction": float(np.mean(raw == 0)) if raw.size else np.nan,
        "n_steps": n_steps,
        "n_cycles": 1,
    }
    if scan_axis is not None:
        meta["raw_axes"]["z"] = {"values": scan_axis, "label": scan_label}
        meta.update({
            "fs_data": fs_data,
            "fs_kx": kx,
            "fs_ky": scan_axis,
            "fs_energy": energy,
            "fs_kind": "scan-kx-energy",
            "fs_scan_axis": scan_axis,
            "fs_scan_axis_label": scan_label,
            "fs_scan_axis_summary": scan_axis_summary(scan_axis),
            "shiftx_raw": scan_axis if "shiftx" in scan_label.lower() else None,
            "shiftx_unit": scan_label if "shiftx" in scan_label.lower() else "",
            "ky_conversion": "raw_itx_scan_axis_not_calibrated",
        })

    _add_instrument_resolution_metadata(meta, source=info.header)
    _add_loader_diagnostics(
        meta,
        capability="ALLS SpecsLab Prodigy Igor Text",
        assumptions=[
            "ITX x axis is analyzer theta_y and maps to kx",
            "ITX y axis is kinetic energy and is converted to E-EF by kinetic-hν+work_function",
            "3D ITX z axis is kept as a raw scan coordinate unless its unit is later calibrated",
            "Displayed 2D data is the mean over the third ITX axis when a 3D volume is present",
        ],
        warnings_=loader_warnings,
        geometry_confidence="low" if scan_axis is not None else "medium",
        axis_sources={
            "energy": "ITX SetScale y + header Excitation Energy + manual work function",
            "kx": "ITX SetScale x theta_y",
            "ky": "raw ITX SetScale z scan axis, not calibrated",
            "hv": "ITX Excitation Energy, then manual fallback",
        },
    )
    ds = ARPESData(data=data, energy=energy, kx=kx, ky=None, hv=hv_val, path=path,
                   source_format="alls_itx", metadata=meta)
    return assert_arpes_data_valid(ds)


register_loader("alls_itx", _is_alls_itx_file, load_alls_itx,
                "ALLS SpecsLab Prodigy Igor Text")
