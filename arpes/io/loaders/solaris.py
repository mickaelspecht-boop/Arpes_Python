"""Loader Solaris/DA30 (via erlab)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ARPESData,
    SUPPORTED_SOLARIS_EXTENSIONS,
    _add_instrument_resolution_metadata,
    _add_loader_diagnostics,
    _require_erlab,
    _set_da30_loader,
    _transpose_to_axes,
    assert_arpes_data_valid,
    register_loader,
)


def _is_solaris_da30_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SOLARIS_EXTENSIONS


def load_solaris_da30_bandmap(path, work_func: float, ef_offset: float = 0.0,
                              a_lattice: float = 0.0, convert_kspace: bool = True) -> ARPESData:
    path = Path(path); erlab_io = _require_erlab(); _set_da30_loader(erlab_io)
    da = erlab_io.load(str(path))
    hv = float(da.attrs.get("hv", np.nan))
    if not np.isfinite(hv) or hv <= 0:
        raise ValueError(
            f"hv missing or invalid in {path.name} (read={hv!r}). "
            f"Check the logbook or enter hν manually before reloading."
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


def _load_solaris_from_registry(path, *, work_func: float, ef_offset: float = 0.0,
                                a_lattice: float = 0.0, **_: Any) -> ARPESData:
    return load_solaris_da30_bandmap(path, work_func=work_func, ef_offset=ef_offset, a_lattice=a_lattice)


register_loader("solaris_da30", _is_solaris_da30_file, _load_solaris_from_registry,
                "Solaris/DA30 via erlab")
