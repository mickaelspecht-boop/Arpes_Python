"""Pure calculations for ARPES k//-kz maps.

Internal convention:
- energy in eV, axis `energy = E - EF`.
- k// provided by loaders in pi/a.
- kz computed internally in A^-1, with optional display in pi/c.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Above this |energy| (eV), the slice is no longer the Fermi surface.
FS_ENERGY_TOL_EV = 0.05

try:
    from scipy.interpolate import griddata as _scipy_griddata
except Exception:  # pragma: no cover - numpy fallback if scipy is missing
    _scipy_griddata = None

try:
    from scipy.spatial import cKDTree as _ScipyCKDTree
except Exception:  # pragma: no cover
    _ScipyCKDTree = None


K_INV_A_PER_SQRT_EV = 0.5123167


@dataclass(frozen=True)
class KzScanInput:
    data: np.ndarray
    kpar: np.ndarray
    energy: np.ndarray
    hv: float
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KzParams:
    work_func: float = 0.0
    inner_potential: float = 12.0
    a_lattice: float = 0.0
    c_lattice: float = 11.6
    energy_center: float = 0.0
    energy_window: float = 0.030
    k_bins: int = 240
    kz_bins: int = 240
    kz_unit: str = "A^-1"
    normalize: str = "per_scan_median"
    display_mode: str = "interpolated"


@dataclass(frozen=True)
class KzMapResult:
    image: np.ndarray
    k_grid: np.ndarray
    kz_grid: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HvKMapResult:
    image: np.ndarray
    k_grid: np.ndarray
    hv_grid: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MdcWaterfallResult:
    curves: np.ndarray
    k_grid: np.ndarray
    hv_grid: np.ndarray
    offsets: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _warn_energy_center(energy_center: float) -> None:
    """P2.5: warn if the slice is no longer at E_F."""
    if abs(float(energy_center)) > FS_ENERGY_TOL_EV:
        warnings.warn(
            f"kz: |E_center|={float(energy_center):.3f} eV > {FS_ENERGY_TOL_EV} "
            "→ this is no longer the Fermi surface (kz plotted away from E_F).",
            RuntimeWarning, stacklevel=2,
        )


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def scan_from_legacy_dict(d: dict[str, Any]) -> KzScanInput:
    hv = _finite_float(d.get("hv") or (d.get("metadata", {}) or {}).get("hv"))
    if hv is None or hv <= 0:
        raise ValueError("kz impossible: hν missing or invalid")
    return KzScanInput(
        data=np.asarray(d["data"], dtype=float),
        kpar=np.asarray(d["kpar"], dtype=float),
        energy=np.asarray(d["ev_arr"], dtype=float),
        hv=hv,
        path=str(d.get("path") or ""),
        metadata=dict(d.get("metadata", {}) or {}),
    )


def standardize_scan(scan: KzScanInput) -> KzScanInput:
    data = np.asarray(scan.data, dtype=float)
    kpar = np.asarray(scan.kpar, dtype=float)
    energy = np.asarray(scan.energy, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"invalid kz scan: data.ndim={data.ndim}, expected 2")
    if data.shape != (kpar.size, energy.size):
        raise ValueError(
            f"invalid kz scan: shape={data.shape}, axes={(kpar.size, energy.size)}"
        )
    if kpar.size < 2 or energy.size < 2:
        raise ValueError("invalid kz scan: axes too short")
    if kpar[0] > kpar[-1]:
        kpar = kpar[::-1]
        data = data[::-1, :]
    if energy[0] > energy[-1]:
        energy = energy[::-1]
        data = data[:, ::-1]
    return KzScanInput(data=data, kpar=kpar, energy=energy, hv=float(scan.hv),
                       path=scan.path, metadata=dict(scan.metadata or {}))


def energy_slice(scan: KzScanInput, center: float, window: float) -> np.ndarray:
    scan = standardize_scan(scan)
    half = max(float(window), 0.0)
    mask = np.abs(scan.energy - float(center)) <= half
    if not mask.any():
        mask[np.argmin(np.abs(scan.energy - float(center)))] = True
    sliced = scan.data[:, mask]
    valid = np.isfinite(sliced)
    counts = valid.sum(axis=1)
    sums = np.where(valid, sliced, 0.0).sum(axis=1)
    out = np.full(scan.kpar.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def kz_from_hv_kpar(
    hv: float,
    kpar_pi_over_a,
    *,
    work_func: float,
    inner_potential: float,
    a_lattice: float,
    energy: float = 0.0,
) -> np.ndarray:
    """Return kz in A^-1 from hν and k// in pi/a."""
    hv = float(hv)
    work_func = float(work_func)
    inner_potential = float(inner_potential)
    a_lattice = float(a_lattice)
    if inner_potential <= 0:
        raise ValueError(f"invalid kz: V0={inner_potential:.3f} eV")
    if a_lattice <= 0:
        raise ValueError(f"invalid kz: a={a_lattice:.3f} A")
    ekin = hv - work_func + float(energy)
    if ekin <= 0:
        raise ValueError(f"invalid kz: Ekin={ekin:.3f} eV")
    kpar_a = np.asarray(kpar_pi_over_a, dtype=float) * np.pi / a_lattice
    ktot2 = (K_INV_A_PER_SQRT_EV ** 2) * (ekin + inner_potential)
    radicand = ktot2 - kpar_a**2
    out = np.full_like(kpar_a, np.nan, dtype=float)
    valid = radicand > 0
    out[valid] = np.sqrt(radicand[valid])
    # P2.5: k// > k_tot: negative radicand, kz undefined (free final state invalid).
    n_neg = int(np.sum(np.isfinite(kpar_a) & ~valid))
    if n_neg:
        warnings.warn(
            f"kz: {n_neg}/{kpar_a.size} points with radicand ≤ 0 (k// > k_tot, "
            f"hν={hv:.0f} eV) → kz=NaN. k// too large for this hν.",
            RuntimeWarning, stacklevel=2,
        )
    return out


def fold_kz_to_1bz(
    kz_inv_a: float,
    c_lattice: float,
    *,
    plane_tol: float = 0.05,
) -> dict:
    """Fold kz into the 1st Brillouin zone (kz ∈ [0, π/c]).

    Returns ``{"kz_reduced_pi_over_c", "n_zone", "plane", "near_boundary"}``.

    - ``plane`` ∈ {"Gamma", "Z", "intermediate"} depending on proximity to 0 or π/c.
    - ``near_boundary`` = True if |kz_red − π/c| < plane_tol·π/c (or |kz_red| < tol).
    - ``plane_tol``: relative tolerance to π/c (default 5%).
    """
    if c_lattice <= 0:
        raise ValueError(f"fold_kz: c={c_lattice:.3f} A invalid")
    kz = float(kz_inv_a)
    if not np.isfinite(kz):
        raise ValueError("fold_kz: kz is not finite")
    g_z = 2.0 * np.pi / float(c_lattice)  # reciprocal vector
    n_zone = int(np.floor((abs(kz) + 0.5 * g_z) / g_z))
    kz_red = abs(kz) - n_zone * g_z  # ∈ [-π/c, +π/c]
    kz_red = abs(kz_red)             # fold kz ↔ -kz symmetry: ∈ [0, π/c]
    kz_red_pi_c = kz_red / (np.pi / float(c_lattice))  # in π/c units, ∈ [0, 1]
    tol = float(plane_tol)
    if kz_red_pi_c <= tol:
        plane = "Gamma"
    elif kz_red_pi_c >= 1.0 - tol:
        plane = "Z"
    else:
        plane = "intermediate"
    near_boundary = (kz_red_pi_c <= tol) or (kz_red_pi_c >= 1.0 - tol) or (
        abs(kz_red_pi_c - 0.5) <= tol
    )
    return {
        "kz_reduced_pi_over_c": float(kz_red_pi_c),
        "n_zone": int(n_zone),
        "plane": plane,
        "near_boundary": bool(near_boundary),
    }


def convert_kz_unit(kz_inv_a, *, unit: str, c_lattice: float) -> np.ndarray:
    kz = np.asarray(kz_inv_a, dtype=float)
    if unit == "A^-1":
        return kz
    if unit == "pi/c":
        if c_lattice <= 0:
            raise ValueError(f"invalid kz: c={c_lattice:.3f} A")
        return kz * float(c_lattice) / np.pi
    raise ValueError(f"unknown kz unit: {unit}")


def _normalize_slice(values: np.ndarray, mode: str) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if mode == "none":
        return vals
    if mode == "per_scan_median":
        finite = vals[np.isfinite(vals) & (vals > 0)]
        if finite.size:
            scale = float(np.nanmedian(finite))
            if scale > 1e-12:
                return vals / scale
        return vals
    raise ValueError(f"unknown kz normalization: {mode}")


def _interpolate_cloud_to_grid(
    k: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    kk_grid: np.ndarray,
    zz_grid: np.ndarray,
) -> np.ndarray:
    """Cloud -> grid interpolation, with support mask to avoid false filling."""
    points = np.column_stack([k, z])
    if _scipy_griddata is not None:
        return _scipy_griddata(points, intensity, (kk_grid, zz_grid), method="linear")

    # Fallback without scipy.interpolate.griddata: local IDW over k-NN.
    out = np.full(kk_grid.shape, np.nan, dtype=float)
    k_span = max(float(np.nanmax(k) - np.nanmin(k)), 1e-12)
    z_span = max(float(np.nanmax(z) - np.nanmin(z)), 1e-12)
    pts = np.column_stack([k / k_span, z / z_span])
    n_neigh = min(8, pts.shape[0])
    flat_k = (kk_grid.ravel() / k_span)
    flat_z = (zz_grid.ravel() / z_span)
    flat_out = out.ravel()
    grid_pts = np.column_stack([flat_k, flat_z])
    if _ScipyCKDTree is not None:
        tree = _ScipyCKDTree(pts)
        d, idx = tree.query(grid_pts, k=n_neigh)
        if n_neigh == 1:
            d = d[:, None]
            idx = idx[:, None]
        weights = 1.0 / np.maximum(d ** 2, 1e-24)
        exact = d[:, 0] < 1e-12
        flat_out[:] = (weights * intensity[idx]).sum(axis=1) / weights.sum(axis=1)
        if exact.any():
            flat_out[exact] = intensity[idx[exact, 0]]
    else:
        # cKDTree missing: Python loop (slow, but avoids crashing in a minimal env).
        for idx_row, (kg, zg) in enumerate(zip(flat_k, flat_z)):
            d2 = (pts[:, 0] - kg) ** 2 + (pts[:, 1] - zg) ** 2
            nearest = np.argpartition(d2, n_neigh - 1)[:n_neigh]
            if d2[nearest[0]] < 1e-24:
                flat_out[idx_row] = intensity[nearest[0]]
                continue
            weights_b = 1.0 / np.maximum(d2[nearest], 1e-24)
            flat_out[idx_row] = float(np.sum(weights_b * intensity[nearest]) / np.sum(weights_b))
    for col, kval in enumerate(kk_grid[0, :]):
        step = abs(kk_grid[0, 1] - kk_grid[0, 0]) if kk_grid.shape[1] > 1 else k_span
        close = np.abs(k - kval) <= max(2.0 * step, 1e-12)
        if not close.any():
            nearest_k = np.argmin(np.abs(k - kval))
            close = np.abs(k - k[nearest_k]) <= max(2.0 * step, 1e-12)
        z_lo = float(np.nanmin(z[close]))
        z_hi = float(np.nanmax(z[close]))
        out[(zz_grid[:, col] < z_lo) | (zz_grid[:, col] > z_hi), col] = np.nan
    return out


def compute_kz_map(scans: Iterable[KzScanInput], params: KzParams) -> KzMapResult:
    scans_std = [standardize_scan(s) for s in scans]
    if len(scans_std) < 2:
        raise ValueError("kz: at least two hν scans required")
    if params.inner_potential <= 0:
        raise ValueError(f"invalid kz: V0={params.inner_potential:.3f} eV")
    _warn_energy_center(params.energy_center)

    point_k: list[np.ndarray] = []
    point_z: list[np.ndarray] = []
    point_i: list[np.ndarray] = []
    skipped: list[str] = []
    for scan in scans_std:
        try:
            vals = energy_slice(scan, params.energy_center, params.energy_window)
            vals = _normalize_slice(vals, params.normalize)
            kz = kz_from_hv_kpar(
                scan.hv, scan.kpar,
                work_func=params.work_func,
                inner_potential=params.inner_potential,
                a_lattice=params.a_lattice,
                energy=params.energy_center,
            )
            kz = convert_kz_unit(kz, unit=params.kz_unit, c_lattice=params.c_lattice)
        except ValueError as exc:
            skipped.append(f"{scan.path or scan.hv}: {exc}")
            continue
        valid = np.isfinite(scan.kpar) & np.isfinite(kz) & np.isfinite(vals)
        if valid.any():
            point_k.append(scan.kpar[valid])
            point_z.append(kz[valid])
            point_i.append(vals[valid])
    if not point_i:
        raise ValueError("kz: no valid points")

    k = np.concatenate(point_k)
    z = np.concatenate(point_z)
    intensity = np.concatenate(point_i)
    k_grid = np.linspace(float(np.nanmin(k)), float(np.nanmax(k)), max(2, int(params.k_bins)))
    z_grid = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), max(2, int(params.kz_bins)))
    kk_grid, zz_grid = np.meshgrid(k_grid, z_grid)

    ki = np.searchsorted(k_grid, k, side="left")
    zi = np.searchsorted(z_grid, z, side="left")
    ki = np.clip(ki, 0, k_grid.size - 1)
    zi = np.clip(zi, 0, z_grid.size - 1)
    image_binned = np.full((z_grid.size, k_grid.size), np.nan, dtype=float)
    counts = np.zeros_like(image_binned, dtype=int)
    sums = np.zeros_like(image_binned, dtype=float)
    for row, col, val in zip(zi, ki, intensity):
        sums[row, col] += float(val)
        counts[row, col] += 1
    filled = counts > 0
    image_binned[filled] = sums[filled] / counts[filled]

    mode = str(params.display_mode or "interpolated")
    if mode == "binned":
        image = image_binned
    elif mode == "interpolated":
        image = _interpolate_cloud_to_grid(k, z, intensity, kk_grid, zz_grid)
        if _scipy_griddata is not None and np.isnan(image).any():
            points = np.column_stack([k, z])
            nearest = _scipy_griddata(points, intensity, (kk_grid, zz_grid), method="nearest")
            inside = np.isfinite(image_binned)
            image[np.isnan(image) & inside] = nearest[np.isnan(image) & inside]
    elif mode == "points":
        image = image_binned
    else:
        raise ValueError(f"unknown KZ mode: {mode}")

    diagnostics = {
        "n_scans": len(scans_std),
        "n_points": int(intensity.size),
        "n_bins_filled": int(filled.sum()),
        "skipped": skipped,
        "kz_unit": params.kz_unit,
        "energy_center": float(params.energy_center),
        "energy_window": float(params.energy_window),
        "display_mode": mode,
        "interpolation_backend": "scipy" if _scipy_griddata is not None else "numpy_idw",
        "point_k": k,
        "point_kz": z,
        "point_i": intensity,
    }
    return KzMapResult(image=image, k_grid=k_grid, kz_grid=z_grid, diagnostics=diagnostics)


def compute_hv_k_map(scans: Iterable[KzScanInput], params: KzParams) -> HvKMapResult:
    """Raw hν-k// map integrated around an energy, without kz conversion."""
    scans_std = [standardize_scan(s) for s in scans]
    if len(scans_std) < 2:
        raise ValueError("hν map: at least two scans required")

    _warn_energy_center(params.energy_center)
    hv_values = np.asarray([scan.hv for scan in scans_std], dtype=float)
    if np.unique(np.round(hv_values, 6)).size < 2:
        raise ValueError("hν map: hν must vary between scans")

    k_min = min(float(np.nanmin(scan.kpar)) for scan in scans_std)
    k_max = max(float(np.nanmax(scan.kpar)) for scan in scans_std)
    k_grid = np.linspace(k_min, k_max, max(2, int(params.k_bins)))

    rows: list[np.ndarray] = []
    skipped: list[str] = []
    hv_kept: list[float] = []
    for scan in scans_std:
        try:
            vals = energy_slice(scan, params.energy_center, params.energy_window)
            vals = _normalize_slice(vals, params.normalize)
        except ValueError as exc:
            skipped.append(f"{scan.path or scan.hv}: {exc}")
            continue
        valid = np.isfinite(scan.kpar) & np.isfinite(vals)
        if valid.sum() < 2:
            skipped.append(f"{scan.path or scan.hv}: not enough valid k points")
            continue
        rows.append(np.interp(k_grid, scan.kpar[valid], vals[valid], left=np.nan, right=np.nan))
        hv_kept.append(float(scan.hv))

    if len(rows) < 2:
        raise ValueError("hν map: not enough valid scans")

    order = np.argsort(hv_kept)
    hv_grid = np.asarray(hv_kept, dtype=float)[order]
    image = np.asarray(rows, dtype=float)[order, :]
    diagnostics = {
        "n_scans": int(len(rows)),
        "n_points": int(np.isfinite(image).sum()),
        "skipped": skipped,
        "energy_center": float(params.energy_center),
        "energy_window": float(params.energy_window),
        "display_mode": "hv map",
        "hv_min": float(np.nanmin(hv_grid)),
        "hv_max": float(np.nanmax(hv_grid)),
    }
    return HvKMapResult(image=image, k_grid=k_grid, hv_grid=hv_grid, diagnostics=diagnostics)


def compute_mdc_waterfall(scans: Iterable[KzScanInput], params: KzParams) -> MdcWaterfallResult:
    """MDC I(k//) curves stacked by hν, integrated around an energy."""
    hv_map = compute_hv_k_map(scans, params)
    curves = np.asarray(hv_map.image, dtype=float)
    curves = np.where(np.isfinite(curves), curves, np.nan)
    if params.normalize != "none":
        for idx in range(curves.shape[0]):
            finite = curves[idx, np.isfinite(curves[idx]) & (curves[idx] > 0)]
            if finite.size:
                scale = float(np.nanpercentile(finite, 95))
                if scale > 1e-12:
                    curves[idx] = curves[idx] / scale

    finite_all = curves[np.isfinite(curves)]
    span = float(np.nanpercentile(finite_all, 95) - np.nanpercentile(finite_all, 5)) if finite_all.size else 1.0
    offset_step = max(span, 1e-12) * 1.15
    offsets = np.arange(curves.shape[0], dtype=float) * offset_step
    diagnostics = dict(hv_map.diagnostics)
    diagnostics["display_mode"] = "MDC waterfall"
    diagnostics["offset_step"] = float(offset_step)
    return MdcWaterfallResult(
        curves=curves,
        k_grid=hv_map.k_grid,
        hv_grid=hv_map.hv_grid,
        offsets=offsets,
        diagnostics=diagnostics,
    )
