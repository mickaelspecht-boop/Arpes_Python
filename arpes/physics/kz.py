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


def kz_unit_to_inv_a(kz_value, *, unit: str, c_lattice: float) -> np.ndarray:
    """Inverse of ``convert_kz_unit``: bring a display-unit kz back to A^-1."""
    kz = np.asarray(kz_value, dtype=float)
    if unit == "A^-1":
        return kz
    if unit == "pi/c":
        if c_lattice <= 0:
            raise ValueError(f"invalid kz: c={c_lattice:.3f} A")
        return kz * np.pi / float(c_lattice)
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

    # Single output: interpolated cloud -> grid (the publishable kz map). Raw
    # sample points are always returned in diagnostics for the optional overlay.
    k_spread = float(np.ptp(k)) if k.size else 0.0
    z_spread = float(np.ptp(z)) if z.size else 0.0
    degenerate = k_spread <= 1e-9 or z_spread <= 1e-9
    if degenerate:
        # Collinear cloud (e.g. normal-emission scan with k//≈const): a 2D
        # triangulation is impossible, so fall back to the binned map instead
        # of letting scipy raise a QhullError. P2: a flat k// usually means the
        # angle→k// conversion lacked the lattice parameter a.
        warnings.warn(
            "kz: degenerate point cloud (k// or kz has no spread) → showing "
            "binned map. Check the lattice parameter a (k// may be all zero).",
            RuntimeWarning, stacklevel=2,
        )
        image = image_binned
    else:
        try:
            image = _interpolate_cloud_to_grid(k, z, intensity, kk_grid, zz_grid)
        except Exception:
            image = image_binned
        if _scipy_griddata is not None and np.isnan(image).any():
            points = np.column_stack([k, z])
            try:
                nearest = _scipy_griddata(points, intensity, (kk_grid, zz_grid), method="nearest")
                inside = np.isfinite(image_binned)
                image[np.isnan(image) & inside] = nearest[np.isnan(image) & inside]
            except Exception:
                pass

    diagnostics = {
        "n_scans": len(scans_std),
        "n_points": int(intensity.size),
        "n_bins_filled": int(filled.sum()),
        "skipped": skipped,
        "kz_unit": params.kz_unit,
        "energy_center": float(params.energy_center),
        "energy_window": float(params.energy_window),
        "interpolation_backend": "scipy" if _scipy_griddata is not None else "numpy_idw",
        "degenerate_kpar": bool(degenerate),
        "point_k": k,
        "point_kz": z,
        "point_i": intensity,
    }
    return KzMapResult(image=image, k_grid=k_grid, kz_grid=z_grid, diagnostics=diagnostics)


def _kz_at_normal_emission(hv: float, *, work_func: float, inner_potential: float,
                           energy: float = 0.0) -> float:
    """kz (A^-1) at k//=0 for one photon energy; nan if non-physical."""
    ekin = float(hv) - float(work_func) + float(energy)
    val = (K_INV_A_PER_SQRT_EV ** 2) * (ekin + float(inner_potential))
    return float(np.sqrt(val)) if val > 0 else float("nan")


def _normal_emission_intensity(
    scans: Iterable[KzScanInput], params: KzParams, kpar_window: float,
) -> dict:
    """Per-scan E_F intensity at k//≈0, normalized by each scan's full-k mean.

    The full-k normalization removes the hν-dependent cross-section drift while
    keeping the normal-emission modulation (the kz signal lives *between* scans,
    so per-scan median normalization would erase it).
    """
    scans_std = [standardize_scan(s) for s in scans]
    if len(scans_std) < 2:
        raise ValueError("kz profile: at least two hν scans required")
    hv: list[float] = []
    inten: list[float] = []
    for scan in scans_std:
        vals = energy_slice(scan, params.energy_center, params.energy_window)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        full = float(np.nanmean(vals[finite]))
        if not np.isfinite(full) or abs(full) < 1e-12:
            continue
        sel = (np.abs(scan.kpar) <= float(kpar_window)) & finite
        if sel.any():
            center = float(np.nanmean(vals[sel]))
        else:
            idx = int(np.argmin(np.abs(scan.kpar)))
            center = float(vals[idx]) if finite[idx] else float("nan")
        if not np.isfinite(center):
            continue
        hv.append(float(scan.hv))
        inten.append(center / full)
    if len(hv) < 2:
        raise ValueError("kz profile: not enough valid scans")
    return {"hv": np.asarray(hv, dtype=float), "intensity": np.asarray(inten, dtype=float)}


def kz_profile_at_normal_emission(
    scans: Iterable[KzScanInput], params: KzParams, *, kpar_window: float = 0.05,
) -> dict:
    """1D normal-emission profile I(kz) at k//≈0, sorted by kz, with FFT period.

    Returns ``{"kz", "intensity", "hv", "c_implied"}``. ``c_implied`` is the
    lattice c inferred from the dominant FFT period of the modulation
    (``c = 2π · f_peak``); compare it to the input c as a consistency check.
    """
    prof = _normal_emission_intensity(scans, params, kpar_window)
    hv = prof["hv"]
    inten = prof["intensity"]
    kz = np.asarray([
        _kz_at_normal_emission(h, work_func=params.work_func,
                               inner_potential=params.inner_potential,
                               energy=params.energy_center)
        for h in hv
    ])
    good = np.isfinite(kz) & np.isfinite(inten)
    kz, inten, hv = kz[good], inten[good], hv[good]
    order = np.argsort(kz)
    kz, inten, hv = kz[order], inten[order], hv[order]
    c_implied = float("nan")
    if kz.size >= 4 and float(np.ptp(kz)) > 1e-6:
        ku = np.linspace(float(kz[0]), float(kz[-1]), max(64, kz.size * 4))
        iu = np.interp(ku, kz, inten)
        iu = iu - float(np.mean(iu))
        spec = np.abs(np.fft.rfft(iu))
        freqs = np.fft.rfftfreq(ku.size, d=float(ku[1] - ku[0]))
        spec[0] = 0.0
        f_peak = float(freqs[int(np.argmax(spec))])
        if f_peak > 0:
            c_implied = float(2.0 * np.pi * f_peak)
    return {"kz": kz, "intensity": inten, "hv": hv, "c_implied": c_implied}


def _lomb_scargle_power(x: np.ndarray, y: np.ndarray, omega: float) -> float:
    """Normalized Lomb-Scargle power at one angular frequency, ∈ [0, 1].

    Measures how well ``y`` sampled at (uneven) positions ``x`` fits a sinusoid
    of angular frequency ``omega``. 1 = perfect single sinusoid, 0 = none. No
    clustering bias (unlike a naive circular order parameter), and it handles
    the non-uniform kz spacing produced by the sqrt mapping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - float(np.mean(y))
    var = float(np.sum(y ** 2))
    if var < 1e-30 or x.size < 3 or omega <= 0:
        return 0.0
    w2 = 2.0 * omega
    tau = np.arctan2(np.sum(np.sin(w2 * x)), np.sum(np.cos(w2 * x))) / w2
    cc = np.cos(omega * (x - tau))
    ss = np.sin(omega * (x - tau))
    sc = float(np.sum(cc ** 2))
    sss = float(np.sum(ss ** 2))
    term_c = (float(np.sum(y * cc)) ** 2 / sc) if sc > 1e-30 else 0.0
    term_s = (float(np.sum(y * ss)) ** 2 / sss) if sss > 1e-30 else 0.0
    return float(np.clip((term_c + term_s) / var, 0.0, 1.0))


def fit_inner_potential(
    scans: Iterable[KzScanInput], params: KzParams, *,
    v0_min: float = 5.0, v0_max: float = 30.0, n_steps: int = 251,
    kpar_window: float = 0.05,
) -> dict:
    """Fit the inner potential V0 from kz periodicity at normal emission.

    Sweeps V0 and maximizes the Lomb-Scargle power of the normal-emission
    profile ``I(kz0(V0))`` at the crystal angular frequency ``ω = 2π/(2π/c) = c``
    (period ``2π/c``). At the right V0 the kz axis is undistorted, so the E_F
    modulation is a clean sinusoid of period ``2π/c`` → power → 1.

    Returns ``{"v0_best", "v0_sigma", "power", "cluster_phase", "n_zones",
    "boundary", "confidence", "v0_grid", "power_curve"}``. ``power`` ∈ [0,1] is
    the periodicity significance; ``confidence`` is "low" when power is weak, the
    optimum rails to the V0 range edge, or fewer than ~1.5 zones are covered —
    i.e. the scan does not actually constrain V0. ``cluster_phase`` ≈ 0 → maxima
    on Γ, ≈ 0.5 → on Z.
    """
    if params.c_lattice <= 0:
        raise ValueError(f"fit V0: c={params.c_lattice:.3f} A invalid")
    prof = _normal_emission_intensity(scans, params, kpar_window)
    hv = prof["hv"]
    inten = prof["intensity"]

    omega = float(params.c_lattice)  # 2π / (period 2π/c)
    v0_grid = np.linspace(float(v0_min), float(v0_max), max(5, int(n_steps)))
    power_curve = np.zeros_like(v0_grid)
    ekin = hv - params.work_func + params.energy_center
    for j, v0 in enumerate(v0_grid):
        val = (K_INV_A_PER_SQRT_EV ** 2) * (ekin + v0)
        kz0 = np.sqrt(np.where(val > 0, val, np.nan))
        m = np.isfinite(kz0)
        if m.sum() < 3:
            continue
        power_curve[j] = _lomb_scargle_power(kz0[m], inten[m], omega)

    j_best = int(np.argmax(power_curve))
    v0_best = float(v0_grid[j_best])
    power = float(power_curve[j_best])
    boundary = j_best in (0, v0_grid.size - 1)

    val = (K_INV_A_PER_SQRT_EV ** 2) * (ekin + v0_best)
    kz0 = np.sqrt(np.where(val > 0, val, np.nan))
    m = np.isfinite(kz0)
    z = np.sum((inten[m] - np.mean(inten[m])) * np.exp(1j * omega * kz0[m]))
    cluster_phase = float((np.angle(z) / (2.0 * np.pi)) % 1.0)
    n_zones = float((np.nanmax(kz0) - np.nanmin(kz0)) / (2.0 * np.pi / params.c_lattice)) if m.any() else 0.0

    v0_sigma = _parabola_sigma(v0_grid, power_curve, j_best)
    weak = power < 0.5 or boundary or n_zones < 1.5
    confidence = "low" if weak else "ok"
    return {
        "v0_best": v0_best,
        "v0_sigma": v0_sigma,
        "power": power,
        "cluster_phase": cluster_phase,
        "n_zones": n_zones,
        "boundary": bool(boundary),
        "confidence": confidence,
        "v0_grid": v0_grid,
        "power_curve": power_curve,
    }


def _parabola_sigma(x: np.ndarray, y: np.ndarray, j: int) -> float:
    """Curvature-based 1σ of a peak at index ``j`` from a local parabola fit."""
    if j <= 0 or j >= x.size - 1:
        return float("nan")
    x0, x1, x2 = float(x[j - 1]), float(x[j]), float(x[j + 1])
    y0, y1, y2 = float(y[j - 1]), float(y[j]), float(y[j + 1])
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-30:
        return float("nan")
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    if a >= 0:  # not a concave peak
        return float("nan")
    # Treat (R_max − R) like a χ²-style well: σ where the order parameter drops
    # by ~1 curvature unit. Scale by the peak height for a meaningful spread.
    return float(np.sqrt(max(y1, 1e-6) / (-a)))


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


def hv_for_kz(
    kz_inv_a: float,
    *,
    work_func: float,
    inner_potential: float,
    energy: float = 0.0,
) -> float:
    """Inverse of ``kz_from_hv_kpar`` at k//=0: photon energy giving this kz.

    From ``kz = K·sqrt(Ekin + V0)`` at normal emission (k//=0), with
    ``K = K_INV_A_PER_SQRT_EV`` and ``Ekin = hν − φ + E``::

        hν = (kz / K)**2 − V0 + φ − E

    Returns ``nan`` if the resulting photon energy would be non-physical.
    """
    kz = float(kz_inv_a)
    if not np.isfinite(kz):
        return float("nan")
    ekin = (kz / K_INV_A_PER_SQRT_EV) ** 2 - float(inner_potential)
    hv = ekin + float(work_func) - float(energy)
    return float(hv) if np.isfinite(hv) and hv > 0 else float("nan")


def kz_high_symmetry_planes(
    kz_min_inv_a: float,
    kz_max_inv_a: float,
    c_lattice: float,
    *,
    unit: str = "A^-1",
) -> list[dict]:
    """High-symmetry kz planes within ``[kz_min, kz_max]`` (in A^-1).

    Γ planes sit at ``kz = m·2π/c`` and Z planes at ``kz = (2m+1)·π/c``; the
    spacing between consecutive Γ/Z planes is ``π/c``. Each returned dict holds
    ``{"kz", "label", "n"}`` with ``kz`` already converted to ``unit`` and
    ``label`` ∈ {"Γ", "Z"} (Γ for even ``n``, Z for odd).
    """
    if c_lattice <= 0:
        raise ValueError(f"kz planes: c={c_lattice:.3f} A invalid")
    lo = min(float(kz_min_inv_a), float(kz_max_inv_a))
    hi = max(float(kz_min_inv_a), float(kz_max_inv_a))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return []
    spacing = np.pi / float(c_lattice)  # Γ ↔ Z distance in A^-1
    n0 = int(np.floor(lo / spacing))
    n1 = int(np.ceil(hi / spacing))
    planes: list[dict] = []
    for n in range(n0, n1 + 1):
        kz = n * spacing
        if kz < lo - 1e-12 or kz > hi + 1e-12:
            continue
        kz_disp = float(convert_kz_unit(kz, unit=unit, c_lattice=c_lattice))
        planes.append({"kz": kz_disp, "label": "Γ" if n % 2 == 0 else "Z", "n": int(n)})
    return planes


def kz_coverage_summary(
    kz_min_inv_a: float,
    kz_max_inv_a: float,
    c_lattice: float,
    *,
    work_func: float,
    inner_potential: float,
    energy: float = 0.0,
) -> dict:
    """Periodicity readout for the kz range: zones covered + plane hν hints.

    Returns ``{"n_zones", "gamma_hv", "z_hv"}`` where ``n_zones`` is the kz span
    expressed in units of ``π/c`` (one Γ→Z step = 1) and ``gamma_hv``/``z_hv``
    are sorted lists of photon energies (eV, at k//=0) hitting Γ/Z planes.
    """
    if c_lattice <= 0:
        raise ValueError(f"kz coverage: c={c_lattice:.3f} A invalid")
    lo = min(float(kz_min_inv_a), float(kz_max_inv_a))
    hi = max(float(kz_min_inv_a), float(kz_max_inv_a))
    spacing = np.pi / float(c_lattice)
    n_zones = float((hi - lo) / spacing) if np.isfinite(hi - lo) else 0.0
    planes = kz_high_symmetry_planes(lo, hi, c_lattice, unit="A^-1")
    gamma_hv: list[float] = []
    z_hv: list[float] = []
    for plane in planes:
        hv = hv_for_kz(
            plane["kz"], work_func=work_func,
            inner_potential=inner_potential, energy=energy,
        )
        if not np.isfinite(hv):
            continue
        (gamma_hv if plane["label"] == "Γ" else z_hv).append(round(float(hv), 1))
    return {"n_zones": n_zones, "gamma_hv": sorted(gamma_hv), "z_hv": sorted(z_hv)}
