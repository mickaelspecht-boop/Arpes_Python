"""DFT 3D grid : kz from photon energy + 2D slice + iso-EF contour.

Free-electron final-state model :
    kz(hν, V0, W) = 0.5123 × sqrt(hν - W + V0)  [Å⁻¹]
with hν, V0, W in eV. W = analyzer work function (~4.5 eV typical).
Constant 0.5123 = sqrt(2·m_e) / ħ expressed to give 1/Å from eV.

No PyQt. CLAUDE.md layering rule 2 respected.
"""
from __future__ import annotations

from dataclasses import dataclass

import contourpy
import numpy as np


_KZ_CONST_1_PER_ANG_SQRT_EV = 0.5123  # = sqrt(2·m_e)/ħ in Å⁻¹·eV^(−1/2)


def kz_from_hv(hv_eV: float, v0_eV: float, work_function_eV: float = 4.5) -> float:
    """Final-state kz at normal emission for a photoelectron at EF."""
    inner = float(hv_eV) - float(work_function_eV) + float(v0_eV)
    if inner <= 0.0:
        raise ValueError(
            f"kz_from_hv: (hv - W + V0) = {inner:.3f} eV ≤ 0 — impossible."
        )
    return float(_KZ_CONST_1_PER_ANG_SQRT_EV * np.sqrt(inner))


@dataclass(frozen=True)
class DFTSlice:
    kx: np.ndarray         # (n_kx,) 1/Å
    ky: np.ndarray         # (n_ky,) 1/Å
    energy_2d: np.ndarray  # (n_ky, n_kx) eV
    kz_used: float         # 1/Å, where the slice was interpolated


def slice_grid_at_kz(
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    energies: np.ndarray,
    kz_target: float,
) -> DFTSlice:
    """Trilinear-z interpolation: returns the 2D energy map at ``kz_target``.

    ``kz_target`` is clamped to ``[kz[0], kz[-1]]`` (matching periodic BZ folding
    should be handled by the caller if needed).
    """
    z = float(np.clip(kz_target, float(kz[0]), float(kz[-1])))
    iz = int(np.searchsorted(kz, z) - 1)
    iz = max(0, min(iz, kz.size - 2))
    z0, z1 = float(kz[iz]), float(kz[iz + 1])
    t = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    e0 = energies[iz]
    e1 = energies[iz + 1]
    e_slice = (1.0 - t) * e0 + t * e1
    return DFTSlice(kx=np.asarray(kx, dtype=float),
                    ky=np.asarray(ky, dtype=float),
                    energy_2d=np.asarray(e_slice, dtype=float),
                    kz_used=z)


def isocontour_at_energy(
    slice_: DFTSlice,
    energy_eV: float = 0.0,
    seed_point_1_per_ang: tuple[float, float] | None = None,
) -> np.ndarray:
    """Iso-energy contour (in 1/Å) from a DFT 2D slice.

    If ``seed_point`` is given, returns the closed contour containing it.
    Otherwise returns the largest closed contour. Raises ``ValueError`` if
    no closed contour exists at the requested energy.
    """
    from matplotlib.path import Path as MplPath

    z = slice_.energy_2d
    e = float(energy_eV)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("DFT slice: no finite energy.")
    if e < float(np.nanmin(finite)) or e > float(np.nanmax(finite)):
        raise ValueError(
            f"DFT slice: energy {e:.3f} eV out of range "
            f"[{np.nanmin(finite):.3f}, {np.nanmax(finite):.3f}]."
        )
    work = np.where(np.isfinite(z), z, float(np.nanmin(finite)))
    gen = contourpy.contour_generator(x=slice_.kx, y=slice_.ky, z=work, name="serial")
    contours = [np.asarray(c, dtype=float) for c in gen.lines(e)]
    closed = [
        c for c in contours
        if c.shape[0] >= 4 and np.linalg.norm(c[0] - c[-1]) <= 1e-8
    ]
    if not closed:
        raise ValueError("DFT slice: no closed contour at this energy.")
    if seed_point_1_per_ang is not None:
        seed = tuple(map(float, seed_point_1_per_ang))
        containing = [c for c in closed if MplPath(c).contains_point(seed)]
        if not containing:
            raise ValueError("DFT slice: no contour contains the seed.")
        closed = containing

    def _signed_area(c: np.ndarray) -> float:
        return float(0.5 * np.sum(c[:-1, 0] * c[1:, 1] - c[1:, 0] * c[:-1, 1]))

    return max(closed, key=lambda c: abs(_signed_area(c)))
