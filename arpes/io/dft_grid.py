"""DFT 3D bandstructure grid loader (npz).

Accepted schema (npz):
- ``kx`` (n_kx,): kx axis in 1/Å, strictly increasing
- ``ky`` (n_ky,): ky axis in 1/Å, strictly increasing
- ``kz`` (n_kz,): kz axis in 1/Å, strictly increasing
- ``energies`` (n_kz, n_ky, n_kx): E - EF energies in eV (EF assumed at 0)
- ``a_lattice`` (scalar, optional): lattice parameter (Å) for conversion to
  π/a on the display side. If absent, the caller must provide a_lattice.

No PyQt dependency and no Materials Project dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DFTGrid:
    kx: np.ndarray          # (n_kx,) 1/Å
    ky: np.ndarray          # (n_ky,) 1/Å
    kz: np.ndarray          # (n_kz,) 1/Å
    energies: np.ndarray    # (n_kz, n_ky, n_kx) eV (E - EF)
    a_lattice: float        # Å
    source_path: str = ""


def load_dft_grid_npz(path: str | Path, *, a_lattice_fallback: float | None = None) -> DFTGrid:
    """Load a DFT 3D bandstructure from a .npz file.

    Raises ``ValueError`` if required keys are missing or shapes are inconsistent.
    """
    p = Path(path)
    data = np.load(str(p), allow_pickle=False)
    required = {"kx", "ky", "kz", "energies"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"DFT npz: missing keys {sorted(missing)}.")
    kx = np.asarray(data["kx"], dtype=float)
    ky = np.asarray(data["ky"], dtype=float)
    kz = np.asarray(data["kz"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    for axis_name, arr in (("kx", kx), ("ky", ky), ("kz", kz)):
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"DFT npz: axis {axis_name} must be 1D with size >= 2.")
        if not np.all(np.diff(arr) > 0):
            raise ValueError(f"DFT npz: axis {axis_name} must be strictly increasing.")
    expected = (kz.size, ky.size, kx.size)
    if energies.shape != expected:
        raise ValueError(
            f"DFT npz: energies shape {energies.shape} != expected {expected} "
            "(order: kz, ky, kx)."
        )
    if "a_lattice" in data.files:
        a = float(np.asarray(data["a_lattice"]).reshape(-1)[0])
    elif a_lattice_fallback is not None:
        a = float(a_lattice_fallback)
    else:
        raise ValueError("DFT npz: 'a_lattice' is missing and no fallback was provided.")
    if a <= 0.0:
        raise ValueError("DFT npz: a_lattice must be > 0.")
    return DFTGrid(kx=kx, ky=ky, kz=kz, energies=energies, a_lattice=a, source_path=str(p))
