"""DFT 3D bandstructure grid loader (npz).

Schema accepté (npz) :
- ``kx`` (n_kx,) : axe kx en 1/Å, strictement croissant
- ``ky`` (n_ky,) : axe ky en 1/Å, strictement croissant
- ``kz`` (n_kz,) : axe kz en 1/Å, strictement croissant
- ``energies`` (n_kz, n_ky, n_kx) : énergies E - EF en eV (EF supposé à 0)
- ``a_lattice`` (scalar, optionnel) : paramètre de maille (Å) pour conversion
  vers π/a côté affichage. Si absent, le caller doit fournir a_lattice.

Aucun PyQt, aucun MP. Layering CLAUDE.md règle 2 respecté.
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
        raise ValueError(f"DFT npz : clefs manquantes {sorted(missing)}.")
    kx = np.asarray(data["kx"], dtype=float)
    ky = np.asarray(data["ky"], dtype=float)
    kz = np.asarray(data["kz"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    for axis_name, arr in (("kx", kx), ("ky", ky), ("kz", kz)):
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"DFT npz : axe {axis_name} doit être 1D de taille ≥ 2.")
        if not np.all(np.diff(arr) > 0):
            raise ValueError(f"DFT npz : axe {axis_name} doit être strictement croissant.")
    expected = (kz.size, ky.size, kx.size)
    if energies.shape != expected:
        raise ValueError(
            f"DFT npz : energies shape {energies.shape} ≠ attendu {expected} "
            "(ordre: kz, ky, kx)."
        )
    if "a_lattice" in data.files:
        a = float(np.asarray(data["a_lattice"]).reshape(-1)[0])
    elif a_lattice_fallback is not None:
        a = float(a_lattice_fallback)
    else:
        raise ValueError("DFT npz : 'a_lattice' absent et pas de fallback fourni.")
    if a <= 0.0:
        raise ValueError("DFT npz : a_lattice doit être > 0.")
    return DFTGrid(kx=kx, ky=ky, kz=kz, energies=energies, a_lattice=a, source_path=str(p))
