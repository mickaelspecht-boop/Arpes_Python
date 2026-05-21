"""Mapping points haute symétrie BZ cristal → repère détecteur ARPES.

Sert l'overlay BZ dans la fenêtre FS (Fig. 4 Ideta 2014 BaNi2P2 ou équiv.).
Réutilise la convention de rotation azi de ``physics/gamma.py``
(``project_gamma_by_azi``) pour cohérence avec Γ FS→BM.

Module pur (numpy uniquement). Aucun PyQt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bz import Lattice3D, bz_points_for_lattice_plane


@dataclass(frozen=True)
class HSProjection:
    """Point haute symétrie projeté dans repère détecteur."""
    kx: float
    ky: float
    label: str
    color: str


def _rot2(theta_rad: float) -> np.ndarray:
    """Matrice rotation 2D, convention cohérente avec ``project_gamma_by_azi``.

    R(Δazi) appliquée à (kx_ref, ky_ref) donne (k_parallel, k_perp) avec
    ``k_perp = -kx*sin + ky*cos`` (cf. gamma.py L110-111).
    """
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array([[c, s], [-s, c]], dtype=float)


def project_hs_points(
    lattice: Lattice3D,
    *,
    plane: str = "Gamma",
    phi_c_deg: float = 0.0,
    azi_ref_deg: float | None = None,
    azi_target_deg: float | None = None,
    gamma_kx: float = 0.0,
    gamma_ky: float = 0.0,
) -> tuple[list[HSProjection], np.ndarray]:
    """Projette points HS d'un cristal 3D dans le plan détecteur ARPES.

    Pipeline :
    1. Récupère polygone BZ + points HS dans repère cristal via
       ``bz_points_for_lattice_plane`` (unités π/a, π/b).
    2. Rotation cristal → détecteur d'angle ``phi_c_deg + (azi_target − azi_ref)``.
       ``phi_c`` = offset cristal/détecteur intrinsèque (a* vs slit analyseur).
       Δazi = différence d'azimut manipulateur (déjà géré pour Γ par gamma.py).
    3. Translation : place Γ à ``(gamma_kx, gamma_ky)`` dans repère détecteur.

    Retourne ``(points_projetés, polygon_xy_projeté)``.

    Notes :
    - Si ``azi_ref`` ou ``azi_target`` est None, Δazi=0 (rotation = phi_c seul).
    - Points conservent labels HS du plan (Γ-plane vs Z-plane) — voir
      ``bz_points_for_lattice_plane``.
    """
    poly, pts, _ = bz_points_for_lattice_plane(lattice, plane=plane)

    d_azi_deg = 0.0
    if azi_ref_deg is not None and azi_target_deg is not None:
        d_azi_deg = float(azi_target_deg) - float(azi_ref_deg)
    theta = np.radians(float(phi_c_deg) + d_azi_deg)
    R = _rot2(theta)

    poly_rot = poly @ R.T
    poly_proj = poly_rot + np.array([float(gamma_kx), float(gamma_ky)])

    projected: list[HSProjection] = []
    for x, y, lab, col in pts:
        v = R @ np.array([float(x), float(y)])
        projected.append(HSProjection(
            kx=float(v[0]) + float(gamma_kx),
            ky=float(v[1]) + float(gamma_ky),
            label=str(lab),
            color=str(col),
        ))
    return projected, poly_proj


def fit_phi_c_from_clicks(
    lattice: Lattice3D,
    *,
    plane: str = "Gamma",
    clicks_kx_ky: list[tuple[float, float]],
    expected_labels: list[str],
    gamma_init_kx: float = 0.0,
    gamma_init_ky: float = 0.0,
) -> dict:
    """Ajuste (phi_c, Γ_kx, Γ_ky) par moindres carrés à partir de clics user.

    ``clicks_kx_ky`` : liste de (kx, ky) cliqués sur la FS ARPES.
    ``expected_labels`` : labels HS correspondants (même ordre), ex ["X", "M"].

    Recherche phi_c sur balayage grossier puis raffinement local.

    Retourne ``{"phi_c_deg", "gamma_kx", "gamma_ky", "residual", "candidates"}``
    où ``candidates`` liste les rotations équivalentes (mod π/2 tétragonal,
    mod π/3 hexagonal) pour lever ambiguïté manuellement.
    """
    if not clicks_kx_ky or not expected_labels:
        raise ValueError("fit_phi_c: clics et labels requis")
    if len(clicks_kx_ky) != len(expected_labels):
        raise ValueError(
            f"fit_phi_c: {len(clicks_kx_ky)} clics ≠ {len(expected_labels)} labels"
        )

    _, hs_raw, _ = bz_points_for_lattice_plane(lattice, plane=plane)
    label_to_xy: dict[str, list[tuple[float, float]]] = {}
    for x, y, lab, _col in hs_raw:
        label_to_xy.setdefault(lab, []).append((float(x), float(y)))

    targets = np.asarray(clicks_kx_ky, dtype=float)

    def residual(phi_deg: float, gx: float, gy: float) -> float:
        R = _rot2(np.radians(phi_deg))
        total = 0.0
        for (kx_obs, ky_obs), lab in zip(targets, expected_labels):
            candidates = label_to_xy.get(lab)
            if not candidates:
                return float("inf")  # label inconnu pour ce plan
            best = float("inf")
            for x, y in candidates:
                v = R @ np.array([x, y])
                kx_p = v[0] + gx
                ky_p = v[1] + gy
                d2 = (kx_obs - kx_p) ** 2 + (ky_obs - ky_p) ** 2
                if d2 < best:
                    best = d2
            total += best
        return float(total)

    # Sweep grossier phi ∈ [0, 360), Γ via centroïde target − centroïde rot HS.
    def gamma_from_phi(phi_deg: float) -> tuple[float, float]:
        R = _rot2(np.radians(phi_deg))
        # Pour chaque clic, on prend la position HS nominale la plus proche après
        # rotation, sans translation (=> Γ=0). Γ = moyenne(observed - rotated_nominal).
        deltas: list[tuple[float, float]] = []
        for (kx_obs, ky_obs), lab in zip(targets, expected_labels):
            cands = label_to_xy.get(lab, [])
            if not cands:
                continue
            best_d2 = float("inf")
            best_dxy = (0.0, 0.0)
            for x, y in cands:
                v = R @ np.array([x, y])
                d2 = (kx_obs - v[0]) ** 2 + (ky_obs - v[1]) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_dxy = (kx_obs - v[0], ky_obs - v[1])
            deltas.append(best_dxy)
        if not deltas:
            return 0.0, 0.0
        arr = np.asarray(deltas, dtype=float)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    sweep = np.linspace(0.0, 360.0, 361, endpoint=False)
    best_phi = 0.0
    best_res = float("inf")
    best_gxy = (float(gamma_init_kx), float(gamma_init_ky))
    for phi in sweep:
        gx, gy = gamma_from_phi(float(phi))
        r = residual(float(phi), gx, gy)
        if r < best_res:
            best_res = r
            best_phi = float(phi)
            best_gxy = (gx, gy)

    # Raffinement local ±2° pas 0.1°.
    for phi in np.arange(best_phi - 2.0, best_phi + 2.001, 0.1):
        gx, gy = gamma_from_phi(float(phi))
        r = residual(float(phi), gx, gy)
        if r < best_res:
            best_res = r
            best_phi = float(phi)
            best_gxy = (gx, gy)

    # Candidats équivalents par symétrie du preset.
    preset = lattice.preset_key()
    if preset in ("square", "rectangle"):
        sym_step = 90.0
    elif preset == "hexagonal":
        sym_step = 60.0
    else:
        sym_step = 180.0
    candidates = [(best_phi + k * sym_step) % 360.0 for k in range(int(360.0 / sym_step))]

    return {
        "phi_c_deg": float(best_phi % 360.0),
        "gamma_kx": float(best_gxy[0]),
        "gamma_ky": float(best_gxy[1]),
        "residual": float(best_res),
        "candidates": sorted(set(round(c, 3) for c in candidates)),
        "n_clicks": int(len(clicks_kx_ky)),
        "preset": preset,
    }
