"""Opérations pures sur FS map 2D : regrid commun, diff/sum/ratio.

Sert le panneau "Comparaison polarisation" (LV vs LH, etc.) : permet de
projeter deux FS d'origine grilles différentes sur une grille commune
avant calcul de différence / somme / ratio normalisé.

Module pur numpy/scipy (interpolation optionnelle). Aucun PyQt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator as _RGI
except Exception:  # pragma: no cover
    _RGI = None


@dataclass(frozen=True)
class FsPair:
    """Deux FS regrillées sur axes communs (kx, ky)."""
    kx: np.ndarray
    ky: np.ndarray
    fs_a: np.ndarray
    fs_b: np.ndarray
    label_a: str = "A"
    label_b: str = "B"
    overlap_ratio: float = 1.0   # fraction de la grille commune couverte par les 2


def _bilinear_regrid(
    kx_src: np.ndarray, ky_src: np.ndarray, fs_src: np.ndarray,
    kx_dst: np.ndarray, ky_dst: np.ndarray,
) -> np.ndarray:
    """Interpole bilinéairement fs_src(kx_src, ky_src) → grille (kx_dst, ky_dst).

    Hors domaine source : NaN. Pas d'extrapolation.
    """
    fs_src = np.asarray(fs_src, dtype=float)
    if fs_src.ndim != 2:
        raise ValueError(f"fs_src doit être 2D, got ndim={fs_src.ndim}")
    if fs_src.shape != (len(ky_src), len(kx_src)) and fs_src.shape != (len(kx_src), len(ky_src)):
        raise ValueError(
            f"fs_src shape {fs_src.shape} incompatible axes "
            f"({len(kx_src)}, {len(ky_src)})"
        )
    # Standardise : on travaille en (ky, kx) — convention pcolormesh.
    if fs_src.shape == (len(kx_src), len(ky_src)):
        fs_src = fs_src.T  # → (ky, kx)

    # Tri axes croissants (RGI exige strict-monotone croissant).
    if kx_src[0] > kx_src[-1]:
        kx_src = kx_src[::-1]; fs_src = fs_src[:, ::-1]
    if ky_src[0] > ky_src[-1]:
        ky_src = ky_src[::-1]; fs_src = fs_src[::-1, :]

    if _RGI is None:
        # Fallback : repli sur valeur du voisin le plus proche (lent mais robuste).
        out = np.full((len(ky_dst), len(kx_dst)), np.nan, dtype=float)
        for jy, y in enumerate(ky_dst):
            if y < ky_src[0] or y > ky_src[-1]:
                continue
            iy = int(np.clip(np.searchsorted(ky_src, y), 1, len(ky_src) - 1))
            for ix, x in enumerate(kx_dst):
                if x < kx_src[0] or x > kx_src[-1]:
                    continue
                jx = int(np.clip(np.searchsorted(kx_src, x), 1, len(kx_src) - 1))
                out[jy, ix] = fs_src[iy, jx]
        return out

    rgi = _RGI((ky_src, kx_src), fs_src, method="linear",
               bounds_error=False, fill_value=np.nan)
    YY, XX = np.meshgrid(ky_dst, kx_dst, indexing="ij")
    pts = np.column_stack([YY.ravel(), XX.ravel()])
    return rgi(pts).reshape(YY.shape)


def regrid_to_common(
    kx_a: np.ndarray, ky_a: np.ndarray, fs_a: np.ndarray,
    kx_b: np.ndarray, ky_b: np.ndarray, fs_b: np.ndarray,
    *,
    n_kx: int | None = None,
    n_ky: int | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> FsPair:
    """Projette fs_a et fs_b sur grille commune (intersection des plages).

    - Si ``n_kx``/``n_ky`` None : reprend la finesse min des deux sources.
    - Retourne ``FsPair`` avec overlap_ratio = fraction de cellules ayant des
      valeurs valides dans les 2 FS regrillées.

    Raises:
        ValueError si overlap < 1 cellule (plages disjointes).
    """
    kx_a = np.asarray(kx_a, dtype=float); ky_a = np.asarray(ky_a, dtype=float)
    kx_b = np.asarray(kx_b, dtype=float); ky_b = np.asarray(ky_b, dtype=float)
    kx_lo = max(float(np.nanmin(kx_a)), float(np.nanmin(kx_b)))
    kx_hi = min(float(np.nanmax(kx_a)), float(np.nanmax(kx_b)))
    ky_lo = max(float(np.nanmin(ky_a)), float(np.nanmin(ky_b)))
    ky_hi = min(float(np.nanmax(ky_a)), float(np.nanmax(ky_b)))
    if kx_hi <= kx_lo or ky_hi <= ky_lo:
        raise ValueError(
            f"regrid_to_common : plages disjointes "
            f"(kx [{kx_lo:.3f},{kx_hi:.3f}], ky [{ky_lo:.3f},{ky_hi:.3f}])"
        )
    if n_kx is None:
        n_kx = max(8, min(len(kx_a), len(kx_b)))
    if n_ky is None:
        n_ky = max(8, min(len(ky_a), len(ky_b)))
    kx_grid = np.linspace(kx_lo, kx_hi, int(n_kx))
    ky_grid = np.linspace(ky_lo, ky_hi, int(n_ky))
    A = _bilinear_regrid(kx_a, ky_a, fs_a, kx_grid, ky_grid)
    B = _bilinear_regrid(kx_b, ky_b, fs_b, kx_grid, ky_grid)
    both = np.isfinite(A) & np.isfinite(B)
    overlap_ratio = float(both.sum()) / max(both.size, 1)
    if both.sum() == 0:
        raise ValueError("regrid_to_common : overlap nul après interpolation.")
    return FsPair(
        kx=kx_grid, ky=ky_grid, fs_a=A, fs_b=B,
        label_a=str(label_a), label_b=str(label_b),
        overlap_ratio=overlap_ratio,
    )


def fs_diff(pair: FsPair, *, normalize: str = "none") -> np.ndarray:
    """Différence A − B sur grille commune.

    ``normalize`` :
    - ``"none"`` : A − B brut.
    - ``"max"``  : (A − B) / max(|A|, |B|) cell-wise (évite div par 0).
    - ``"sum"``  : (A − B) / (A + B)  (dichroïsme circulaire-like).
    """
    A = pair.fs_a; B = pair.fs_b
    if normalize == "none":
        return A - B
    if normalize == "max":
        denom = np.maximum(np.abs(A), np.abs(B))
        out = np.full_like(A, np.nan)
        mask = denom > 1e-12
        out[mask] = (A[mask] - B[mask]) / denom[mask]
        return out
    if normalize == "sum":
        denom = A + B
        out = np.full_like(A, np.nan)
        mask = np.abs(denom) > 1e-12
        out[mask] = (A[mask] - B[mask]) / denom[mask]
        return out
    raise ValueError(f"fs_diff : normalize inconnu : {normalize!r}")


def fs_sum(pair: FsPair) -> np.ndarray:
    """Somme A + B (utile pour FS totale moyennée sur polarisations)."""
    return pair.fs_a + pair.fs_b


def fs_ratio(pair: FsPair, *, eps: float = 1e-12) -> np.ndarray:
    """Rapport A / B. NaN où |B| < eps."""
    out = np.full_like(pair.fs_a, np.nan)
    mask = np.abs(pair.fs_b) > eps
    out[mask] = pair.fs_a[mask] / pair.fs_b[mask]
    return out


def group_files_by_pol(
    logbook_records: list[dict],
    *,
    pol_key: str = "Pol",
    group_keys: tuple[str, ...] = ("material", "run_id"),
) -> dict[tuple, dict[str, list[str]]]:
    """Groupe entrées logbook par (matériau, run_id) → {pol: [paths...]}.

    - Ignore lignes sans pol non vide (warning compté côté caller via len()).
    - Polarisations normalisées en majuscules (LV/LH/RC/LC).
    - Retourne dict :
      ``{(material, run_id): {"LV": [path1], "LH": [path2, path3], ...}}``
    """
    grouped: dict[tuple, dict[str, list[str]]] = {}
    for rec in (logbook_records or []):
        pol = str((rec.get(pol_key) or "")).strip().upper()
        if not pol:
            continue
        path = str(rec.get("path") or rec.get("filename") or "")
        if not path:
            continue
        key = tuple(str(rec.get(k, "") or "") for k in group_keys)
        if not any(key):
            continue
        grouped.setdefault(key, {}).setdefault(pol, []).append(path)
    return grouped


def find_pol_partner(
    grouped: dict[tuple, dict[str, list[str]]],
    current_path: str,
    *,
    other_pol: str = "LH",
) -> str | None:
    """Trouve le partenaire `other_pol` d'un fichier dans un groupe."""
    cur = str(current_path)
    other_pol = str(other_pol).upper()
    for _, by_pol in grouped.items():
        all_paths = [p for paths in by_pol.values() for p in paths]
        if cur not in all_paths:
            continue
        candidates = by_pol.get(other_pol, [])
        if candidates:
            return candidates[0]
    return None
