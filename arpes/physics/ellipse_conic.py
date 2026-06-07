"""Ellipse fit by algebraic conic (Halir-Flusser 1998).

Numerically stable variant of the Fitzgibbon-Pilu-Fisher 1999 direct
least-squares fit (constraint 4ac-b^2=1). Unlike PCA, it works on open partial
arcs: Fermi pockets extending beyond the scan are not closed artificially; the
visible part is fitted and the ellipse is extrapolated, explicitly flagged as
not publication-ready as-is (see P2.4).

Guardrails:
- explicit refusal (``PocketFitRefusedError``) if the arc is too short, points
  are nearly collinear (singular matrix), or the conic is non-elliptic
  (discriminant 4ac-b^2 <= 0) before any square root, so no silent NaN axis.

No PyQt. Pure numpy.
"""
from __future__ import annotations

import numpy as np

# Angular coverage thresholds: more conservative than the mathematical fit
# limit; the unseen axis is unconstrained below about 120 degrees.
ARC_REFUSE_DEG = 120.0      # below this contiguous span: refuse (unconstrained axis)
ARC_FULL_DEG = 340.0        # above this: nearly closed pocket, not extrapolated
MIN_CONIC_POINTS = 6        # 5 DOF ellipse + 1


class PocketFitRefusedError(ValueError):
    """Refus métier d'ajuster une poche (arc trop court / non-ellipse).

    Sous-classe ``ValueError`` pour que le bootstrap saute la run, mais
    distincte pour que ``characterize_pocket`` NE retombe PAS silencieusement
    sur l'aperçu iso-contour (qui referme de force).
    """


def _dedupe_closing(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def contiguous_coverage_deg(points: np.ndarray, center: tuple[float, float]) -> float:
    """Span angulaire CONTIGU couvert (deg) = 360 − plus grand trou.

    Mesure la vraie couverture de l'arc, pas la fraction de directions
    valides : deux arcs séparés de 90° donnent un span faible (gros trou),
    contrairement à ``N_ok/N_total`` (arpes-geometry, arpes-redteam).
    """
    pts = _dedupe_closing(points)
    if pts.shape[0] < 3:
        return 0.0
    ang = np.degrees(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    ang = np.sort(np.mod(ang, 360.0))
    gaps = np.diff(ang)
    wrap = 360.0 - (ang[-1] - ang[0])
    largest = float(max(gaps.max() if gaps.size else 0.0, wrap))
    return float(max(0.0, 360.0 - largest))


def _conic_to_geometric(coef: np.ndarray) -> dict:
    """Conique Ax²+Bxy+Cy²+Dx+Ey+F=0 → (cx, cy, a, b, angle_deg).

    Forme matricielle (robuste). Renvoie ``ok=False`` si non-ellipse.
    """
    A, B, C, D, E, F = (float(v) for v in coef)
    disc = B * B - 4.0 * A * C
    if disc >= 0.0:               # ellipse exige B²−4AC < 0
        return {"ok": False, "reason": f"discriminant {disc:.3g} ≥ 0 (not an ellipse)"}
    m33 = np.array([[A, B / 2.0], [B / 2.0, C]], dtype=float)
    try:
        center = np.linalg.solve(m33, np.array([-D / 2.0, -E / 2.0]))
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "undetermined center (singular matrix)"}
    mfull = np.array([[A, B / 2.0, D / 2.0],
                      [B / 2.0, C, E / 2.0],
                      [D / 2.0, E / 2.0, F]], dtype=float)
    det_full = float(np.linalg.det(mfull))
    det_m33 = float(np.linalg.det(m33))
    if abs(det_m33) < 1e-18:
        return {"ok": False, "reason": "singular m33"}
    eigvals, eigvecs = np.linalg.eigh(m33)
    axes_sq = -det_full / (det_m33 * eigvals)
    if not np.all(np.isfinite(axes_sq)) or np.any(axes_sq <= 0.0):
        return {"ok": False, "reason": "axes² ≤ 0 (degenerate conic)"}
    axes = np.sqrt(axes_sq)
    # Grand axe ↔ plus grand axe² ; direction = SON vecteur propre.
    major_i = int(np.argmax(axes_sq))
    minor_i = 1 - major_i
    a_major = float(axes[major_i])
    b_minor = float(axes[minor_i])
    vec = eigvecs[:, major_i]
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    angle = (angle + 90.0) % 180.0 - 90.0
    return {"ok": True, "cx": float(center[0]), "cy": float(center[1]),
            "a": max(a_major, b_minor), "b": min(a_major, b_minor),
            "angle_deg": angle, "reason": ""}


def fit_ellipse_conic(points: np.ndarray) -> dict:
    """Halir-Flusser : ajuste une ellipse sur ``points`` (N≥6), arc OK.

    Renvoie ``{ok, cx, cy, a, b, angle_deg, reason}``. ``a`` = grand demi-axe,
    ``b`` = petit. ``ok=False`` + ``reason`` si dégénéré (jamais de NaN).
    Normalisation isotrope interne pour le conditionnement (axes/​centre
    dé-normalisés ; l'angle est invariant par homothétie).
    """
    pts = _dedupe_closing(points)
    if pts.shape[0] < MIN_CONIC_POINTS:
        return {"ok": False, "reason": f"{pts.shape[0]} points < {MIN_CONIC_POINTS}"}
    x = pts[:, 0]
    y = pts[:, 1]
    mx, my = float(np.mean(x)), float(np.mean(y))
    xc, yc = x - mx, y - my
    scale = float(np.sqrt(np.mean(xc * xc + yc * yc)))
    if not np.isfinite(scale) or scale <= 0.0:
        return {"ok": False, "reason": "degenerate points (zero scale)"}
    xn, yn = xc / scale, yc / scale

    d1 = np.column_stack([xn * xn, xn * yn, yn * yn])
    d2 = np.column_stack([xn, yn, np.ones_like(xn)])
    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2
    try:
        s3_inv = np.linalg.inv(s3)
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "singular S3 (collinear points)"}
    t = -s3_inv @ s2.T
    m = s1 + s2 @ t
    c1_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    m = c1_inv @ m
    try:
        eigval, eigvec = np.linalg.eig(m)
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "conic eig failed"}
    cond = 4.0 * eigvec[0].real * eigvec[2].real - eigvec[1].real ** 2
    valid = np.where(cond > 0)[0]
    if valid.size == 0:
        return {"ok": False, "reason": "no elliptic eigenvector (4ac−b²≤0)"}
    a1 = eigvec[:, valid[0]].real
    a2 = t @ a1
    coef_n = np.concatenate([a1, a2])  # coords normalisées

    geo = _conic_to_geometric(coef_n)
    if not geo.get("ok"):
        return geo
    # Dé-normalisation : homothétie isotrope de facteur ``scale`` + translation.
    geo["cx"] = geo["cx"] * scale + mx
    geo["cy"] = geo["cy"] * scale + my
    geo["a"] = geo["a"] * scale
    geo["b"] = geo["b"] * scale
    return geo


def conic_axis_sigma(points: np.ndarray, *, n_boot: int = 120,
                     seed: int | None = 0) -> tuple[float, float]:
    """sigma(a), sigma(b) by bootstrapping arc points and refitting the conic.

    Returns ``(nan, nan)`` if too many runs fail. Fit scatter underestimates
    extrapolation error (model error excluded), so the caller inflates it based
    on the missing arc fraction.
    """
    pts = _dedupe_closing(points)
    n = pts.shape[0]
    if n < MIN_CONIC_POINTS:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a_s, b_s = [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        g = fit_ellipse_conic(pts[idx])
        if g.get("ok"):
            a_s.append(g["a"])
            b_s.append(g["b"])
    if len(a_s) < max(5, n_boot // 4):
        return float("nan"), float("nan")
    return float(np.std(a_s, ddof=1)), float(np.std(b_s, ddof=1))
