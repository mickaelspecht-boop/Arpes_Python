"""Shared linear dispersion fit E(k) near E_F.

Single implementation for P2.2 quantitative rigor (vF/kF/m*):

* ``linear_dispersion_fit``: E ~= slope*k + intercept with 2x2 covariance.
  Uses orthogonal regression (total least squares) when a per-point k
  uncertainty is provided because both k and E carry noise near E_F; vertical
  OLS underestimates the slope. Without a real sigma_k, falls back to weighted
  OLS via ``polyfit(cov=True)``.

  TLS is implemented with a 2x2 eigendecomposition of the cloud scaled by
  (sigma_k, sigma_E), without ``scipy.odr``. Parameter covariance comes from a
  leave-one-out jackknife.

* ``curvature_ratio``: |a|*delta_k / |b| for the quadratic fit E=a*k^2+b*k+c.
  Linearity gate before trusting kF=-intercept/slope.

No PyQt, no deprecated dependency. Used by ``physics.fit`` (Im Sigma path) and
``analysis.results`` / ``analysis.bootstrap`` (table/export).
"""
from __future__ import annotations

import numpy as np

# Shared thresholds.
MIN_DISP_POINTS = 5       # quadratic gate + meaningless regression below 5 points
CURVATURE_MAX = 0.10      # tolerated |a|·Δk / |b| before nonlinear refusal
SLOPE_FLOOR = 1e-9        # below this slope -> kF=-b/a diverges


def curvature_ratio(k: np.ndarray, e: np.ndarray) -> float:
    """|a|·Δk / |b| du fit quadratique E=a·k²+b·k+c sur (k, e).

    Mesure la contribution quadratique relative à la pente linéaire sur la
    plage de k ajustée. ``nan`` si fit impossible ou pente nulle.
    """
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    dk = float(np.ptp(k)) if k.size else 0.0
    if dk <= 0 or k.size < 3:
        return float("nan")
    try:
        a2, b2, _ = np.polyfit(k, e, 2)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    return float(abs(a2) * dk / max(abs(b2), 1e-12))


def _tls_slope_intercept(
    k: np.ndarray, e: np.ndarray, sx: float, sy: float
) -> tuple[float, float]:
    """Slope/intercept by total least squares scaled by (sx, sy).

    Scale (k, e) by uncertainties so errors are isotropic, then take the
    dominant eigen-direction of the centered cloud. Returns (nan, nan) if
    degenerate (vertical direction in scaled space).
    """
    kbar = float(np.mean(k))
    ebar = float(np.mean(e))
    u = (k - kbar) / sx
    v = (e - ebar) / sy
    cuu = float(np.dot(u, u))
    cvv = float(np.dot(v, v))
    cuv = float(np.dot(u, v))
    cov = np.array([[cuu, cuv], [cuv, cvv]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return float("nan"), float("nan")
    vals, vecs = np.linalg.eigh(cov)         # croissant
    cu, cv = vecs[0, 1], vecs[1, 1]          # vecteur propre dominant
    if abs(cu) < 1e-15:
        return float("nan"), float("nan")
    slope = (sy / sx) * (cv / cu)            # retour aux unités physiques
    intercept = ebar - slope * kbar
    return float(slope), float(intercept)


def linear_dispersion_fit(
    k: np.ndarray, e: np.ndarray, sk: np.ndarray | None
) -> dict:
    """Fit E = slope*k + intercept with full 2x2 covariance.

    ``sk``: per-point standard deviation on k. Finite and > 0 means orthogonal
    regression (TLS, sigma_k scale vs median energy step), covariance by
    jackknife. Otherwise use weighted vertical OLS via ``np.polyfit(cov=True)``.

    Renvoie ``{ok, slope, intercept, cov, method}``. ``cov`` = matrice 2×2
    [[var_s, cov_si], [cov_si, var_i]] (ordre slope, intercept) pour propager
    σ_kF/σ_m* en gardant la corrélation slope↔intercept (non négligeable en
    fenêtre étroite). ``ok=False`` si non fini / pente nulle / dégénéré.
    """
    fail = {"ok": False, "slope": float("nan"), "intercept": float("nan"),
            "cov": None, "method": "none"}
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    if k.size < 2 or k.size != e.size:
        return fail

    sk_arr = None if sk is None else np.asarray(sk, dtype=float)
    use_tls = (
        sk_arr is not None and sk_arr.size == k.size
        and np.all(np.isfinite(sk_arr)) and np.all(sk_arr > 0)
    )

    if not use_tls:
        # OLS pondéré vertical + covariance analytique.
        try:
            beta0, cov0 = np.polyfit(k, e, 1, cov=True)
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return fail
        slope, intercept = float(beta0[0]), float(beta0[1])
        if (not np.isfinite(slope) or abs(slope) < SLOPE_FLOOR
                or not np.all(np.isfinite(cov0))):
            return fail
        return {"ok": True, "slope": slope, "intercept": intercept,
                "cov": np.asarray(cov0, dtype=float), "method": "ols_regression"}

    # TLS : échelle x = σ_k (médian), y = pas d'énergie médian.
    sx = float(np.median(sk_arr))
    de = np.abs(np.diff(np.sort(e)))
    de = de[de > 0]
    sy = float(np.median(de)) if de.size else 1.0
    if not (np.isfinite(sx) and sx > 0 and np.isfinite(sy) and sy > 0):
        return fail

    slope, intercept = _tls_slope_intercept(k, e, sx, sy)
    if not (np.isfinite(slope) and np.isfinite(intercept)) or abs(slope) < SLOPE_FLOOR:
        return fail

    # Covariance des paramètres par jackknife leave-one-out (n petit).
    n = k.size
    if n >= 3:
        thetas = np.empty((n, 2), dtype=float)
        ok = True
        for i in range(n):
            ki = np.delete(k, i)
            ei = np.delete(e, i)
            si = np.delete(sk_arr, i)
            sxi = float(np.median(si))
            dei = np.abs(np.diff(np.sort(ei)))
            dei = dei[dei > 0]
            syi = float(np.median(dei)) if dei.size else sy
            s_i, b_i = _tls_slope_intercept(ki, ei, max(sxi, 1e-12), max(syi, 1e-12))
            if not (np.isfinite(s_i) and np.isfinite(b_i)):
                ok = False
                break
            thetas[i] = (s_i, b_i)
        if ok:
            mean = thetas.mean(axis=0)
            d = thetas - mean
            cov = (n - 1) / n * (d.T @ d)
        else:
            ok = False
    else:
        ok = False

    if not ok or not np.all(np.isfinite(cov)):
        return fail
    return {"ok": True, "slope": slope, "intercept": intercept,
            "cov": np.asarray(cov, dtype=float), "method": "orthogonal_tls"}
