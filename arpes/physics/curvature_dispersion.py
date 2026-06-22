"""Curvature-based dispersion extraction (Zhang et al. RSI 82, 043712 (2011)).

Tracks the maxima of the 1D momentum curvature C(k) at each energy slice to
produce an independent kF(E) estimate, as an alternative to Lorentzian MDC
fitting for difficult (broad / incoherent) bands.

Positions ONLY: the curvature transform distorts peak widths, so this must
never be used to extract Gamma / lifetimes (ImΣ needs the Lorentzian width of
the *raw* intensity). The result is meant as a cross-check overlay against the
MDC-fit dispersion: agreement validates kF, divergence flags the energy range
where the Lorentzian pair has merged and is unreliable.

Pure numpy/scipy — no PyQt.
"""
from __future__ import annotations

import numpy as np

from arpes.physics.plot_compute import _sigma_px, _smooth_masked


def momentum_curvature_1d(intensity, kpar, *, c0_alpha: float = 0.05) -> np.ndarray:
    """1D Zhang curvature of I(k): ``C = -I_kk / (C0 + I_k²)^{3/2}``.

    ``C0 = c0_alpha · (95th-percentile |I_k|)²`` regularizes against noise
    (Zhang's free parameter). A peak of I(k) becomes a maximum of C.
    """
    I = np.asarray(intensity, dtype=float)
    k = np.asarray(kpar, dtype=float)
    I_k = np.gradient(I, k)
    I_kk = np.gradient(I_k, k)
    g = np.abs(I_k)
    g = g[np.isfinite(g)]
    c0 = float(c0_alpha) * (float(np.percentile(g, 95)) ** 2 if g.size else 1.0)
    c0 = max(c0, 1e-30)
    return -I_kk / ((c0 + I_k ** 2) ** 1.5 + 1e-30)


def extract_curvature_dispersion(
    data, kpar, ev_arr, *,
    ev_start: float, ev_end: float,
    k_min=None, k_max=None, center_init: float = 0.0,
    n_pairs: int = 1, c0_alpha: float = 0.05,
    sigma_k_inv_a: float = 0.04, smooth_e_px: float = 1.0,
    min_prominence: float = 0.05,
) -> dict:
    """kF(E) per pair from curvature maxima, split ± around ``center_init``.

    ``data`` layout is ``[k, E]`` (axis 0 = k, axis 1 = E), matching the MDC fit
    pipeline. Returns a ``fit_result``-like dict so the dispersion plot reuses
    its existing consumers::

        {e_fitted, kF_minus, kF_plus, method:"curvature", c0_alpha, n_pairs}

    Width fields are intentionally absent (no Gamma from curvature). Within each
    energy slice, peaks are ranked by prominence and the strongest on each side
    of ``center_init`` is assigned to pair 0, the next to pair 1, etc.
    """
    from scipy.signal import find_peaks

    data = np.asarray(data, dtype=float)
    kpar = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev_arr, dtype=float)
    n_pairs = max(1, int(n_pairs))

    empty = {
        "e_fitted": [],
        "kF_minus": [[] for _ in range(n_pairs)],
        "kF_plus": [[] for _ in range(n_pairs)],
        "method": "curvature",
        "c0_alpha": float(c0_alpha),
        "n_pairs": n_pairs,
    }
    if data.ndim != 2 or kpar.size < 5 or ev_arr.size < 2:
        return empty

    sk_px = _sigma_px(kpar, float(sigma_k_inv_a), fallback=2.0)
    smooth, _mask = _smooth_masked(data, [sk_px, float(smooth_e_px)])

    k_lo = float(kpar.min()) if k_min is None else float(k_min)
    k_hi = float(kpar.max()) if k_max is None else float(k_max)
    kmask = (kpar >= k_lo) & (kpar <= k_hi)
    kw = kpar[kmask]
    if kw.size < 5:
        return empty

    e_lo, e_hi = min(ev_start, ev_end), max(ev_start, ev_end)
    e_idx = np.where((ev_arr >= e_lo) & (ev_arr <= e_hi))[0]
    e_idx = e_idx[np.argsort(ev_arr[e_idx])[::-1]]  # E_F side first
    if e_idx.size == 0:
        return empty

    dk = abs(float(np.median(np.diff(kpar)))) or 1.0
    min_dist = max(1, int(0.03 / dk))

    e_fitted: list[float] = []
    km_list: list[list[float]] = [[] for _ in range(n_pairs)]
    kp_list: list[list[float]] = [[] for _ in range(n_pairs)]

    for ie in e_idx:
        col = smooth[kmask, ie]
        rng = np.nanmax(col) - np.nanmin(col)
        if not np.isfinite(rng) or rng <= 0:
            continue
        curv = momentum_curvature_1d(col, kw, c0_alpha=c0_alpha)
        cn = curv - np.nanmin(curv)
        cn = cn / (np.nanmax(cn) + 1e-12)
        pk, props = find_peaks(cn, prominence=min_prominence, distance=min_dist)
        if pk.size == 0:
            continue
        kpk = kw[pk]
        prom = props.get("prominences", np.ones(pk.size))
        neg = sorted(
            ((float(kpk[j]), float(prom[j])) for j in range(pk.size) if kpk[j] < center_init),
            key=lambda t: -t[1],
        )
        pos = sorted(
            ((float(kpk[j]), float(prom[j])) for j in range(pk.size) if kpk[j] >= center_init),
            key=lambda t: -t[1],
        )
        e_fitted.append(float(ev_arr[ie]))
        for p in range(n_pairs):
            km_list[p].append(neg[p][0] if p < len(neg) else float("nan"))
            kp_list[p].append(pos[p][0] if p < len(pos) else float("nan"))

    return {
        "e_fitted": e_fitted,
        "kF_minus": km_list,
        "kF_plus": kp_list,
        "method": "curvature",
        "c0_alpha": float(c0_alpha),
        "n_pairs": n_pairs,
    }
