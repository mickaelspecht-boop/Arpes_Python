"""Model-curve helpers for PlotController diagnostics."""
from __future__ import annotations

import numpy as np

from arpes.physics.mdc_geometry import symmetric_peak_positions
from arpes.physics.plot_compute import _axis_signature as _full_axis_signature


def axis_cache_signature(axis) -> tuple:
    return _full_axis_signature(axis)


def lorentzian(k, k0, gamma, A):
    return A * gamma**2 / ((k - k0)**2 + gamma**2)


def pseudo_voigt(k, k0, gamma, A, eta=0.5):
    """Pseudo-Voigt preview with ``gamma`` interpreted as HWHM."""
    gamma = max(float(gamma), 1e-9)
    delta2 = (np.asarray(k, dtype=float) - float(k0)) ** 2
    lor = gamma**2 / (delta2 + gamma**2)
    gauss = np.exp(-np.log(2.0) * delta2 / gamma**2)
    eta = float(np.clip(eta, 0.0, 1.0))
    return float(A) * (eta * lor + (1.0 - eta) * gauss)


def build_model_pairs(k_arr, mdc, n_pairs, gamma_init,
                      k_min, k_max, center_init, smooth_sigma,
                      spacing=0.25, pair_params=None, shape="lorentzian"):
    """Build the live initial-model preview shown in the MDC Cut.

    When ``pair_params`` is supplied, peak positions and widths come directly
    from the right-hand controls. Detection remains only as a legacy fallback.
    This keeps the preview identical to the initial parameters sent to the
    fitter instead of silently replacing ``kF_init`` with detected peaks.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_w = k_arr[mask]
    m_w = mdc[mask]

    s_full = max(0.5, float(smooth_sigma))
    m_sm_full = gaussian_filter1d(np.nan_to_num(mdc.copy()), sigma=s_full)
    lo_f, hi_f = m_sm_full.min(), m_sm_full.max()
    mdc_smooth_norm = (m_sm_full - lo_f) / (hi_f - lo_f + 1e-12)

    if k_w.size < 10:
        return [], mdc_smooth_norm

    s = max(1, int(smooth_sigma))
    m_sm = gaussian_filter1d(np.nan_to_num(m_w), sigma=s)
    lo, hi = m_sm.min(), m_sm.max()
    supplied = list(pair_params or [])
    if hi - lo < 1e-10 and not supplied:
        return [], mdc_smooth_norm
    m_n = (m_sm - lo) / (hi - lo + 1e-12)
    bg = float(np.nanpercentile(m_sm, 10))
    A0 = max(float(hi - lo), 1.0)

    pks, _ = find_peaks(m_n, height=0.10, distance=max(3, s))
    if len(pks):
        pks = pks[np.argsort(m_n[pks])[::-1]]

    params = []
    for i, pp in enumerate(supplied[:n_pairs]):
        kf = abs(float(pp.get("kF_init", spacing * (i + 1))))
        gamma = max(float(pp.get("gamma_init", gamma_init)), 1e-9)
        km, kp = symmetric_peak_positions(center_init, kf)
        near = np.argsort(np.minimum(abs(k_w - km), abs(k_w - kp)))[:2]
        amplitude = max(
            float(np.nanmax(m_sm[near]) - bg) if near.size else A0,
            0.15 * A0,
        )
        params.append((km, kp, amplitude, gamma))

    if not supplied:
        if len(pks) >= 2:
            k_pks = k_w[pks]
            A_pks = m_sm[pks] - bg
            pos = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp >= center_init]
            neg = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp < center_init]
            for i in range(min(n_pairs, max(len(pos), len(neg)))):
                km = neg[i][0] if i < len(neg) else center_init - spacing * (i + 1)
                kp = pos[i][0] if i < len(pos) else center_init + spacing * (i + 1)
                params.append((km, kp, A0, float(gamma_init)))
        elif len(pks) == 1:
            k0 = float(k_w[pks[0]])
            d = abs(k0 - center_init)
            params.append((center_init - d, center_init + d, A0, float(gamma_init)))

    while len(params) < n_pairs:
        i = len(params)
        params.append((center_init - spacing * (i + 1),
                       center_init + spacing * (i + 1), A0 * 0.6,
                       float(gamma_init)))

    pairs = []
    peak_fn = pseudo_voigt if str(shape) == "voigt" else lorentzian
    for km, kp, A, gamma in params[:n_pairs]:
        cl = peak_fn(k_arr, km, gamma, A)
        cr = peak_fn(k_arr, kp, gamma, A)
        raw = cl + cr + bg
        raw_lo = float(np.nanmin(raw))
        raw_span = float(np.nanmax(raw) - raw_lo) + 1e-12
        curve = (raw - raw_lo) / raw_span
        cln = cl / raw_span
        crn = cr / raw_span
        pairs.append((curve, km, kp, cln, crn))
    return pairs, mdc_smooth_norm
