"""Model-curve helpers for PlotController diagnostics."""
from __future__ import annotations

import numpy as np

from arpes.physics.plot_compute import _axis_signature as _full_axis_signature


def axis_cache_signature(axis) -> tuple:
    return _full_axis_signature(axis)


def lorentzian(k, k0, gamma, A):
    return A * gamma**2 / ((k - k0)**2 + gamma**2)


def build_model_pairs(k_arr, mdc, n_pairs, gamma_init,
                      k_min, k_max, center_init, smooth_sigma,
                      spacing=0.25):
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
    if hi - lo < 1e-10:
        return [], mdc_smooth_norm
    m_n = (m_sm - lo) / (hi - lo)
    bg = float(np.nanpercentile(m_sm, 10))
    A0 = float(hi - lo)

    pks, _ = find_peaks(m_n, height=0.10, distance=max(3, s))
    if len(pks):
        pks = pks[np.argsort(m_n[pks])[::-1]]

    params = []
    if len(pks) >= 2:
        k_pks = k_w[pks]
        A_pks = m_sm[pks] - bg
        pos = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp >= center_init]
        neg = [(kp, ap) for kp, ap in zip(k_pks, A_pks) if kp < center_init]
        for i in range(min(n_pairs, max(len(pos), len(neg)))):
            km = neg[i][0] if i < len(neg) else center_init - spacing * (i + 1)
            kp = pos[i][0] if i < len(pos) else center_init + spacing * (i + 1)
            params.append((km, kp, A0))
    elif len(pks) == 1:
        k0 = float(k_w[pks[0]])
        d = abs(k0 - center_init)
        params.append((center_init - d, center_init + d, A0))

    while len(params) < n_pairs:
        i = len(params)
        params.append((center_init - spacing * (i + 1),
                       center_init + spacing * (i + 1), A0 * 0.6))

    pairs = []
    for km, kp, A in params[:n_pairs]:
        cl = lorentzian(k_arr, km, gamma_init, A)
        cr = lorentzian(k_arr, kp, gamma_init, A)
        curve = cl + cr + bg
        curve = (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        cln = (cl - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        crn = (cr - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)
        pairs.append((curve, km, kp, cln, crn))
    return pairs, mdc_smooth_norm
