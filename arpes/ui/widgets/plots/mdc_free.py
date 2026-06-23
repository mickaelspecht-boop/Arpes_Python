"""Free-peak / non-symmetric MDC fitting (split from mdc_fit.py, 700-LOC cap).

width_mode="free": 2N independent Lorentzian peaks for non-Γ-symmetric cuts
(Igor "Independent Parameters" equivalent). Same result schema as peak-pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import (
    _lor_peak,
    _resolution_correct_gamma,
    _voigt_pseudo,
)


def _fit_mdc_free_peaks(
    data_cut, kpar, ev_arr,
    *,
    n_pairs=1,
    ev_start=-0.15,
    ev_end=-0.01,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.20,
    kF_init=None,
    center_init=0.0,
    min_amplitude=0.02,
    max_jump=0.15,
    mdc_energy_window=0.0,
    scan_direction="down",
    k_min=None,
    k_max=None,
    dE_eV=0.0,
    dk_inv_a=0.0,
    resolution_source="",
    shape="lorentzian",
    hold_gamma=False,
    verbose=False,
):
    """Fit 2*N independent MDC peaks, then expose them as left/right branches.

    This is the rigorous equivalent of Igor's "Independent Parameters" for
    non-Γ-symmetric cuts. It keeps the same result schema as peak-pairs so
    Results/export/lifetime diagnostics still work. Pair ``i`` is ordered by
    distance from ``center_init``: inner pair first, then outer pairs.
    """
    n_pairs = max(1, int(n_pairs))
    n_peaks = 2 * n_pairs
    is_voigt = (shape == "voigt")
    I_fit = gaussian_filter1d(data_cut.astype(float), sigma=smooth_fit, axis=0)
    I_detect = gaussian_filter1d(data_cut.astype(float), sigma=smooth_detect, axis=0)

    k_lo_fit = k_min if k_min is not None else float(kpar[0])
    k_hi_fit = k_max if k_max is not None else float(kpar[-1])
    k_mask = (kpar >= k_lo_fit) & (kpar <= k_hi_fit)
    kpar_fit = kpar[k_mask]
    if kpar_fit.size < max(8, 3 * n_peaks + 2):
        return _empty_free_result(I_fit, kpar, ev_arr, kpar_fit, n_pairs, shape, resolution_source, dE_eV, dk_inv_a)

    gamma_floor = max(1e-9, float(gamma_init) - max(abs(float(gamma_init)) * 1e-6, 1e-7)) if hold_gamma else min(max(0.001, float(dk_inv_a or 0.0)), float(gamma_max) * 0.95)
    gamma_upper = float(gamma_init) + max(abs(float(gamma_init)) * 1e-6, 1e-7) if hold_gamma else float(gamma_max)

    def model(k, *p):
        p = np.asarray(p, dtype=float)
        y = p[0] * k + p[1]
        eta = float(p[-1]) if is_voigt else 0.0
        for j in range(n_peaks):
            base = 2 + 3 * j
            x0, amp, gam = p[base], p[base + 1], p[base + 2]
            if is_voigt:
                y = y + _voigt_pseudo(k, x0, amp, gam, eta)
            else:
                y = y + _lor_peak(k, x0, amp, gam)
        return y

    lo = [-np.inf, -np.inf]
    hi = [np.inf, np.inf]
    for _ in range(n_peaks):
        lo += [float(kpar_fit[0]), 0.0, gamma_floor]
        hi += [float(kpar_fit[-1]), np.inf, gamma_upper]
    if is_voigt:
        lo += [0.0]
        hi += [1.0]

    ev_lo = min(ev_start, ev_end)
    ev_hi = max(ev_start, ev_end)
    wf_indices = np.where((ev_arr >= ev_lo) & (ev_arr <= ev_hi))[0]
    wf_energies_all = ev_arr[wf_indices]
    wf_energies = np.sort(wf_energies_all)[::-1] if scan_direction == "down" else np.sort(wf_energies_all)
    ev_for_init = ev_hi if scan_direction == "down" else ev_lo

    def initial_positions(ie_init: int) -> list[float]:
        mdc_init = I_detect[k_mask, ie_init]
        mmax = float(np.nanmax(mdc_init)) if mdc_init.size else 0.0
        mdc_n = mdc_init / mmax if mmax > 0 else mdc_init
        dk = abs(float(kpar_fit[1] - kpar_fit[0])) if kpar_fit.size > 1 else 0.01
        pk_idx, props = find_peaks(mdc_n, prominence=0.05, distance=max(1, int(0.04 / dk)))
        if len(pk_idx) >= n_peaks:
            top = pk_idx[np.argsort(props["prominences"])[-n_peaks:]]
            return sorted(float(kpar_fit[j]) for j in top)
        kF_seed = [] if kF_init is None else list(np.ravel(kF_init))
        if kF_seed:
            vals = []
            for v in kF_seed[:n_pairs]:
                vals.extend([center_init - abs(float(v)), center_init + abs(float(v))])
            if len(vals) >= n_peaks:
                return sorted(vals[:n_peaks])
        return list(np.linspace(float(kpar_fit[0]) * 0.8, float(kpar_fit[-1]) * 0.8, n_peaks))

    kF_minus_list = [[] for _ in range(n_pairs)]
    kF_plus_list = [[] for _ in range(n_pairs)]
    gamma_list = [[] for _ in range(n_pairs)]
    sigma_kF_minus_list = [[] for _ in range(n_pairs)]
    sigma_kF_plus_list = [[] for _ in range(n_pairs)]
    sigma_gamma_list = [[] for _ in range(n_pairs)]
    k0_list = [[] for _ in range(n_pairs)]
    xg_list = []
    e_fitted = []
    fit_curve_list = []
    fit_bg_list = []  # fond linéaire par tranche (post-fit viewer)
    residual_list = []
    chi2_list = []
    eta_list = []
    prev_popt = None
    prev_positions = None

    ie_init = int(np.argmin(np.abs(ev_arr - ev_for_init))) if len(ev_arr) else 0
    seed_positions = initial_positions(ie_init)

    def make_p0(positions, mdc_n):
        p = [0.0, 0.05]
        for kg in sorted(positions):
            amp = float(np.interp(kg, kpar_fit, mdc_n))
            p += [float(kg), max(amp, 0.05), float(gamma_init)]
        if is_voigt:
            p += [0.5]
        return p

    def order_into_pairs(positions, gammas, sig_pos, sig_gamma):
        left = sorted([(k, g, sk, sg) for k, g, sk, sg in zip(positions, gammas, sig_pos, sig_gamma) if k <= center_init], key=lambda t: abs(t[0] - center_init))
        right = sorted([(k, g, sk, sg) for k, g, sk, sg in zip(positions, gammas, sig_pos, sig_gamma) if k > center_init], key=lambda t: abs(t[0] - center_init))
        while len(left) < n_pairs:
            left.append((np.nan, np.nan, np.nan, np.nan))
        while len(right) < n_pairs:
            right.append((np.nan, np.nan, np.nan, np.nan))
        return left[:n_pairs], right[:n_pairs]

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        if mdc_energy_window > 0:
            e_mask = np.abs(ev_arr - ev_arr[ie]) <= 0.5 * float(mdc_energy_window)
            block = I_fit[:, e_mask]
            mdc_full = np.nanmean(block, axis=1) if block.shape[1] else I_fit[:, ie]
        else:
            mdc_full = I_fit[:, ie]
        mdc = mdc_full[k_mask]
        mx = float(np.nanmax(mdc)) if mdc.size else 0.0
        if mx <= 0:
            continue
        mdc_n = mdc / mx
        p0 = (
            list(prev_popt)
            if prev_popt is not None
            else make_p0(seed_positions if prev_positions is None else prev_positions, mdc_n)
        )
        try:
            popt, pcov = curve_fit(model, kpar_fit, mdc_n, p0=p0, bounds=(lo, hi), maxfev=10000)
            sigma_full = np.sqrt(np.abs(np.diag(pcov)))
            fit_y = model(kpar_fit, *popt)
            bg_y = popt[0] * kpar_fit + popt[1]  # linear background of this slice
            residual_y = mdc_n - fit_y
            dof = max(1, int(kpar_fit.size) - int(len(popt)))
            chi2_red = float(np.nansum(residual_y ** 2) / dof)
            peaks = []
            for j in range(n_peaks):
                base = 2 + 3 * j
                kfit, amp, gfit = float(popt[base]), float(popt[base + 1]), float(popt[base + 2])
                sk, sg = float(sigma_full[base]), float(sigma_full[base + 2])
                if amp <= min_amplitude:
                    peaks.append((np.nan, np.nan, np.nan, np.nan))
                else:
                    peaks.append((kfit, gfit, sk, sg))
            positions = [p[0] for p in peaks]
            finite_positions = [p for p in positions if np.isfinite(p)]
            jumped = False
            if prev_positions is not None and len(finite_positions) == len(prev_positions):
                if max(abs(a - b) for a, b in zip(sorted(finite_positions), sorted(prev_positions))) > max_jump:
                    jumped = True
            if jumped:
                prev_popt = None
                for i in range(n_pairs):
                    kF_minus_list[i].append(np.nan); kF_plus_list[i].append(np.nan)
                    gamma_list[i].append(np.nan); sigma_gamma_list[i].append(np.nan)
                    sigma_kF_minus_list[i].append(np.nan); sigma_kF_plus_list[i].append(np.nan)
                    k0_list[i].append(np.nan)
                xg_list.append(np.nan)
            else:
                gammas = [p[1] for p in peaks]
                sig_pos = [p[2] for p in peaks]
                sig_gamma = [p[3] for p in peaks]
                left, right = order_into_pairs(positions, gammas, sig_pos, sig_gamma)
                for i in range(n_pairs):
                    km, gm, skm, sgm = left[i]
                    kp, gp, skp, sgp = right[i]
                    kF_minus_list[i].append(km)
                    kF_plus_list[i].append(kp)
                    sigma_kF_minus_list[i].append(skm)
                    sigma_kF_plus_list[i].append(skp)
                    gamma_pair = np.asarray([gm, gp], dtype=float)
                    gamma_list[i].append(float(np.nanmean(gamma_pair)) if np.isfinite(gamma_pair).any() else np.nan)
                    sig_pair = np.asarray([sgm, sgp], dtype=float)
                    sigma_gamma_list[i].append(
                        0.5 * float(np.sqrt(np.nansum(sig_pair ** 2)))
                        if np.isfinite(sig_pair).any() else np.nan
                    )
                    k0_list[i].append(0.5 * abs(kp - km) if np.isfinite(km) and np.isfinite(kp) else np.nan)
                xg_list.append(float(np.nanmean(finite_positions)) if finite_positions else np.nan)
                prev_popt = popt
                prev_positions = finite_positions
            e_fitted.append(ev_arr[ie])
            fit_curve_list.append(fit_y)
            fit_bg_list.append(bg_y)
            residual_list.append(residual_y)
            chi2_list.append(chi2_red)
            eta_list.append(float(popt[-1]) if is_voigt else float("nan"))
        except Exception as exc:
            if verbose:
                print(f"free MDC fit failed at E={ev_arr[ie]:+.3f}: {exc}")
            prev_popt = None
            for i in range(n_pairs):
                kF_minus_list[i].append(np.nan); kF_plus_list[i].append(np.nan)
                gamma_list[i].append(np.nan); sigma_gamma_list[i].append(np.nan)
                sigma_kF_minus_list[i].append(np.nan); sigma_kF_plus_list[i].append(np.nan)
                k0_list[i].append(np.nan)
            xg_list.append(np.nan)
            e_fitted.append(ev_arr[ie])
            fit_curve_list.append(np.full_like(kpar_fit, np.nan, dtype=float))
            fit_bg_list.append(np.full_like(kpar_fit, np.nan, dtype=float))
            residual_list.append(np.full_like(kpar_fit, np.nan, dtype=float))
            chi2_list.append(np.nan)
            eta_list.append(np.nan)

    e_arr_out = np.asarray(e_fitted, dtype=float)
    sort_idx = np.argsort(e_arr_out)
    k0_out = [np.asarray(x, dtype=float)[sort_idx] for x in k0_list]
    gamma_brut = [np.asarray(x, dtype=float)[sort_idx] for x in gamma_list]
    gamma_min = []
    gamma_corrige = []
    for i in range(n_pairs):
        gmin, gcorr = _resolution_correct_gamma(
            e_arr_out[sort_idx], k0_out[i], gamma_brut[i],
            dE_eV=dE_eV, dk_inv_a=dk_inv_a,
        )
        gamma_min.append(gmin)
        gamma_corrige.append(gcorr)
    return dict(
        kF_minus=[np.asarray(x, dtype=float)[sort_idx] for x in kF_minus_list],
        kF_plus=[np.asarray(x, dtype=float)[sort_idx] for x in kF_plus_list],
        sigma_kF_minus=[np.asarray(x, dtype=float)[sort_idx] for x in sigma_kF_minus_list],
        sigma_kF_plus=[np.asarray(x, dtype=float)[sort_idx] for x in sigma_kF_plus_list],
        sigma_gamma=[np.asarray(x, dtype=float)[sort_idx] for x in sigma_gamma_list],
        k0=k0_out,
        xg=np.asarray(xg_list, dtype=float)[sort_idx],
        e_fitted=e_arr_out[sort_idx],
        I_smoothed=I_fit,
        fit_kpar=kpar_fit,
        fit_curves=[np.asarray(x, dtype=float) for x in np.asarray(fit_curve_list, dtype=float)[sort_idx]],
        fit_bg=[np.asarray(x, dtype=float) for x in np.asarray(fit_bg_list, dtype=float)[sort_idx]],
        residuals=[np.asarray(x, dtype=float) for x in np.asarray(residual_list, dtype=float)[sort_idx]],
        chi2_red=np.asarray(chi2_list, dtype=float)[sort_idx],
        kpar=kpar,
        ev_arr=ev_arr,
        n_pairs=n_pairs,
        width_mode="free",
        shape=shape,
        width_convention="HWHM",
        gamma_units="pi/a HWHM",
        fit_constraints={"hold_center": False, "hold_gamma": bool(hold_gamma)},
        eta=np.asarray(eta_list, dtype=float)[sort_idx] if len(eta_list) == len(sort_idx) else np.asarray(eta_list, dtype=float),
        gamma=gamma_brut,
        gamma_brut=gamma_brut,
        gamma_min=gamma_min,
        gamma_corrige=gamma_corrige,
        resolution={
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
    )


def _empty_free_result(I_fit, kpar, ev_arr, kpar_fit, n_pairs, shape, resolution_source, dE_eV, dk_inv_a):
    return {
        "kF_minus": [np.array([]) for _ in range(n_pairs)],
        "kF_plus": [np.array([]) for _ in range(n_pairs)],
        "sigma_kF_minus": [np.array([]) for _ in range(n_pairs)],
        "sigma_kF_plus": [np.array([]) for _ in range(n_pairs)],
        "sigma_gamma": [np.array([]) for _ in range(n_pairs)],
        "k0": [np.array([]) for _ in range(n_pairs)],
        "xg": np.array([]),
        "e_fitted": np.array([]),
        "I_smoothed": I_fit,
        "fit_kpar": kpar_fit,
        "fit_curves": [],
        "fit_bg": [],
        "residuals": [],
        "chi2_red": np.array([]),
        "kpar": kpar,
        "ev_arr": ev_arr,
        "n_pairs": n_pairs,
        "width_mode": "free",
        "shape": shape,
        "width_convention": "HWHM",
        "gamma_units": "pi/a HWHM",
        "fit_constraints": {"hold_center": False, "hold_gamma": False},
        "eta": np.array([]),
        "gamma": [np.array([]) for _ in range(n_pairs)],
        "gamma_brut": [np.array([]) for _ in range(n_pairs)],
        "gamma_min": [np.array([]) for _ in range(n_pairs)],
        "gamma_corrige": [np.array([]) for _ in range(n_pairs)],
        "resolution": {
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
    }

