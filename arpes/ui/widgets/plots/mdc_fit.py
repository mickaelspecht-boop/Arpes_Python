"""Peak-pair MDC fitting routines."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import (
    _lor_peak,
    _make_peak_pairs_model,
    _normalize_width_mode,
    _resolution_correct_gamma,
    _voigt_pseudo,
)

def fit_mdc_peak_pairs(
    data_cut, kpar, ev_arr,
    n_pairs=1,
    ev_start=-0.15, ev_end=-0.01,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.20,
    kF_init=None,
    center_init=0.0,
    xg_range=0.12,
    min_amplitude=0.02,
    max_jump=0.15,
    mdc_energy_window=0.0,
    scan_direction='down',
    width_mode='symmetric',
    k_min=None,
    k_max=None,
    k0_max=None,
    dE_eV=0.0,
    dk_inv_a=0.0,
    resolution_source="",
    shape='lorentzian',
    hold_center=False,
    hold_gamma=False,
    verbose=False,
):
    """
    Fit MDCs with symmetric Lorentzian peak pairs (Igor Peak Pairs method).

    Each pair produces two symmetric peaks around an xg center (about Gamma).
    More stable than independent fits when the bands are symmetric.

    Parameters
    ----------
    data_cut    : np.ndarray (nk, ne)
    kpar, ev_arr: np.ndarray 1D
    n_pairs     : int, number of pairs (1 pair = 2 symmetric kF values)
    ev_start/end: float, energy window
    smooth_fit  : float, smoothing for the fit
    smooth_detect: float, smoothing for initial detection
    gamma_init  : float, initial width (eV)
    gamma_max   : float, maximum width (eV)
    kF_init     : list of float or None, initial k0 positions (one per pair).
                  IMPORTANT: must match the peak position at the scan start
                  energy (ev_end if scan_direction='down', ev_start otherwise).
                  Read from the waterfall at that energy. If None, auto-detect.
    center_init : float, initial xg position (default 0 = Gamma)
    xg_range    : float, half-window constraint around center_init.
                  xg is constrained to [center_init-xg_range, center_init+xg_range].
                  Prevents center drift. Default 0.12 pi/a.
    min_amplitude: float, amplitude threshold for accepting a peak (default 0.02).
                  Lower it if one side is suppressed by matrix-element effects.
    max_jump    : float, threshold (pi/a) above which a k0 jump is rejected.
                  Increase it for fast dispersion (strongly curved band).
    scan_direction: 'down' (default) | 'up'
                  'down': scan from ev_end to ev_start (from EF to lower E).
                  kF_init must match the EF position.
                  'up': scan from ev_start to ev_end (increasing energy).
                  kF_init must match the ev_start position.
                  For a hole band centered at Gamma, use 'down'.
    width_mode  : 'independent' | 'symmetric' | 'global'
    k_min, k_max: float or None, k range used for the fit (Igor cursor equivalent).
                  CRUCIAL: restrict near the peaks of interest to avoid locking
                  onto brighter features outside the region.
                  Example: k_min=-0.6, k_max=0.6 for an alpha pocket around Gamma.
                  If None, the full k axis is used.

    Returns
    --------
    dict with keys:
        'kF_minus' : list of n_pairs arrays, left peak position (-k0+xg)
        'kF_plus'  : list of n_pairs arrays, right peak position (+k0+xg)
        'k0'       : list of n_pairs arrays, k0 half-separation
        'xg'       : array, Gamma center at each energy
        'e_fitted' : array of converged energies
        'I_smoothed': array smoothed over the full k axis (for display)
        'kpar', 'ev_arr': axes
    """
    is_voigt = (shape == 'voigt')
    # Normalize historical aliases (UI 'asymmetric' -> backend 'independent').
    width_mode = _normalize_width_mode(width_mode)
    if width_mode == "free":
        return _fit_mdc_free_peaks(
            data_cut, kpar, ev_arr,
            n_pairs=n_pairs,
            ev_start=ev_start, ev_end=ev_end,
            smooth_fit=smooth_fit, smooth_detect=smooth_detect,
            gamma_init=gamma_init, gamma_max=gamma_max,
            kF_init=kF_init, center_init=center_init,
            min_amplitude=min_amplitude, max_jump=max_jump,
            mdc_energy_window=mdc_energy_window,
            scan_direction=scan_direction,
            k_min=k_min, k_max=k_max,
            dE_eV=dE_eV, dk_inv_a=dk_inv_a,
            resolution_source=resolution_source,
            shape=shape,
            hold_gamma=hold_gamma,
            verbose=verbose,
        )
    model, n_pp, n_extra = _make_peak_pairs_model(n_pairs, width_mode, shape=shape)

    I_fit    = gaussian_filter1d(data_cut.astype(float), sigma=smooth_fit,    axis=0)
    I_detect = gaussian_filter1d(data_cut.astype(float), sigma=smooth_detect, axis=0)

    # Masque k pour restreindre la plage de fit (équivalent curseurs Igor)
    k_lo_fit = k_min if k_min is not None else float(kpar[0])
    k_hi_fit = k_max if k_max is not None else float(kpar[-1])
    k_mask   = (kpar >= k_lo_fit) & (kpar <= k_hi_fit)
    kpar_fit = kpar[k_mask]   # axe k restreint pour le fit

    # Bornes — xg contraint autour de center_init pour éviter la dérive
    dk = abs(float(kpar[1] - kpar[0]))
    k0_lo = max(dk * 2, 0.02)   # k0 > 0 strict pour éviter la solution dégénérée k0→0
    k0_hi_auto = (k_hi_fit - center_init) * 0.95  # k0 borné par la plage de fit

    # k0_max optionnel : borne supérieure explicite par paire
    # ex: k0_max=[0.40, 0.65] pour forcer paire1 < 0.40 et paire2 < 0.65
    # Si None → borne automatique depuis k_max (k0_hi_auto)
    if k0_max is None:
        k0_hi_list = [k0_hi_auto] * n_pairs
    elif np.isscalar(k0_max):
        k0_hi_list = [float(k0_max)] * n_pairs
    else:
        k0_hi_list = list(k0_max)
        while len(k0_hi_list) < n_pairs:
            k0_hi_list.append(k0_hi_auto)

    if hold_center:
        xg_eps = max(abs(float(center_init)) * 1e-9, 1e-9)
        xg_lo, xg_hi = center_init - xg_eps, center_init + xg_eps
    else:
        xg_lo, xg_hi = center_init - xg_range, center_init + xg_range
    lo = [-np.inf, -np.inf, xg_lo]   # bg_a, bg_b, xg
    hi = [ np.inf,  np.inf, xg_hi]
    if hold_gamma:
        gamma_eps = max(abs(float(gamma_init)) * 1e-6, 1e-7)
        gamma_floor = max(1e-9, float(gamma_init) - gamma_eps)
        gamma_upper = float(gamma_init) + gamma_eps
    else:
        gamma_floor = min(max(0.001, float(dk_inv_a or 0.0)), float(gamma_max) * 0.95)
        gamma_upper = float(gamma_max)
    for pi in range(n_pairs):
        k0_hi = k0_hi_list[pi]
        if width_mode == 'independent':
            lo += [k0_lo,  0.0, gamma_floor, 0.0, gamma_floor]
            hi += [k0_hi,  np.inf, gamma_upper, np.inf, gamma_upper]
        elif width_mode == 'symmetric':
            lo += [k0_lo,  0.0,  0.0, gamma_floor]
            hi += [k0_hi,  np.inf, np.inf, gamma_upper]
        else:  # global
            lo += [k0_lo,  0.0,  0.0]
            hi += [k0_hi,  np.inf, np.inf]
    if width_mode == 'global':
        lo += [gamma_floor]; hi += [gamma_upper]
    # Voigt : paramètre η_global ∈ [0,1] appended à la fin de p
    if is_voigt:
        lo += [0.0]; hi += [1.0]

    # Direction du scan
    # 'down' : de ev_end (proche EF) vers ev_start (basses E)
    # kF_init doit correspondre aux pics à ev_end (EF side)
    # 'up'   : de ev_start vers ev_end (ordre croissant)
    # kF_init doit correspondre aux pics à ev_start
    #
    # NOTE : on sélectionne directement les indices de ev_arr dans la fenêtre
    # [ev_start, ev_end] pour être robuste à l'ordre de ev_arr (ascendant OU
    # descendant). np.arange avec ev_step < 0 donnait un tableau vide si ev_arr
    # était en ordre décroissant — bug corrigé.
    ev_lo = min(ev_start, ev_end)
    ev_hi = max(ev_start, ev_end)
    wf_indices  = np.where((ev_arr >= ev_lo) & (ev_arr <= ev_hi))[0]
    wf_energies_all = ev_arr[wf_indices]   # dans l'ordre de ev_arr
    if scan_direction == 'down':
        # Scan du côté EF (ev_hi) vers les basses E (ev_lo)
        wf_energies = np.sort(wf_energies_all)[::-1]
        ev_for_init = ev_hi               # kF_init lu au côté EF
    else:
        # Scan ascendant : ev_lo → ev_hi
        wf_energies = np.sort(wf_energies_all)
        ev_for_init = ev_lo

    print(f'fit_mdc_peak_pairs: {len(wf_energies)} slices in [{ev_lo:.3f}, {ev_hi:.3f}] eV '
          f'(scan {scan_direction!r} from ev_for_init={ev_for_init:.3f} eV)')
    if len(wf_energies) == 0:
        import warnings
        warnings.warn(
            'fit_mdc_peak_pairs: no energy found in the window '
            f'[{ev_lo:.3f}, {ev_hi:.3f}] eV. '
            f'Check ev_arr (range [{float(ev_arr[0]):.3f}, {float(ev_arr[-1]):.3f}] eV), '
            'WF_EV_START et WF_EV_END.', stacklevel=2)

    # Guess initial k0 — auto-détection dans la plage k restreinte à ev_for_init
    if kF_init is None:
        ie_init = int(np.argmin(np.abs(ev_arr - ev_for_init)))
        mdc_init = I_detect[k_mask, ie_init]
        mdc_init_n = mdc_init / (mdc_init.max() or 1)
        pk_idx, _ = find_peaks(mdc_init_n, prominence=0.05,
                               distance=max(1, int(0.05 / dk)))
        pos_k = kpar_fit[pk_idx]
        pos_k_pos = pos_k[pos_k > center_init]
        if len(pos_k_pos) >= n_pairs:
            kF_init = sorted(pos_k_pos)[:n_pairs]
        else:
            # Fallback : pic le plus intense côté k>0 (sans critère de prominence)
            mdc_pos_mask = kpar_fit > center_init
            if mdc_pos_mask.any():
                # Diviser la plage k+ en n_pairs segments et prendre le max de chacun
                kpar_pos = kpar_fit[mdc_pos_mask]
                mdc_pos  = mdc_init_n[mdc_pos_mask]
                segments = np.array_split(np.arange(len(kpar_pos)), n_pairs)
                kF_init  = [float(kpar_pos[seg[np.argmax(mdc_pos[seg])]]) for seg in segments]
            else:
                # Dernier recours : valeurs équiréparties dans la plage k0
                kF_init = list(np.linspace(k0_hi_auto * 0.3, k0_hi_auto * 0.8, n_pairs))
        if verbose:
            print(f'  Auto-detected kF_init at E={ev_arr[ie_init]:+.3f} eV -> {[f"{k:.3f}" for k in kF_init]}'
                  f'  ({"peaks found" if len(pk_idx) >= n_pairs else "max-intensity fallback"})')

    kF_minus_list = [[] for _ in range(n_pairs)]
    kF_plus_list  = [[] for _ in range(n_pairs)]
    k0_list       = [[] for _ in range(n_pairs)]
    gamma_list    = [[] for _ in range(n_pairs)]
    gamma_left_list  = [[] for _ in range(n_pairs)]  # γL (côté kF-)
    gamma_right_list = [[] for _ in range(n_pairs)]  # γR (côté kF+)
    sigma_gamma_left_list  = [[] for _ in range(n_pairs)]
    sigma_gamma_right_list = [[] for _ in range(n_pairs)]
    sigma_kF_minus_list = [[] for _ in range(n_pairs)]
    sigma_kF_plus_list  = [[] for _ in range(n_pairs)]
    sigma_gamma_list    = [[] for _ in range(n_pairs)]
    xg_list       = []
    e_fitted      = []
    fit_curve_list = []
    residual_list = []
    chi2_list = []
    eta_list = []  # paramètre η_global du pseudo-Voigt (NaN si shape=lorentzian)
    prev_popt     = None
    prev_k0       = list(kF_init)   # derniers k0 valides pour détecter les sauts

    def _make_p0(kF_list):
        """Construit le vecteur p0 depuis une liste de k0 (un par paire)."""
        p = [0.0, float(np.percentile(mdc_n, 10)), center_init]
        for k0g in kF_list:
            A_g = float(np.interp(abs(k0g) + center_init, kpar_fit, mdc_n))
            A_g = max(A_g, 0.05)
            if width_mode == 'independent':
                p += [abs(k0g), A_g, gamma_init, A_g, gamma_init]
            elif width_mode == 'symmetric':
                p += [abs(k0g), A_g, A_g, gamma_init]
            else:
                p += [abs(k0g), A_g, A_g]
        if width_mode == 'global':
            p += [gamma_init]
        if is_voigt:
            p += [0.5]
        return p

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        # MDC restreinte à la plage [k_min, k_max] — comme les curseurs Igor.
        # mdc_energy_window > 0 : intègre ±window/2 en énergie autour de ev_i
        # (moyenne des lignes) pour réduire le bruit qui fait serpenter kF(E).
        # kF/Γ varient lentement en E → non biaisés tant que la fenêtre reste
        # petite devant l'échelle de dispersion.
        if mdc_energy_window > 0:
            e_mask = np.abs(ev_arr - ev_arr[ie]) <= 0.5 * float(mdc_energy_window)
            block = I_fit[:, e_mask]
            mdc_full = np.nanmean(block, axis=1) if block.shape[1] else I_fit[:, ie]
        else:
            mdc_full = I_fit[:, ie]
        mdc      = mdc_full[k_mask]
        mx = mdc.max()
        if mx <= 0:
            if verbose:
                print(f'  E={ev_arr[ie]:+.3f} eV: MDC max=0, slice ignored')
            continue
        mdc_n = mdc / mx

        if prev_popt is not None:
            p0 = list(prev_popt)
        else:
            p0 = _make_p0(prev_k0)

        try:
            popt, pcov = curve_fit(model, kpar_fit, mdc_n, p0=p0,
                                bounds=(lo, hi), maxfev=8000)
            eta_list.append(float(popt[-1]) if is_voigt else float("nan"))
            fit_y = model(kpar_fit, *popt)
            residual_y = mdc_n - fit_y
            dof = max(1, int(kpar_fit.size) - int(len(popt)))
            chi2_red = float(np.nansum(residual_y ** 2) / dof)
            sigma_full = np.sqrt(np.abs(np.diag(pcov)))
            sigma_xg = float(sigma_full[2])
            xg_fit = popt[2]
            converged = False
            jumped   = False

            for i in range(n_pairs):
                k0_fit = popt[3 + n_pp*i]
                if width_mode == 'independent':
                    A1 = popt[3 + n_pp*i + 1]
                    A2 = popt[3 + n_pp*i + 3]
                elif width_mode == 'symmetric':
                    A1 = popt[3 + n_pp*i + 1]
                    A2 = popt[3 + n_pp*i + 2]
                else:
                    A1 = popt[3 + n_pp*i + 1]
                    A2 = popt[3 + n_pp*i + 2]

                # Jump detection: reject k0 if it deviates too far from the last valid k0.
                if abs(k0_fit - prev_k0[i]) > max_jump:
                    jumped = True

                km = (-k0_fit + xg_fit) if A1 > min_amplitude else np.nan
                kp = (+k0_fit + xg_fit) if A2 > min_amplitude else np.nan
                sigma_k0 = float(sigma_full[3 + n_pp * i])
                sigma_kF = float(np.sqrt(sigma_k0 ** 2 + sigma_xg ** 2))
                kF_minus_list[i].append(km if not jumped else np.nan)
                kF_plus_list[i].append(kp if not jumped else np.nan)
                sigma_kF_minus_list[i].append(
                    sigma_kF if (np.isfinite(km) and not jumped) else np.nan
                )
                sigma_kF_plus_list[i].append(
                    sigma_kF if (np.isfinite(kp) and not jumped) else np.nan
                )
                k0_list[i].append(
                    k0_fit if (not jumped and (A1>min_amplitude or A2>min_amplitude))
                    else np.nan
                )
                if (np.isfinite(km) or np.isfinite(kp)) and not jumped:
                    converged = True

                if width_mode == 'independent':
                    g1 = float(popt[3 + n_pp*i + 2])  # côté gauche (kF-)
                    g2 = float(popt[3 + n_pp*i + 4])  # côté droit (kF+)
                    gamma_fit = float(np.nanmean([g1, g2]))
                    sg1 = float(sigma_full[3 + n_pp * i + 2])
                    sg2 = float(sigma_full[3 + n_pp * i + 4])
                    sigma_gamma_val = float(0.5 * np.sqrt(sg1 ** 2 + sg2 ** 2))
                elif width_mode == 'symmetric':
                    gamma_fit = float(popt[3 + n_pp*i + 3])
                    sigma_gamma_val = float(sigma_full[3 + n_pp * i + 3])
                    g1 = g2 = gamma_fit
                    sg1 = sg2 = sigma_gamma_val
                else:
                    # 'global' : w_global est avant η si voigt → idx -2 sinon -1
                    g_idx = -2 if is_voigt else -1
                    gamma_fit = float(popt[g_idx])
                    sigma_gamma_val = float(sigma_full[g_idx])
                    g1 = g2 = gamma_fit
                    sg1 = sg2 = sigma_gamma_val
                gamma_list[i].append(gamma_fit if not jumped else np.nan)
                gamma_left_list[i].append(g1 if not jumped else np.nan)
                gamma_right_list[i].append(g2 if not jumped else np.nan)
                sigma_gamma_list[i].append(
                    sigma_gamma_val if not jumped else np.nan
                )
                sigma_gamma_left_list[i].append(sg1 if not jumped else np.nan)
                sigma_gamma_right_list[i].append(sg2 if not jumped else np.nan)

                if verbose:
                    status = 'OK' if (converged and not jumped) else ('JUMP' if jumped else 'LOW_A')
                    print(f'  E={ev_arr[ie]:+.3f} eV | k0={k0_fit:.3f} '
                          f'A-={A1:.3f} A+={A2:.3f} xg={xg_fit:.3f} '
                          f'ref_k0={prev_k0[i]:.3f} [{status}]')

            if converged:
                xg_list.append(xg_fit)
                e_fitted.append(ev_arr[ie])
                fit_curve_list.append(fit_y)
                residual_list.append(residual_y)
                chi2_list.append(chi2_red)
                prev_popt = popt
                # Mise à jour du k0 de référence pour la paire
                # IMPORTANT : on ne met à jour que si A est suffisante (évite
                # de corrompre prev_k0 avec un fit sur fond sans pic réel)
                for i in range(n_pairs):
                    k0_val = popt[3 + n_pp*i]
                    if width_mode == 'independent':
                        A1_upd = popt[3 + n_pp*i + 1]
                        A2_upd = popt[3 + n_pp*i + 3]
                    elif width_mode == 'symmetric':
                        A1_upd = popt[3 + n_pp*i + 1]
                        A2_upd = popt[3 + n_pp*i + 2]
                    else:
                        A1_upd = popt[3 + n_pp*i + 1]
                        A2_upd = popt[3 + n_pp*i + 2]
                    # Ne pas mettre à jour si les deux amplitudes sont trop faibles
                    # (le fit a accroché du fond — on garde l'ancien k0 de référence)
                    if max(A1_upd, A2_upd) > min_amplitude:
                        prev_k0[i] = k0_val
            else:
                xg_list.append(np.nan)
                e_fitted.append(ev_arr[ie])
                fit_curve_list.append(fit_y)
                residual_list.append(residual_y)
                chi2_list.append(chi2_red)
                prev_popt = None   # reset warm-start si saut ou amplitude trop faible
                # prev_k0 conservé intentionnellement : on garde le dernier k0 valide
                # comme référence pour la détection de saut des prochaines tranches

        except Exception as exc:
            if verbose:
                print(f'  E={ev_arr[ie]:+.3f} eV: exception -> {exc}')
            for i in range(n_pairs):
                kF_minus_list[i].append(np.nan)
                kF_plus_list[i].append(np.nan)
                k0_list[i].append(np.nan)
                gamma_list[i].append(np.nan)
                gamma_left_list[i].append(np.nan)
                gamma_right_list[i].append(np.nan)
                sigma_gamma_left_list[i].append(np.nan)
                sigma_gamma_right_list[i].append(np.nan)
                sigma_kF_minus_list[i].append(np.nan)
                sigma_kF_plus_list[i].append(np.nan)
                sigma_gamma_list[i].append(np.nan)
            xg_list.append(np.nan)
            e_fitted.append(ev_arr[ie])
            fit_curve_list.append(np.full_like(kpar_fit, np.nan, dtype=float))
            residual_list.append(np.full_like(kpar_fit, np.nan, dtype=float))
            chi2_list.append(np.nan)
            eta_list.append(np.nan)
            prev_popt = None

    # Réordonner en énergie croissante (indépendant du sens du scan)
    e_arr_out = np.array(e_fitted)
    sort_idx  = np.argsort(e_arr_out)
    k0_out = [np.array(x)[sort_idx] for x in k0_list]
    gamma_brut = [np.array(x)[sort_idx] for x in gamma_list]
    gamma_left_brut  = [np.array(x)[sort_idx] for x in gamma_left_list]
    gamma_right_brut = [np.array(x)[sort_idx] for x in gamma_right_list]
    gamma_left_corrige = []
    gamma_right_corrige = []
    for i in range(n_pairs):
        _, glc = _resolution_correct_gamma(
            e_arr_out[sort_idx], k0_out[i], gamma_left_brut[i],
            dE_eV=dE_eV, dk_inv_a=dk_inv_a,
        )
        _, grc = _resolution_correct_gamma(
            e_arr_out[sort_idx], k0_out[i], gamma_right_brut[i],
            dE_eV=dE_eV, dk_inv_a=dk_inv_a,
        )
        gamma_left_corrige.append(glc)
        gamma_right_corrige.append(grc)
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
        kF_minus   =[np.array(x)[sort_idx] for x in kF_minus_list],
        kF_plus    =[np.array(x)[sort_idx] for x in kF_plus_list],
        sigma_kF_minus=[np.array(x)[sort_idx] for x in sigma_kF_minus_list],
        sigma_kF_plus =[np.array(x)[sort_idx] for x in sigma_kF_plus_list],
        sigma_gamma   =[np.array(x)[sort_idx] for x in sigma_gamma_list],
        k0         =k0_out,
        xg         =np.array(xg_list)[sort_idx],
        e_fitted   =e_arr_out[sort_idx],
        I_smoothed =I_fit,
        fit_kpar   =kpar_fit,
        fit_curves =[np.array(x) for x in np.asarray(fit_curve_list, dtype=float)[sort_idx]],
        residuals  =[np.array(x) for x in np.asarray(residual_list, dtype=float)[sort_idx]],
        chi2_red   =np.asarray(chi2_list, dtype=float)[sort_idx],
        kpar       =kpar,
        ev_arr     =ev_arr,
        n_pairs    =n_pairs,
        width_mode =width_mode,
        shape       =shape,
        width_convention="HWHM",
        gamma_units="pi/a HWHM",
        fit_constraints={
            "hold_center": bool(hold_center),
            "hold_gamma": bool(hold_gamma),
        },
        eta         =np.asarray(eta_list, dtype=float)[sort_idx] if len(eta_list) == len(sort_idx) else np.asarray(eta_list, dtype=float),
        gamma       =gamma_brut,
        gamma_brut  =gamma_brut,
        gamma_min   =gamma_min,
        gamma_corrige=gamma_corrige,
        gamma_left_brut    =gamma_left_brut,
        gamma_right_brut   =gamma_right_brut,
        gamma_left_corrige =gamma_left_corrige,
        gamma_right_corrige=gamma_right_corrige,
        sigma_gamma_left   =[np.array(x)[sort_idx] for x in sigma_gamma_left_list],
        sigma_gamma_right  =[np.array(x)[sort_idx] for x in sigma_gamma_right_list],
        resolution  ={
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
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


# =============================================================================
#  5b. Diagnostics MDC — visualisation des fits slice par slice
# =============================================================================
