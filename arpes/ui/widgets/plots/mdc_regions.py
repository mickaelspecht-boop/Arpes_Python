"""General and region-based MDC Lorentzian fitting."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import _make_multi_lor, _resolution_correct_gamma

def fit_mdc_lorentzians(
    data_cut, kpar, ev_arr,
    ev_start=-0.15, ev_end=-0.01,
    n_lor=2,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.15,
    min_amplitude=0.05,
):
    """
    Fit N Lorentziennes sur les MDCs a chaque energie, avec
    Hungarian matching pour suivre les bandes.

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    kpar, ev_arr : np.ndarray 1D
    ev_start, ev_end : float
        Fenetre en energie pour le fit.
    n_lor : int
        Nombre de Lorentziennes.
    smooth_fit : float
        Lissage pour les donnees fittees.
    smooth_detect : float
        Lissage pour la detection de pics (guess initial).
    gamma_init, gamma_max : float
        Largeur initiale et max des Lorentziennes.
    min_amplitude : float
        Amplitude minimale pour valider un pic.

    Retourne
    --------
    dict avec cles:
        'k_peaks': list de N arrays (un par bande)
        'e_fitted': array des energies ou le fit a converge
        'I_smoothed': array lisse pour le plot
        'kpar', 'ev_arr': axes
    """
    from scipy.optimize import linear_sum_assignment

    lor_model = _make_multi_lor(n_lor)

    I_fit = gaussian_filter1d(data_cut.astype(float), sigma=smooth_fit, axis=0)
    I_detect = gaussian_filter1d(data_cut.astype(float), sigma=smooth_detect, axis=0)

    lo = [-np.inf, -np.inf] + [float(kpar[0]), 0.005, 0.0] * n_lor
    hi = [np.inf, np.inf] + [float(kpar[-1]), gamma_max, 5.0] * n_lor

    k_peaks_list = [[] for _ in range(n_lor)]
    e_fitted = []
    prev_popt = None

    wf_energies = np.arange(ev_start, ev_end + 1e-9,
                            float(ev_arr[1] - ev_arr[0]))

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc = I_fit[:, ie]
        mdc_max = mdc.max()
        if mdc_max <= 0:
            continue
        mdc_n = mdc / mdc_max

        if prev_popt is not None:
            p0 = list(prev_popt)
        else:
            mdc_detect = I_detect[:, ie]
            mdc_detect_max = mdc_detect.max()
            mdc_detect_n = (mdc_detect / mdc_detect_max
                            if mdc_detect_max > 0 else mdc_n)
            pk_idx, props = find_peaks(
                mdc_detect_n, prominence=0.05,
                distance=max(1, int(0.05 / (kpar[1] - kpar[0]))))
            if len(pk_idx) >= n_lor:
                top_n = pk_idx[np.argsort(props['prominences'])[-n_lor:]]
                top_n = top_n[np.argsort(kpar[top_n])]
                k_guesses = [float(kpar[j]) for j in top_n]
            else:
                k_guesses = list(np.linspace(
                    float(kpar[0]) * 0.8, float(kpar[-1]) * 0.8, n_lor))
            p0 = [0.0, float(np.percentile(mdc_n, 10))]
            for kg in k_guesses:
                p0 += [kg, gamma_init, 0.7]

        try:
            popt, _ = curve_fit(lor_model, kpar, mdc_n, p0=p0,
                                bounds=(lo, hi), maxfev=5000)
            k_fitted = [popt[2 + 3*i] for i in range(n_lor)]
            A_fitted = [popt[2 + 3*i + 2] for i in range(n_lor)]

            if len(e_fitted) > 0:
                prev_k = []
                for j in range(n_lor):
                    last_valid = np.nan
                    for v in reversed(k_peaks_list[j]):
                        if np.isfinite(v):
                            last_valid = v
                            break
                    prev_k.append(last_valid)

                if all(np.isfinite(pk) for pk in prev_k):
                    cost = np.array([[abs(k_fitted[i] - prev_k[j])
                                      for j in range(n_lor)]
                                     for i in range(n_lor)])
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for ri, ci in zip(row_ind, col_ind):
                        k_peaks_list[ci].append(
                            k_fitted[ri] if A_fitted[ri] > min_amplitude
                            else np.nan)
                else:
                    order = np.argsort(k_fitted)
                    for j, oj in enumerate(order):
                        k_peaks_list[j].append(
                            k_fitted[oj] if A_fitted[oj] > min_amplitude
                            else np.nan)
            else:
                order = np.argsort(k_fitted)
                for j, oj in enumerate(order):
                    k_peaks_list[j].append(
                        k_fitted[oj] if A_fitted[oj] > min_amplitude
                        else np.nan)

            e_fitted.append(ev_arr[ie])
            prev_popt = popt
        except Exception:
            prev_popt = None

    return dict(
        k_peaks=[np.array(kp) for kp in k_peaks_list],
        e_fitted=np.array(e_fitted),
        I_smoothed=I_fit,
        kpar=kpar,
        ev_arr=ev_arr,
    )


# =============================================================================
#  6. Fit MDC par regions k definies manuellement
# =============================================================================


def fit_mdc_regions(
    data_cut, kpar, ev_arr,
    k_regions,
    ev_start=-0.15, ev_end=-0.01,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.15,
    min_amplitude=0.05,
    n_lor_default=1,
):
    """
    Fit des Lorentziennes sur les MDCs en restraignant chaque fit
    a une region k definie manuellement. Plus robuste que le fit global
    quand les bandes se croisent ou sont tres proches.

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    kpar, ev_arr : np.ndarray 1D
    k_regions : list of (k_min, k_max) ou (k_min, k_max, n_lor)
        Une region par groupe de bandes. Le 3e element optionnel fixe
        le nombre de Lorentziennes dans cette region specifiquement.
        Ex: [(-0.6, -0.05, 2),   # 2 bandes dans la region gauche
             ( 0.05,  0.6,  1)]  # 1 bande dans la region droite
    ev_start, ev_end : float
        Fenetre en energie pour le fit.
    smooth_fit : float
        Lissage pour les donnees fittees.
    smooth_detect : float
        Lissage pour la detection de pics (guess initial).
    gamma_init, gamma_max : float
        Largeur initiale et max des Lorentziennes.
    min_amplitude : float
        Amplitude minimale pour valider un pic.
    n_lor_default : int
        Nombre de Lorentziennes par defaut si non specifie dans le tuple.

    Retourne
    --------
    dict avec cles:
        'k_peaks'   : list de listes d'arrays — k_peaks[ri][bi] = array
                      des positions de la bande bi dans la region ri
        'e_fitted'  : array des energies ou au moins un fit a converge
        'I_smoothed': array lisse pour le plot
        'kpar'      : axe k complet
        'ev_arr'    : axe energie complet
        'k_regions' : regions utilisees (pour reference)
        'n_lors'    : nombre de Lorentziennes par region
    """
    # Normalise les regions : (klo, khi, n_lor)
    regions_full = []
    for reg in k_regions:
        if len(reg) == 3:
            regions_full.append((reg[0], reg[1], int(reg[2])))
        else:
            regions_full.append((reg[0], reg[1], n_lor_default))
    n_regions = len(regions_full)
    n_lors = [r[2] for r in regions_full]

    I_fit    = gaussian_filter1d(data_cut.astype(float), sigma=smooth_fit,   axis=0)
    I_detect = gaussian_filter1d(data_cut.astype(float), sigma=smooth_detect, axis=0)

    # Masques k pour chaque region
    masks = [(kpar >= klo) & (kpar <= khi) for (klo, khi, _) in regions_full]

    # k_peaks_list[ri][bi] = liste de valeurs au fil des energies
    k_peaks_list = [[[] for _ in range(nl)] for nl in n_lors]
    e_fitted      = []
    prev_popts    = [None] * n_regions

    wf_energies = np.arange(ev_start, ev_end + 1e-9,
                            float(ev_arr[1] - ev_arr[0]))

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc_full   = I_fit[:, ie]
        mdc_d_full = I_detect[:, ie]
        if mdc_full.max() <= 0:
            continue

        any_converged = False

        for ri, ((klo, khi, n_lor), mask) in enumerate(zip(regions_full, masks)):
            k_reg   = kpar[mask]
            mdc_reg = mdc_full[mask]
            mdc_d   = mdc_d_full[mask]

            if len(k_reg) < 2 * n_lor + 2:
                for bi in range(n_lor):
                    k_peaks_list[ri][bi].append(np.nan)
                continue

            mdc_max = mdc_reg.max()
            if mdc_max <= 0:
                for bi in range(n_lor):
                    k_peaks_list[ri][bi].append(np.nan)
                continue
            mdc_n = mdc_reg / mdc_max

            lor_model = _make_multi_lor(n_lor)
            lo = [-np.inf, -np.inf] + [klo, 0.005, 0.0] * n_lor
            hi = [ np.inf,  np.inf] + [khi, gamma_max, 5.0] * n_lor

            # Point de depart
            if prev_popts[ri] is not None:
                p0 = list(prev_popts[ri])
            else:
                mdc_d_max = mdc_d.max()
                mdc_d_n   = mdc_d / mdc_d_max if mdc_d_max > 0 else mdc_n
                pk_idx, props = find_peaks(
                    mdc_d_n, prominence=0.05,
                    distance=max(1, int(0.05 / (kpar[1] - kpar[0]))))
                if len(pk_idx) >= n_lor:
                    top_n = pk_idx[np.argsort(props['prominences'])[-n_lor:]]
                    top_n = top_n[np.argsort(k_reg[top_n])]
                    k_guesses = [float(k_reg[j]) for j in top_n]
                else:
                    k_guesses = list(np.linspace(
                        klo + (khi - klo) * 0.1,
                        khi - (khi - klo) * 0.1,
                        n_lor))
                p0 = [0.0, float(np.percentile(mdc_n, 10))]
                for kg in k_guesses:
                    p0 += [kg, gamma_init, 0.7]

            try:
                popt, _ = curve_fit(lor_model, k_reg, mdc_n, p0=p0,
                                    bounds=(lo, hi), maxfev=5000)
                k_fit = [popt[2 + 3*i]     for i in range(n_lor)]
                A_fit = [popt[2 + 3*i + 2] for i in range(n_lor)]

                # Hungarian matching avec les positions precedentes si dispo
                from scipy.optimize import linear_sum_assignment
                if prev_popts[ri] is not None and all(
                        len(k_peaks_list[ri][bi]) > 0 for bi in range(n_lor)):
                    prev_k = [next(
                        (v for v in reversed(k_peaks_list[ri][bi]) if np.isfinite(v)),
                        np.nan) for bi in range(n_lor)]
                    if all(np.isfinite(pk) for pk in prev_k):
                        cost = np.array([[abs(k_fit[i] - prev_k[j])
                                          for j in range(n_lor)]
                                         for i in range(n_lor)])
                        row_ind, col_ind = linear_sum_assignment(cost)
                        ordered = [None] * n_lor
                        for ri2, ci in zip(row_ind, col_ind):
                            ordered[ci] = ri2
                        for bi in range(n_lor):
                            src = ordered[bi]
                            val = k_fit[src] if A_fit[src] > min_amplitude else np.nan
                            k_peaks_list[ri][bi].append(val)
                            if np.isfinite(val):
                                any_converged = True
                        prev_popts[ri] = popt
                        continue

                # Ordre croissant par defaut
                order = np.argsort(k_fit)
                for bi, oi in enumerate(order):
                    val = k_fit[oi] if A_fit[oi] > min_amplitude else np.nan
                    k_peaks_list[ri][bi].append(val)
                    if np.isfinite(val):
                        any_converged = True
                prev_popts[ri] = popt

            except Exception:
                for bi in range(n_lor):
                    k_peaks_list[ri][bi].append(np.nan)
                prev_popts[ri] = None

        if any_converged:
            e_fitted.append(ev_arr[ie])

    return dict(
        k_peaks=[[np.array(k_peaks_list[ri][bi]) for bi in range(n_lors[ri])]
                 for ri in range(n_regions)],
        e_fitted=np.array(e_fitted),
        I_smoothed=I_fit,
        kpar=kpar,
        ev_arr=ev_arr,
        k_regions=regions_full,
        n_lors=n_lors,
    )


def fit_mdc_free_region_result(
    data_cut, kpar, ev_arr,
    *,
    k_min,
    k_max,
    ev_start=-0.15,
    ev_end=-0.01,
    n_lor=1,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.15,
    min_amplitude=0.05,
    center_init=0.0,
    max_jump=0.15,
    mdc_energy_window=0.0,
    scan_direction="up",
    dE_eV=0.0,
    dk_inv_a=0.0,
    resolution_source="",
    verbose=False,
):
    """Fit independent Lorentzian peak(s) in one k-region, Results-compatible.

    This wraps the older free-region idea in the same schema as
    ``fit_mdc_peak_pairs``: kF_minus/kF_plus, Γ(E), σ, residuals, chi² and
    resolution metadata. Each fitted Lorentzian becomes one "pair index"; if
    its median position lies left of ``center_init`` it populates kF_minus,
    otherwise kF_plus. The opposite branch is NaN by construction.
    """
    from scipy.optimize import linear_sum_assignment

    n_lor = max(1, int(n_lor))
    kpar = np.asarray(kpar, dtype=float)
    ev_arr = np.asarray(ev_arr, dtype=float)
    data_cut = np.asarray(data_cut, dtype=float)
    I_fit = gaussian_filter1d(data_cut, sigma=smooth_fit, axis=0)
    I_detect = gaussian_filter1d(data_cut, sigma=smooth_detect, axis=0)

    k_lo = float(k_min)
    k_hi = float(k_max)
    if k_hi < k_lo:
        k_lo, k_hi = k_hi, k_lo
    k_mask = (kpar >= k_lo) & (kpar <= k_hi)
    k_fit = kpar[k_mask]
    if k_fit.size < max(8, 3 * n_lor + 2):
        return _empty_free_region_result(I_fit, kpar, ev_arr, k_fit, n_lor, resolution_source, dE_eV, dk_inv_a)
    dk_pix = abs(float(k_fit[1] - k_fit[0])) if k_fit.size > 1 else 0.0

    model = _make_multi_lor(n_lor)
    lo = [-np.inf, -np.inf] + [float(k_fit[0]), max(1e-9, float(dk_inv_a or 0.0)), 0.0] * n_lor
    hi = [np.inf, np.inf] + [float(k_fit[-1]), float(gamma_max), np.inf] * n_lor

    ev_lo = min(float(ev_start), float(ev_end))
    ev_hi = max(float(ev_start), float(ev_end))
    wf_indices = np.where((ev_arr >= ev_lo) & (ev_arr <= ev_hi))[0]
    wf_energies_all = ev_arr[wf_indices]
    wf_energies = np.sort(wf_energies_all)[::-1] if scan_direction == "down" else np.sort(wf_energies_all)

    k_tracks = [[] for _ in range(n_lor)]
    gamma_tracks = [[] for _ in range(n_lor)]
    sigma_k_tracks = [[] for _ in range(n_lor)]
    sigma_gamma_tracks = [[] for _ in range(n_lor)]
    e_fitted = []
    fit_curves = []
    residuals = []
    chi2_red = []
    prev_popt = None
    prev_k = None

    def _initial_p0(mdc_n, mdc_detect_n):
        dk = abs(float(k_fit[1] - k_fit[0])) if k_fit.size > 1 else 0.01
        pk_idx, props = find_peaks(
            mdc_detect_n,
            prominence=0.05,
            distance=max(1, int(0.04 / dk)),
        )
        if len(pk_idx) >= n_lor:
            top = pk_idx[np.argsort(props["prominences"])[-n_lor:]]
            guesses = sorted(float(k_fit[j]) for j in top)
        else:
            guesses = list(np.linspace(k_lo + 0.15 * (k_hi - k_lo),
                                       k_hi - 0.15 * (k_hi - k_lo), n_lor))
        p0 = [0.0, float(np.nanpercentile(mdc_n, 10))]
        for kg in guesses:
            amp = float(np.interp(kg, k_fit, mdc_n))
            p0 += [kg, float(gamma_init), max(amp, 0.05)]
        return p0

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        if mdc_energy_window > 0:
            e_mask = np.abs(ev_arr - ev_arr[ie]) <= 0.5 * float(mdc_energy_window)
            block = data_cut[:, e_mask]   # intègre la donnée BRUTE (pas I_fit lissé-k)
            mdc_raw = np.nanmean(block, axis=1) if block.shape[1] else data_cut[:, ie].astype(float)
        else:
            mdc_raw = data_cut[:, ie].astype(float)
        # Lissage k une seule fois, après l'intégration énergie (cf mdc_fit.py).
        mdc_full = gaussian_filter1d(mdc_raw, sigma=smooth_fit) if smooth_fit and smooth_fit > 0 else mdc_raw
        mdc = mdc_full[k_mask]
        mx = float(np.nanmax(mdc)) if mdc.size else 0.0
        if mx <= 0:
            continue
        mdc_n = mdc / mx
        mdc_d = I_detect[k_mask, ie]
        mdx = float(np.nanmax(mdc_d)) if mdc_d.size else 0.0
        mdc_d_n = mdc_d / mdx if mdx > 0 else mdc_n
        p0 = list(prev_popt) if prev_popt is not None else _initial_p0(mdc_n, mdc_d_n)
        try:
            popt, pcov = curve_fit(model, k_fit, mdc_n, p0=p0, bounds=(lo, hi), maxfev=10000)
            sigma_full = np.sqrt(np.abs(np.diag(pcov)))
            k_now = [float(popt[2 + 3 * j]) for j in range(n_lor)]
            g_now = [float(popt[3 + 3 * j]) for j in range(n_lor)]
            a_now = [float(popt[4 + 3 * j]) for j in range(n_lor)]
            sk_now = [float(sigma_full[2 + 3 * j]) for j in range(n_lor)]
            sg_now = [float(sigma_full[3 + 3 * j]) for j in range(n_lor)]

            order = list(np.argsort(k_now))
            if prev_k is not None and all(np.isfinite(prev_k)):
                cost = np.array([[abs(k_now[i] - prev_k[j]) for j in range(n_lor)]
                                 for i in range(n_lor)])
                rows, cols = linear_sum_assignment(cost)
                mapped = [None] * n_lor
                for ri, ci in zip(rows, cols):
                    mapped[ci] = ri
                order = [idx if idx is not None else int(np.argsort(k_now)[j])
                         for j, idx in enumerate(mapped)]

            jumped = False
            if prev_k is not None and all(np.isfinite(prev_k)):
                for track_i, src_i in enumerate(order):
                    if abs(k_now[src_i] - prev_k[track_i]) > float(max_jump):
                        jumped = True
                        break

            fit_y = model(k_fit, *popt)
            residual_y = mdc_n - fit_y
            dof = max(1, int(k_fit.size) - int(len(popt)))
            for track_i, src_i in enumerate(order):
                valid = (not jumped) and a_now[src_i] > float(min_amplitude)
                k_tracks[track_i].append(k_now[src_i] if valid else np.nan)
                gamma_tracks[track_i].append(g_now[src_i] if valid else np.nan)
                sigma_k_tracks[track_i].append(sk_now[src_i] if valid else np.nan)
                sigma_gamma_tracks[track_i].append(sg_now[src_i] if valid else np.nan)
            e_fitted.append(float(ev_arr[ie]))
            fit_curves.append(fit_y)
            residuals.append(residual_y)
            chi2_red.append(float(np.nansum(residual_y ** 2) / dof))
            prev_popt = None if jumped else popt
            prev_k = [k_now[src_i] for src_i in order] if not jumped else prev_k
        except Exception as exc:
            if verbose:
                print(f"free region MDC fit failed at E={ev_arr[ie]:+.3f}: {exc}")
            prev_popt = None
            for track_i in range(n_lor):
                k_tracks[track_i].append(np.nan)
                gamma_tracks[track_i].append(np.nan)
                sigma_k_tracks[track_i].append(np.nan)
                sigma_gamma_tracks[track_i].append(np.nan)
            e_fitted.append(float(ev_arr[ie]))
            fit_curves.append(np.full_like(k_fit, np.nan, dtype=float))
            residuals.append(np.full_like(k_fit, np.nan, dtype=float))
            chi2_red.append(np.nan)

    e_out = np.asarray(e_fitted, dtype=float)
    sort_idx = np.argsort(e_out)
    e_sorted = e_out[sort_idx]
    k_sorted = [np.asarray(v, dtype=float)[sort_idx] for v in k_tracks]
    g_sorted = [np.asarray(v, dtype=float)[sort_idx] for v in gamma_tracks]
    sk_sorted = [np.asarray(v, dtype=float)[sort_idx] for v in sigma_k_tracks]
    sg_sorted = [np.asarray(v, dtype=float)[sort_idx] for v in sigma_gamma_tracks]

    kF_minus = []
    kF_plus = []
    sigma_kF_minus = []
    sigma_kF_plus = []
    k0 = []
    for k_arr, sk_arr in zip(k_sorted, sk_sorted):
        med = float(np.nanmedian(k_arr)) if np.isfinite(k_arr).any() else float("nan")
        is_left = np.isfinite(med) and med <= float(center_init)
        nan_arr = np.full_like(k_arr, np.nan, dtype=float)
        if is_left:
            kF_minus.append(k_arr)
            kF_plus.append(nan_arr.copy())
            sigma_kF_minus.append(sk_arr)
            sigma_kF_plus.append(nan_arr.copy())
        else:
            kF_minus.append(nan_arr.copy())
            kF_plus.append(k_arr)
            sigma_kF_minus.append(nan_arr.copy())
            sigma_kF_plus.append(sk_arr)
        k0.append(np.abs(k_arr - float(center_init)))

    gamma_min = []
    gamma_corrige = []
    for i in range(n_lor):
        gmin, gcorr = _resolution_correct_gamma(
            e_sorted, k0[i], g_sorted[i],
            dE_eV=dE_eV, dk_inv_a=dk_inv_a,
            smooth_fit_sigma_px=smooth_fit, dk_pixel=dk_pix,
        )
        gamma_min.append(gmin)
        gamma_corrige.append(gcorr)

    return {
        "fit_model": "free_region",
        "width_mode": "free_region",
        "shape": "lorentzian",
        "width_convention": "HWHM",
        "gamma_units": "pi/a HWHM",
        "fit_constraints": {"region_k_min": k_lo, "region_k_max": k_hi},
        "e_fitted": e_sorted,
        "kF_minus": kF_minus,
        "kF_plus": kF_plus,
        "sigma_kF_minus": sigma_kF_minus,
        "sigma_kF_plus": sigma_kF_plus,
        "k0": k0,
        "xg": np.full_like(e_sorted, float(center_init), dtype=float),
        "gamma": g_sorted,
        "gamma_brut": g_sorted,
        "gamma_min": gamma_min,
        "gamma_corrige": gamma_corrige,
        "sigma_gamma": sg_sorted,
        "chi2_red": np.asarray(chi2_red, dtype=float)[sort_idx],
        "fit_kpar": k_fit,
        "fit_curves": [np.asarray(v, dtype=float) for v in np.asarray(fit_curves, dtype=float)[sort_idx]],
        "residuals": [np.asarray(v, dtype=float) for v in np.asarray(residuals, dtype=float)[sort_idx]],
        "I_smoothed": I_fit,
        "kpar": kpar,
        "ev_arr": ev_arr,
        "n_pairs": n_lor,
        "n_lor": n_lor,
        "k_regions": [(k_lo, k_hi, n_lor)],
        "resolution": {
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
    }


def _empty_free_region_result(I_fit, kpar, ev_arr, k_fit, n_lor, resolution_source, dE_eV, dk_inv_a):
    empty = [np.array([]) for _ in range(n_lor)]
    return {
        "fit_model": "free_region",
        "width_mode": "free_region",
        "shape": "lorentzian",
        "width_convention": "HWHM",
        "gamma_units": "pi/a HWHM",
        "e_fitted": np.array([]),
        "kF_minus": [x.copy() for x in empty],
        "kF_plus": [x.copy() for x in empty],
        "sigma_kF_minus": [x.copy() for x in empty],
        "sigma_kF_plus": [x.copy() for x in empty],
        "k0": [x.copy() for x in empty],
        "xg": np.array([]),
        "gamma": [x.copy() for x in empty],
        "gamma_brut": [x.copy() for x in empty],
        "gamma_min": [x.copy() for x in empty],
        "gamma_corrige": [x.copy() for x in empty],
        "sigma_gamma": [x.copy() for x in empty],
        "chi2_red": np.array([]),
        "fit_kpar": k_fit,
        "fit_curves": [],
        "residuals": [],
        "I_smoothed": I_fit,
        "kpar": kpar,
        "ev_arr": ev_arr,
        "n_pairs": n_lor,
        "n_lor": n_lor,
        "resolution": {
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
    }


# =============================================================================
#  6b. Suppression de l'artefact de grille (détecteur DA30 / MCP)
# =============================================================================
