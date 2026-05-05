"""General and region-based MDC Lorentzian fitting."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import _make_multi_lor

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


# =============================================================================
#  6b. Suppression de l'artefact de grille (détecteur DA30 / MCP)
# =============================================================================
