"""EDC fitting routines."""

import numpy as np
from scipy.optimize import curve_fit

from .fit_overlay import _fd, _gauss_peak, _lor_peak, _make_edc_model, _voigt_pseudo

def fit_edc(edc, ev_arr, ef=0.0, temperature=20.0,
            n_peaks=1, shape='lorentzian', bg='linear',
            with_fd=True,
            x0_guesses=None, A_guesses=None, width_guess=0.05,
            ev_min=None, ev_max=None):
    """
    Fit un EDC (spectre 1D en energie) avec N pics convoluees par FD.

    Parametres
    ----------
    edc         : np.ndarray 1D
    ev_arr      : np.ndarray 1D
    ef          : float — position initiale de EF (eV, defaut 0)
    temperature : float — temperature en K (pour kT initial)
    n_peaks     : int   — nombre de pics
    shape       : 'lorentzian' | 'gaussian' | 'voigt'
    bg          : 'linear' | 'quadratic'
    with_fd     : bool  — multiplier par Fermi-Dirac
    x0_guesses  : list de float — positions initiales (defaut: equireparties)
    A_guesses   : list de float — amplitudes initiales (defaut: max/n_peaks)
    width_guess : float — largeur initiale commune (eV)
    ev_min/max  : float — fenetre de fit (defaut: tout ev_arr)

    Retourne
    --------
    dict avec cles :
        'popt'   : parametres ajustes
        'pcov'   : covariance
        'fit'    : courbe ajustee sur ev_arr complet
        'peaks'  : list de N courbes de pics individuels
        'bg'     : courbe de fond
        'ev_fit' : axe energie utilise pour le fit
        'success': bool
    """
    # Fenetre de fit
    mask = np.ones(len(ev_arr), dtype=bool)
    if ev_min is not None:
        mask &= ev_arr >= ev_min
    if ev_max is not None:
        mask &= ev_arr <= ev_max
    ev_fit  = ev_arr[mask]
    edc_fit = edc[mask]

    model, n_bg, n_pp = _make_edc_model(n_peaks, shape, bg, with_fd)

    kT0 = 8.6173303e-5 * temperature

    # Point de depart
    A0 = float(np.nanmax(edc_fit)) / n_peaks
    if x0_guesses is None:
        x0_guesses = list(np.linspace(float(ev_fit[0]) * 0.8,
                                       float(ev_fit[-1]) * 0.8, n_peaks))
    if A_guesses is None:
        A_guesses = [A0] * n_peaks

    p0 = [float(np.percentile(edc_fit, 5)), 0.0]  # bg offset + slope
    if bg == 'quadratic':
        p0 += [float(ev_fit.mean())]                # bg quad centre

    for i in range(n_peaks):
        p0 += [x0_guesses[i], A_guesses[i], width_guess]
        if shape == 'voigt':
            p0 += [0.5]                              # eta initial

    if with_fd:
        p0 += [ef, max(kT0, 1e-4)]

    try:
        popt, pcov = curve_fit(model, ev_fit, edc_fit, p0=p0, maxfev=10000)
        success = True
    except Exception:
        popt, pcov = np.array(p0), np.full((len(p0), len(p0)), np.nan)
        success = False

    # Reconstruction des courbes
    fit_curve = model(ev_arr, *popt)

    # Fond seul
    if bg == 'quadratic':
        bg_curve = popt[0] + popt[1] * (ev_arr - popt[2])**2
        ofs = 3
    else:
        bg_curve = popt[0] + popt[1] * ev_arr
        ofs = 2

    # Pics individuels
    peak_curves = []
    for i in range(n_peaks):
        x0 = popt[ofs + n_pp * i]
        A  = popt[ofs + n_pp * i + 1]
        w  = popt[ofs + n_pp * i + 2]
        if shape == 'gaussian':
            pc = _gauss_peak(ev_arr, x0, A, w)
        elif shape == 'voigt':
            eta = popt[ofs + n_pp * i + 3]
            pc  = _voigt_pseudo(ev_arr, x0, A, w, eta)
        else:
            pc = _lor_peak(ev_arr, x0, A, w)
        if with_fd:
            pc = pc * _fd(ev_arr, popt[-2], popt[-1])
        peak_curves.append(pc)

    return dict(popt=popt, pcov=pcov, fit=fit_curve,
                peaks=peak_curves, bg=bg_curve,
                ev_fit=ev_fit, success=success)
