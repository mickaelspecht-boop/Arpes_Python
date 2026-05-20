"""Fit model helpers shared by MDC and EDC routines."""

import numpy as np

def _fd(x, ef, width):
    """Distribution Fermi-Dirac. width = kT en eV."""
    return 1.0 / (np.exp(np.clip((x - ef) / width, -500, 500)) + 1.0)


def _gauss_peak(x, x0, A, width):
    """Gaussienne normalisee : A × exp(-2(x-x0)²/width²)."""
    return A * np.exp(-2.0 * (x - x0)**2 / width**2)


def _lor_peak(x, x0, A, width):
    """Lorentzienne : A × (width/2)² / ((x-x0)² + (width/2)²)."""
    return A * (width / 2)**2 / ((x - x0)**2 + (width / 2)**2)


def _voigt_pseudo(x, x0, A, sigma, eta):
    """Pseudo-Voigt : eta×Lor + (1-eta)×Gauss. eta in [0,1]."""
    eta = np.clip(eta, 0.0, 1.0)
    return A * (eta * (sigma/2)**2 / ((x-x0)**2 + (sigma/2)**2)
                + (1 - eta) * np.exp(-2*(x-x0)**2 / sigma**2))


def _make_edc_model(n_peaks, shape='lorentzian', bg='linear', with_fd=True):
    """
    Fabrique un modele EDC :
        EDC(E) = [fond(E) + Σ peak_i(E)] × FD(E)  [si with_fd]
              ou  fond(E) + Σ peak_i(E)             [sinon]

    shape : 'lorentzian' | 'gaussian' | 'voigt'
    bg    : 'linear' | 'quadratic'

    Parametres du vecteur p :
      p[0]        : bg_offset
      p[1]        : bg_slope
      [p[2]       : bg_quad  (si quadratic)]
      bkg_end = 2 + (1 if quadratic else 0)
      Pour chaque pic i (3 params si lor/gauss, 4 si voigt) :
        p[bkg_end + 3*i]   : x0_i  (position)
        p[bkg_end + 3*i+1] : A_i   (amplitude)
        p[bkg_end + 3*i+2] : w_i   (largeur)
       [p[bkg_end + 4*i+3] : eta_i (voigt uniquement)]
      Derniers params si with_fd :
        p[-2] : ef    (position EF)
        p[-1] : kT    (temperature thermique en eV)
    """
    n_bg   = 3 if bg == 'quadratic' else 2
    n_pp   = 4 if shape == 'voigt' else 3  # params par pic

    def model(x, *p):
        p = np.asarray(p)
        # Fond
        if bg == 'quadratic':
            bg_val = p[0] + p[1] * (x - p[2])**2
            ofs = 3
        else:
            bg_val = p[0] + p[1] * x
            ofs = 2

        # Pics
        peak_val = np.zeros_like(x, dtype=float)
        for i in range(n_peaks):
            x0 = p[ofs + n_pp * i]
            A  = p[ofs + n_pp * i + 1]
            w  = p[ofs + n_pp * i + 2]
            if shape == 'gaussian':
                peak_val += _gauss_peak(x, x0, A, w)
            elif shape == 'voigt':
                eta = p[ofs + n_pp * i + 3]
                peak_val += _voigt_pseudo(x, x0, A, w, eta)
            else:  # lorentzian
                peak_val += _lor_peak(x, x0, A, w)

        spec = bg_val + peak_val

        if with_fd:
            ef_fit = p[-2]
            kT_fit = p[-1]
            spec   = spec * _fd(x, ef_fit, kT_fit)

        return spec

    return model, n_bg, n_pp


_WIDTH_MODE_ALIASES = {"asymmetric": "independent", "asym": "independent"}


def _normalize_width_mode(width_mode: str) -> str:
    """Mappe les alias historiques (UI) vers les noms backend canoniques."""
    return _WIDTH_MODE_ALIASES.get(str(width_mode), str(width_mode))


def _make_peak_pairs_model(n_pairs, width_mode='independent', shape='lorentzian'):
    """
    Fabrique un modele MDC a N paires de pics symétriques.

    Chaque paire i fitte deux pics à ±k0_i + xg (centre commun).

    width_mode :
        'independent' — chaque pic a sa propre largeur (2 largeurs/paire)
        'global'      — une seule largeur partagée par toutes les paires
        'symmetric'   — w1=w2 au sein de chaque paire, mais varie entre paires

    shape :
        'lorentzian' — défaut, élargissement intrinsèque (durée de vie).
        'voigt'      — pseudo-Voigt = (1-η)·L + η·G (résolution incluse).
                       Ajoute un paramètre global η ∈ [0,1] à la FIN de p.

    Vecteur de paramètres p :
        p[0] : bg_a  (fond linéaire pente)
        p[1] : bg_b  (fond linéaire offset)
        p[2] : xg    (centre commun de toutes les paires, souvent ~0)
        Puis pour chaque paire i :
          independent : k0_i, A1_i, w1_i, A2_i, w2_i   (5 params)
          symmetric   : k0_i, A1_i, A2_i, w_i           (4 params)
          global      : k0_i, A1_i, A2_i                 (3 params) + w_global
        Si shape='voigt' : un dernier paramètre η_global appended.
    """
    width_mode = _normalize_width_mode(width_mode)
    is_voigt = (shape == 'voigt')

    def model(k, *p):
        p  = np.asarray(p, dtype=float)
        bg = p[0] * k + p[1]
        xg = p[2]
        res = bg.copy()
        eta = float(p[-1]) if is_voigt else 0.0

        def _peak(x, x0, A, w):
            if is_voigt:
                return _voigt_pseudo(x, x0, A, w, eta)
            return _lor_peak(x, x0, A, w)

        if width_mode == 'global':
            # En voigt, w_global est avant eta (donc p[-2]) ; sinon p[-1].
            w_global = p[-2] if is_voigt else p[-1]
            for i in range(n_pairs):
                k0 = p[3 + 3*i]
                A1 = p[3 + 3*i + 1]
                A2 = p[3 + 3*i + 2]
                res += _peak(k, -k0 + xg, A1, w_global)
                res += _peak(k, +k0 + xg, A2, w_global)
        elif width_mode == 'symmetric':
            for i in range(n_pairs):
                k0 = p[3 + 4*i]
                A1 = p[3 + 4*i + 1]
                A2 = p[3 + 4*i + 2]
                w  = p[3 + 4*i + 3]
                res += _peak(k, -k0 + xg, A1, w)
                res += _peak(k, +k0 + xg, A2, w)
        else:  # independent
            for i in range(n_pairs):
                k0 = p[3 + 5*i]
                A1 = p[3 + 5*i + 1]
                w1 = p[3 + 5*i + 2]
                A2 = p[3 + 5*i + 3]
                w2 = p[3 + 5*i + 4]
                res += _peak(k, -k0 + xg, A1, w1)
                res += _peak(k, +k0 + xg, A2, w2)
        return res

    # Nombre de params par paire selon le mode
    n_pp = {'independent': 5, 'symmetric': 4, 'global': 3}[width_mode]
    n_extra = (1 if width_mode == 'global' else 0) + (1 if is_voigt else 0)
    return model, n_pp, n_extra


def _local_velocity_from_k(e_arr, k_arr, idx, half_window=2):
    """Estime |dE/dk| local en eV/(pi/a) autour d'un point de dispersion."""
    e = np.asarray(e_arr, dtype=float)
    k = np.asarray(k_arr, dtype=float)
    n = min(e.size, k.size)
    if n < 2:
        return np.nan
    lo = max(0, int(idx) - int(half_window))
    hi = min(n, int(idx) + int(half_window) + 1)
    ee = e[lo:hi]
    kk = k[lo:hi]
    mask = np.isfinite(ee) & np.isfinite(kk)
    if mask.sum() < 2:
        return np.nan
    try:
        dk_dE = float(np.polyfit(ee[mask], kk[mask], 1)[0])
    except Exception:
        return np.nan
    if not np.isfinite(dk_dE) or abs(dk_dE) < 1e-9:
        return np.nan
    return abs(1.0 / dk_dE)


def _resolution_correct_gamma(e_arr, k0_series, gamma_series, dE_eV, dk_inv_a):
    """Retourne gamma_min et gamma_corrige pour une serie de largeurs MDC."""
    e = np.asarray(e_arr, dtype=float)
    k0 = np.asarray(k0_series, dtype=float)
    gamma = np.asarray(gamma_series, dtype=float)
    dE = max(float(dE_eV or 0.0), 0.0)
    dk = max(float(dk_inv_a or 0.0), 0.0)
    gamma_min = np.full_like(gamma, dk, dtype=float)
    for i in range(gamma.size):
        vf = _local_velocity_from_k(e, k0, i)
        if np.isfinite(vf) and vf > 1e-3:
            gamma_min[i] = float(np.sqrt(dk * dk + (dE / vf) ** 2))
    gamma_corr = np.sqrt(np.maximum(0.0, gamma * gamma - gamma_min * gamma_min))
    gamma_corr[~np.isfinite(gamma)] = np.nan
    gamma_min[~np.isfinite(gamma)] = np.nan
    return gamma_min, gamma_corr


def _make_multi_lor(n):
    """Fabrique un modele a N Lorentziennes + fond lineaire."""
    def model(k, *args):
        bg_a, bg_b = args[0], args[1]
        res = bg_a * k + bg_b
        for i in range(n):
            ki, gi, Ai = args[2 + 3*i], args[3 + 3*i], args[4 + 3*i]
            res = res + Ai * gi**2 / ((k - ki)**2 + gi**2)
        return res
    return model
