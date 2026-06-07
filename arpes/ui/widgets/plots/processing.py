"""Map processing and correction helpers for ARPES plots."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def remove_grid_artifact(data, kpar=None, method='fft',
                          grid_freq=None, notch_width=3, notch_sigma=1.0,
                          ev_ref_range=None, plot=False):
    """
    Remove the periodic DA30 detector delay-grid artifact.

    The artifact appears as periodic stripes along the k axis
    (same frequency at all energies). Two methods are available:

    method='fft'     : notch filter in Fourier space (1D along k).
                       Effective when the grid period is well defined.
    method='profile' : divide each energy slice by the average angular
                       profile (smoothed). Simpler, also removes fixed noise.

    Parameters
    ----------
    data         : np.ndarray (nk, ne)
    kpar         : np.ndarray 1D, k axis (display only)
    method       : 'fft' | 'profile'
    grid_freq    : float or None, grid frequency in cycles/pixel.
                   If None, auto-detect from the average spectrum.
    notch_width  : int, half-width of the notch filter in FFT bins (method='fft')
    notch_sigma  : float, Gaussian smoothing for the notch filter (0 = hard gate, >0 = soft)
    ev_ref_range : tuple (ev_lo, ev_hi) or None, energy window for estimating
                   the reference profile. If None, all energies are used.
    plot         : bool, show the Fourier profile and applied filter

    Returns
    --------
    data_clean : np.ndarray (nk, ne) — données sans artefact de grille
    info       : dict — 'grid_freq', 'grid_period_px', 'method'
    """
    nk, ne = data.shape
    data_clean = data.astype(float).copy()

    if method == 'fft':
        # ── Spectre de référence (moyenne sur énergie) ─────────────────
        if ev_ref_range is not None and kpar is not None:
            pass  # pas utilisé ici (slice sur la dim énergie non fournie)
        ref_profile = np.mean(data, axis=1).astype(float)

        # ── FFT 1D le long de k ────────────────────────────────────────
        fft_ref  = np.fft.rfft(ref_profile)
        freqs    = np.fft.rfftfreq(nk)
        power    = np.abs(fft_ref)**2
        power[0] = 0.0   # ignorer DC

        # ── Auto-détection de la fréquence de grille ───────────────────
        if grid_freq is None:
            idx_peak = int(np.argmax(power))
            grid_freq = float(freqs[idx_peak])
        else:
            idx_peak = int(np.argmin(np.abs(freqs - grid_freq)))

        # ── Construction du filtre notch ───────────────────────────────
        filt = np.ones(len(freqs))
        lo   = max(0, idx_peak - notch_width)
        hi   = min(len(freqs) - 1, idx_peak + notch_width)
        if notch_sigma > 0:
            for i in range(lo, hi + 1):
                filt[i] = 1.0 - np.exp(-0.5 * ((i - idx_peak) / notch_sigma)**2
                                        * (notch_width**2))
            filt[idx_peak] = 0.0
        else:
            filt[lo:hi + 1] = 0.0

        # ── Application à chaque tranche en énergie ────────────────────
        for ie in range(ne):
            fft_slice = np.fft.rfft(data[:, ie])
            data_clean[:, ie] = np.fft.irfft(fft_slice * filt, n=nk)

        grid_period_px = 1.0 / grid_freq if grid_freq > 0 else np.inf

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].semilogy(freqs[1:], power[1:], 'k-', lw=0.8)
            axes[0].axvline(grid_freq, color='red', lw=1.5, ls='--',
                            label=f'grid f={grid_freq:.4f} (T={grid_period_px:.1f} px)')
            axes[0].set(xlabel='Frequency (cycles/px)', ylabel='Power',
                        title='Fourier Spectrum of the Mean Profile')
            axes[0].legend()
            axes[1].plot(freqs, filt, 'b-', lw=1.5)
            axes[1].set(xlabel='Frequency', ylabel='Filter', title='Applied Notch Filter',
                        ylim=(-0.05, 1.05))
            plt.tight_layout()
            plt.show()

    else:  # method == 'profile'
        # ── Profil angulaire moyen (lissé) ─────────────────────────────
        ref_profile = np.mean(data, axis=1).astype(float)
        ref_smooth  = gaussian_filter1d(ref_profile, sigma=max(nk // 20, 3))
        ref_norm    = ref_smooth / (ref_smooth.mean() or 1.0)

        for ie in range(ne):
            data_clean[:, ie] = data[:, ie] / (ref_norm + 1e-10)

        grid_freq      = None
        grid_period_px = None

        if plot and kpar is not None:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(kpar, ref_profile / ref_profile.max(), 'k-', lw=0.8,
                    label='raw profile')
            ax.plot(kpar, ref_smooth / ref_smooth.max(), 'r-', lw=1.5,
                    label='smoothed profile (divisor)')
            ax.set(xlabel='k// (pi/a)', ylabel='I norm.',
                   title='Mean Angular Profile Used for Correction')
            ax.legend()
            plt.tight_layout()
            plt.show()

    return data_clean, dict(method=method, grid_freq=grid_freq,
                            grid_period_px=grid_period_px)


# =============================================================================
#  7. Normalisation par profil angulaire du detecteur
#     Inspire de normalization_profile() — ARPES_analyzer Igor
# =============================================================================


def normalize_by_profile(data_cut, ev_arr, ev_min, ev_max, smooth=False):
    """
    Normalise une coupe E(k) par le profil angulaire du detecteur.

    Calcule l'intensite integree sur une fenetre en energie haute
    (zone sans structure de bande, ex: 1-3 eV sous EF) pour extraire
    l'efficacite pixel-par-pixel du detecteur, puis divise la carte entiere
    par ce profil. Corrige les asymetries gauche/droite artificielles.

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    ev_arr   : np.ndarray 1D (energies)
    ev_min, ev_max : float
        Fenetre d'integration (ex: ev_min=-2.0, ev_max=-0.5 eV sous EF).
        Choisir une zone plate sans structure de bande.
    smooth   : bool
        Si True, lisse le profil avant normalisation (reduit le bruit).

    Retourne
    --------
    data_norm : np.ndarray (nk, ne)  — donnees normalisees
    profile   : np.ndarray (nk,)     — profil angulaire utilise
    """
    mask = (ev_arr >= ev_min) & (ev_arr <= ev_max)
    if mask.sum() == 0:
        raise ValueError(f"Fenetre [{ev_min}, {ev_max}] eV vide dans ev_arr.")

    # Profil : moyenne de l'intensite sur la fenetre energie (axe 1)
    profile = data_cut[:, mask].mean(axis=1)
    profile = np.where(profile > 0, profile, 1.0)  # evite division par zero

    if smooth:
        profile = gaussian_filter1d(profile.astype(float), sigma=3)

    # Normalisation : chaque colonne k divisee par son efficacite
    profile_norm = profile / profile.mean()          # ramene autour de 1
    data_norm = data_cut / profile_norm[:, np.newaxis]

    return data_norm, profile_norm


# =============================================================================
#  8. Soustraction fond Shirley iteratif
#     Inspire de background_shirley() — ARPES_analyzer Igor
# =============================================================================


def shirley_background(edc, ev_arr, ev_lo, ev_hi, n_iter=10):
    """
    Calcule le fond Shirley iteratif sur un EDC (spectre 1D en energie).

    Le fond de Shirley est proportionnel a l'integrale du spectre au-dessus
    de chaque point en energie : BG(E) ∝ ∫[E, E_hi] [I(E') - BG(E')] dE'.
    Itere jusqu'a convergence (typiquement 5-15 iterations).

    Parametres
    ----------
    edc    : np.ndarray 1D — intensite en fonction de l'energie
    ev_arr : np.ndarray 1D — axe energie (eV)
    ev_lo  : float — borne basse de la fenetre Shirley (eV)
    ev_hi  : float — borne haute (typiquement juste sous EF)
    n_iter : int   — nombre d'iterations (defaut 10)

    Retourne
    --------
    bg : np.ndarray 1D — fond Shirley sur toute la plage ev_arr
    """
    i_lo = int(np.argmin(np.abs(ev_arr - ev_lo)))
    i_hi = int(np.argmin(np.abs(ev_arr - ev_hi)))
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo

    de   = float(ev_arr[1] - ev_arr[0])
    seg  = edc[i_lo:i_hi + 1].copy().astype(float)
    n    = len(seg)

    bg = np.zeros(n)
    I_lo, I_hi = seg[0], seg[-1]

    for _ in range(n_iter):
        bg_new = np.zeros(n)
        for i in range(n):
            integral = np.trapz(seg[i:] - bg[i:], dx=de)
            total    = np.trapz(seg - bg, dx=de)
            if abs(total) > 1e-30:
                bg_new[i] = (I_hi - I_lo) * integral / total + I_lo
        bg = bg_new

    # Etend le fond sur toute la plage ev_arr (plat aux extremites)
    full_bg = np.full_like(edc, float(I_lo), dtype=float)
    full_bg[i_lo:i_hi + 1] = bg
    full_bg[i_hi + 1:]     = I_hi

    return full_bg


def subtract_shirley(data_cut, ev_arr, ev_lo, ev_hi, n_iter=10, mode='linear'):
    """
    Soustrait le fond inélastique de chaque EDC d'une coupe 2D E(k).

    mode='shirley' : fond Shirley itératif (adapté XPS, peut diverger en ARPES
                     si la bande peak dans la fenêtre [ev_lo, ev_hi]).
    mode='linear'  : interpolation linéaire entre I(ev_lo) et I(ev_hi) —
                     plus stable pour ARPES près de EF (recommandé).

    Parametres
    ----------
    data_cut     : np.ndarray (nk, ne)
    ev_arr       : np.ndarray 1D
    ev_lo, ev_hi : float — bornes de la fenêtre (eV)
    n_iter       : int   — iterations Shirley (ignoré si mode='linear')
    mode         : str   — 'linear' | 'shirley'

    Retourne
    --------
    data_corr : np.ndarray (nk, ne) — données sans fond
    bg_map    : np.ndarray (nk, ne) — carte des fonds soustraits
    """
    data_corr = np.empty_like(data_cut, dtype=float)
    bg_map    = np.empty_like(data_cut, dtype=float)

    i_lo = int(np.argmin(np.abs(ev_arr - ev_lo)))
    i_hi = int(np.argmin(np.abs(ev_arr - ev_hi)))
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo

    for ik in range(data_cut.shape[0]):
        edc = data_cut[ik, :].astype(float)
        if mode == 'shirley':
            bg = shirley_background(edc, ev_arr, ev_lo, ev_hi, n_iter)
        else:  # linear
            I_lo_val = edc[i_lo]
            I_hi_val = edc[i_hi]
            bg = np.full_like(edc, I_lo_val)
            n_seg = i_hi - i_lo + 1
            if n_seg > 1:
                bg[i_lo:i_hi + 1] = np.linspace(I_lo_val, I_hi_val, n_seg)
            bg[i_hi + 1:] = I_hi_val
        bg_map[ik, :]    = bg
        data_corr[ik, :] = np.maximum(edc - bg, 0.0)

    return data_corr, bg_map


# =============================================================================
#  9. Division par distribution Fermi-Dirac -> acces a A(k,E)
#     Inspire de fermi_div_fun() — ARPES_analyzer Igor
# =============================================================================


def fermi_dirac_divide(data_cut, ev_arr, ef, temperature, cutoff_ev=0.05):
    """
    Divise les donnees ARPES par la distribution Fermi-Dirac pour
    extraire la densite spectrale A(k,E) au-dessus de EF.

    The data measure N(k,E) = A(k,E) x f(E,T).
    Division removes the thermal cutoff and gives access to A(k,E)
    up to cutoff_ev above EF.

    Warning: amplifies noise above EF; use on sufficiently smooth
    data or at low temperature.

    Parameters
    ----------
    data_cut    : np.ndarray (nk, ne)
    ev_arr      : np.ndarray 1D — energies en eV (referees a EF=0)
    ef          : float — position de EF dans ev_arr (souvent 0.0)
    temperature : float — temperature en Kelvin
    cutoff_ev   : float — limite au-dessus de EF (eV) jusqu'ou diviser.
                  Au-dela, les donnees sont mises a zero.

    Retourne
    --------
    data_fd : np.ndarray (nk, ne) — donnees corrigees FD, >= 0
    fd      : np.ndarray (ne,)    — distribution FD utilisee
    """
    kT  = 8.6173303e-5 * temperature          # eV
    dE  = ev_arr - ef                          # energie relative a EF

    # FD(E) = 1 / (exp((E-EF)/kT) + 1)
    fd  = 1.0 / (np.exp(np.clip(dE / kT, -500, 500)) + 1.0)

    # Indice de coupure : ne pas diviser trop loin au-dessus de EF
    i_cut = int(np.argmin(np.abs(dE - cutoff_ev)))
    fd_safe = fd.copy()
    fd_safe[i_cut:] = np.inf                   # on mettra ces points a zero

    # Division : I_corr = I × (1 + exp(ΔE/kT)) = I / FD
    with np.errstate(divide='ignore', invalid='ignore'):
        data_fd = data_cut / fd_safe[np.newaxis, :]

    data_fd = np.nan_to_num(data_fd, nan=0.0, posinf=0.0)
    data_fd = np.maximum(data_fd, 0.0)

    return data_fd, fd


# =============================================================================
# 10. Reconstruction dispersion en kz (scans en energie de photon)
#     Inspire de kz_disperssion_multwv() — ARPES_analyzer Igor
# =============================================================================


def kz_dispersion(cuts, theta_arr, hv_arr, ef_arr, work_function, V0,
                  phi=0.0, a_lattice=None, nkx=200, nkz=200):
    """
    Reconstruit la carte kx-kz a partir d'une serie de coupes MDC
    mesurees a differentes energies de photon hv.

    Formules (const = 0.512 Å^-1 eV^-1/2) :
        Ekin(hv) = hv - work_function + ef_arr[i]   (energie cinetique a EF)
        kx = const × sqrt(Ekin) × sin(theta - phi)
        kz = const × sqrt(Ekin × cos²(theta) + V0)

    En mode pi/a : const = 0.512 × a_lattice / pi

    Parametres
    ----------
    cuts         : list de np.ndarray (ntheta, ne) — une coupe par hv
    theta_arr    : np.ndarray 1D — angles en degres (meme pour tous)
    hv_arr       : np.ndarray 1D — energies de photon (eV)
    ef_arr       : np.ndarray 1D — EF mesure pour chaque hv (eV)
    work_function: float — fonction de travail (eV), typiquement 4-5 eV
    V0           : float — potentiel interieur (eV), typiquement 10-15 eV
    phi          : float — angle azimuthal de centrage (deg, defaut 0)
    a_lattice    : float ou None — parametre de reseau (Å).
                   Si None -> unites Å^-1. Si fourni -> unites pi/a.
    nkx, nkz     : int — resolution de la grille de sortie

    Retourne
    --------
    dict avec cles :
        'map'    : np.ndarray (nkx, nkz) — intensite interpolee
        'kx_arr' : np.ndarray (nkx,)
        'kz_arr' : np.ndarray (nkz,)
        'unit'   : str — '1/A' ou 'pi/a'
    """
    const = 0.512  # Å^-1 eV^-1/2
    if a_lattice is not None:
        const = 0.512 * a_lattice / np.pi
        unit  = 'pi/a'
    else:
        unit  = '1/A'

    theta_rad = np.deg2rad(theta_arr - phi)

    # Collecte de tous les points (kx, kz, I_at_EF)
    kx_pts, kz_pts, I_pts = [], [], []

    for i, (cut, hv, ef) in enumerate(zip(cuts, hv_arr, ef_arr)):
        ekin = hv - work_function + ef        # energie cinetique a EF
        if ekin <= 0:
            continue

        # kx et kz pour chaque angle
        kx = const * np.sqrt(ekin) * np.sin(theta_rad)            # (ntheta,)
        kz = const * np.sqrt(ekin * np.cos(theta_rad)**2 + V0)    # (ntheta,)

        # Intensite a EF : moyenne sur une fenetre ±10 meV
        ie_ef = int(np.argmin(np.abs(np.arange(cut.shape[1]) - cut.shape[1]//2)))
        I_ef  = cut.mean(axis=1)  # moyenne en energie comme approximation

        kx_pts.append(kx)
        kz_pts.append(kz)
        I_pts.append(I_ef)

    if not kx_pts:
        raise ValueError("No valid cut (check hv, work_function, V0).")

    kx_all = np.concatenate(kx_pts)
    kz_all = np.concatenate(kz_pts)
    I_all  = np.concatenate(I_pts)

    # Grille reguliere de sortie
    kx_min, kx_max = kx_all.min(), kx_all.max()
    kz_min, kz_max = kz_all.min(), kz_all.max()
    kx_grid = np.linspace(kx_min, kx_max, nkx)
    kz_grid = np.linspace(kz_min, kz_max, nkz)
    KX, KZ  = np.meshgrid(kx_grid, kz_grid, indexing='ij')

    # Interpolation par triangulation (griddata gere les nuages de points)
    from scipy.interpolate import griddata
    I_map = griddata(
        np.column_stack([kx_all, kz_all]),
        I_all,
        (KX, KZ),
        method='linear',
        fill_value=0.0,
    )

    return dict(map=I_map, kx_arr=kx_grid, kz_arr=kz_grid, unit=unit)


# =============================================================================
# 11. Fit EDC : Lorentzien/Gaussien/Voigt convoluee avec Fermi-Dirac
#     Inspire de bfv2_fitting_functions.ipf — ARPES_analyzer Igor
# =============================================================================
