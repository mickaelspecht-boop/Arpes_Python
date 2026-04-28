"""
arpes_plots.py - Fonctions d'analyse ARPES pour Ba122

Pipeline:
  0. fit_fermi_edge()       -> Calibration EF : fit FD sur EDC moyennee (gold ou sample)
  1. find_gamma_mdc()       -> Position de Gamma par milieu des kF
  2. extract_cut()          -> Coupe E(k) le long d'un chemin arbitraire
  3. mdc_waterfall()        -> Waterfall de MDCs (I vs k a differentes energies)
  4. secdev_curvature()     -> Seconde derivee + courbure 2D (Zhang 2011)
  5. fit_mdc_lorentzians()  -> Fit N Lorentziennes + Hungarian matching
  5b. debug_mdc_fit()       -> Diagnostic MDC — une tranche, plot fit vs donnees
  5b. plot_mdc_fit_panel()  -> Grille de diagnostics MDC sur N energies
  5b. plot_mdc_waterfall_with_fit() -> Carte 2D + waterfall apres fit
  6. fit_mdc_regions()      -> Fit Lorentziennes par regions k definies manuellement

  Corrections (inspire de ARPES_analyzer Igor — colleque):
  7. normalize_by_profile() -> Normalisation par profil angulaire du detecteur
  8. shirley_background()   -> Soustraction fond Shirley iteratif (par EDC)
  9. fermi_dirac_divide()   -> Division par distribution Fermi-Dirac -> A(k,E)
 10. kz_dispersion()        -> Reconstruction dispersion en kz (scans hv)
 11. fit_edc()              -> Fit EDC : Gauss/Lor/Voigt convoluee avec FD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


# =============================================================================
#  0. Calibration EF — fit de la distribution Fermi-Dirac sur une EDC
# =============================================================================

_kB = 8.617333e-5   # eV/K

def fit_fermi_edge(
    ev_arr,
    I_arr,
    temperature_K=100,
    fit_range=(-0.3, 0.1),
    sigma_resolution_init=0.025,
    fix_kBT=True,
    units='binding',
    ax=None,
    title=None,
    verbose=True,
):
    """
    Fit la distribution Fermi-Dirac convoluee avec une Gaussienne
    (resolution instrumentale) sur une EDC moyennee.

    Modele :
        I(E) = A × [FD(E; EF, kBT) ⊗ Gauss(σ_res)] + slope×E + bg

    Peut etre utilise sur :
    - Une EDC d'un fichier gold (reference absolue) — meilleure precision
    - Une EDC moyennee sur tous les angles du sample — auto-calibration

    Parametres
    ----------
    ev_arr : np.ndarray 1D
        Axe energie.
        - Si units='binding' : valeurs en eV relatif a EF nominal (0 = EF suppose)
          → EF retourné = offset residuel par rapport a 0
        - Si units='kinetic' : valeurs en eV cinetique
          → EF retourné = EF_kin absolu (ex: 95.5 eV)
    I_arr : np.ndarray 1D
        Intensite de l'EDC (sera normalisee en interne a max=1).
    temperature_K : float
        Temperature de mesure en K — definit kBT = kB × T.
        Si fix_kBT=True, kBT est fixe a cette valeur (recommande si T est connue).
        Si fix_kBT=False, kBT est un parametre libre (utile si T inconnue).
    fit_range : tuple (float, float)
        Plage en energie pour le fit.
        En binding energy : typiquement (-0.3, +0.1) eV.
        En kinetic energy : typiquement (EF_kin_nominal - 0.3, EF_kin_nominal + 0.1).
    sigma_resolution_init : float
        Valeur initiale pour la resolution instrumentale (eV FWHM / 2.355).
        Typique synchrotron 100 eV : 15–30 meV. Défaut 25 meV.
    fix_kBT : bool
        True (recommande) = kBT fixe a kB×temperature_K, seule σ est libre.
        False = kBT libre aussi (utile si la temperature exacte est incertaine).
    units : str
        'binding' ou 'kinetic' — affecte l'interpretation du resultat.
    ax : matplotlib.Axes ou None
    title : str ou None

    Retourne
    --------
    dict :
        'EF'         : float — position du bord de Fermi ajuste
                       En binding : offset residuel (ideal = 0)
                       En kinetic : EF_kin absolu
        'EF_err'     : float — incertitude (1σ) sur EF (eV)
        'sigma_res'  : float — resolution instrumentale ajustee (eV)
        'fwhm_res'   : float — FWHM = 2.355 × sigma_res (eV)
        'kBT_eff'    : float — kBT utilise dans le fit (eV)
        'T_eff_K'    : float — temperature effective = kBT_eff / kB
        'residual'   : float — rms des residus normalises
        'phi_eff'    : float ou None — si units='kinetic' et hv fourni :
                       phi_eff = hv - EF_kin (eV)
        'popt'       : array — parametres optimises
        'model_ev'   : np.ndarray — axe E pour la courbe modelisee
        'model_I'    : np.ndarray — intensite modelisee
        'ax'         : matplotlib.Axes
    """
    from scipy.ndimage import gaussian_filter1d

    kBT_fixed = _kB * temperature_K

    # --- restriction a la plage de fit ---
    mask = (ev_arr >= fit_range[0]) & (ev_arr <= fit_range[1])
    if mask.sum() < 10:
        raise ValueError(
            f"fit_fermi_edge : seulement {mask.sum()} points dans fit_range={fit_range}. "
            "Elargir fit_range ou verifier les unites de ev_arr."
        )
    e = ev_arr[mask].astype(float)
    I = I_arr[mask].astype(float)
    I = np.where(np.isfinite(I), I, 0.0)
    I_norm = I / (I.max() or 1.0)   # normalise a [0, 1] environ
    dE = abs(float(e[1] - e[0]))

    # --- modele FD convoluee ---
    def _fd_conv(e_arr, EF, A, slope, bg, sigma):
        """FD convoluee avec Gaussienne de sigma en eV."""
        kBT = kBT_fixed  # closure sur kBT_fixed ou kBT_free
        raw = np.exp(np.clip((e_arr - EF) / kBT, -500, 500))
        fd  = 1.0 / (raw + 1.0)
        pix = sigma / dE
        if pix > 0.3:
            fd = gaussian_filter1d(fd, sigma=pix)
        return A * fd + slope * (e_arr - EF) + bg

    def _fd_conv_free_T(e_arr, EF, A, slope, bg, sigma, kBT_free):
        """Version avec kBT libre."""
        raw = np.exp(np.clip((e_arr - EF) / max(kBT_free, 1e-4), -500, 500))
        fd  = 1.0 / (raw + 1.0)
        pix = sigma / dE
        if pix > 0.3:
            fd = gaussian_filter1d(fd, sigma=pix)
        return A * fd + slope * (e_arr - EF) + bg

    # --- guess initial : EF a la position du gradient max ---
    grad = np.gradient(I_norm, e)
    i_ef = int(np.argmax(np.abs(grad)))
    EF_guess = float(e[i_ef])

    if fix_kBT:
        model_fn = _fd_conv
        p0     = [EF_guess, 0.8,  -0.3, 0.05, sigma_resolution_init]
        lo     = [EF_guess - 0.15, 0.0, -5.0, 0.0,  0.003]
        hi     = [EF_guess + 0.15, 3.0,  5.0, 1.0,  0.10 ]
    else:
        model_fn = _fd_conv_free_T
        p0     = [EF_guess, 0.8, -0.3, 0.05, sigma_resolution_init, kBT_fixed]
        lo     = [EF_guess - 0.15, 0.0, -5.0, 0.0,  0.003, kBT_fixed * 0.5]
        hi     = [EF_guess + 0.15, 3.0,  5.0, 1.0,  0.10,  kBT_fixed * 3.0]

    try:
        popt, pcov = curve_fit(model_fn, e, I_norm, p0=p0,
                               bounds=(lo, hi), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        success = True
    except Exception as exc:
        if verbose:
            print(f"[fit_fermi_edge] Echec : {exc}")
        # fallback : EF au gradient max
        popt = np.array(p0)
        perr = np.full(len(p0), np.nan)
        success = False

    EF_fit    = float(popt[0])
    EF_err    = float(perr[0])
    sigma_fit = float(popt[4])
    fwhm_fit  = sigma_fit * 2.355
    kBT_eff   = kBT_fixed if fix_kBT else float(popt[5])
    T_eff     = kBT_eff / _kB

    # courbe modelisee sur toute la plage
    e_model = np.linspace(fit_range[0], fit_range[1], 500)
    I_model = model_fn(e_model, *popt)

    # residus
    I_fitted = model_fn(e, *popt)
    residual = float(np.sqrt(np.mean((I_norm - I_fitted)**2)))

    if verbose:
        print(f"{'✓' if success else '✗'} Fit EF   "
              f"EF = {EF_fit:+.4f} eV  ±{EF_err*1000:.1f} meV  "
              f"| FWHM_res = {fwhm_fit*1000:.0f} meV  "
              f"| T_eff = {T_eff:.0f} K  (T_nominal={temperature_K:.0f} K)  "
              f"| résidu = {residual:.4f}")
        if units == 'binding' and abs(EF_fit) > 0.05:
            print(f"  ⚠ EF offset = {EF_fit*1000:+.0f} meV — l'axe energie est decale de cette valeur.")
            print(f"    → Appliquer ev_arr = ev_arr - ({EF_fit:.4f}) pour corriger.")

    # --- figure ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(e, I_norm, 'k-', lw=1.5, label='EDC (normalisee)')
    ax.plot(e_model, I_model, 'r-', lw=2,
            label=f'Fit FD  EF={EF_fit:+.4f} eV  FWHM={fwhm_fit*1000:.0f} meV')

    # FD pure (sans convolution) pour reference
    raw_fd = 1.0 / (np.exp(np.clip((e_model - EF_fit) / kBT_eff, -500, 500)) + 1.0)
    ax.plot(e_model, raw_fd * float(popt[1]) + float(popt[3]),
            'b--', lw=0.8, alpha=0.5, label=f'FD pure  T={T_eff:.0f} K')

    ax.axvline(EF_fit, color='red', lw=1.2, ls='-', alpha=0.8)
    ax.axvspan(EF_fit - EF_err, EF_fit + EF_err,
               color='red', alpha=0.15, label=f'±{EF_err*1000:.1f} meV')
    ax.axvline(0.0, color='gray', lw=0.8, ls='--', alpha=0.5, label='EF nominal (0)')

    # residus
    ax2 = ax.twinx()
    ax2.plot(e, I_norm - I_fitted, color='#888888', lw=0.7, alpha=0.6)
    ax2.axhline(0, color='gray', lw=0.5, ls='--')
    ax2.set_ylabel('Résidu', color='gray', fontsize=8)
    ax2.tick_params(axis='y', colors='gray', labelsize=7)
    ax2.set_ylim(-0.4, 0.4)

    ttl = title or ('EDC calibration EF' + (' [ECHEC]' if not success else ''))
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('E (eV)')
    ax.set_ylabel('Intensite normalisee')
    ax.legend(fontsize=8, loc='center right')
    ax.set_xlim(fit_range)

    # annotations texte
    ax.text(0.02, 0.05,
            f"EF = {EF_fit:+.4f} eV  ±{EF_err*1000:.1f} meV\n"
            f"FWHM = {fwhm_fit*1000:.0f} meV   T_eff = {T_eff:.0f} K\n"
            f"résidu rms = {residual:.4f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    return dict(
        success=success, EF=EF_fit, EF_err=EF_err,
        sigma_res=sigma_fit, fwhm_res=fwhm_fit,
        kBT_eff=kBT_eff, T_eff_K=T_eff,
        residual=residual,
        popt=popt, perr=perr,
        model_ev=e_model, model_I=I_model,
        ax=ax,
    )


# =============================================================================
#  1. Detection de Gamma par milieu des kF (MDC midpoint)
# =============================================================================

def _two_lorentzians(k, bg_a, bg_b, k1, g1, A1, k2, g2, A2):
    """Modele : 2 Lorentziennes + fond lineaire."""
    return (bg_a * k + bg_b
            + A1 * g1**2 / ((k - k1)**2 + g1**2)
            + A2 * g2**2 / ((k - k2)**2 + g2**2))


def _fit_kf_pair(k_arr, mdc, k_range=(-0.6, 0.6)):
    """Fit 2 Lorentziennes sur une MDC, retourne (kF_gauche, kF_droite)."""
    mask = (k_arr >= k_range[0]) & (k_arr <= k_range[1])
    k = k_arr[mask]
    I = np.where(np.isfinite(mdc[mask]), mdc[mask], 0.0)

    I_sm = gaussian_filter1d(I, sigma=3)
    pks_idx, props = find_peaks(I_sm, height=np.nanmax(I_sm) * 0.15,
                                distance=len(k) // 6)
    if len(pks_idx) < 2:
        return np.nan, np.nan

    idx_sort = np.argsort(-I_sm[pks_idx])
    p1, p2 = sorted(pks_idx[idx_sort[:2]])

    k1g, k2g = float(k[p1]), float(k[p2])
    A_max = float(np.nanmax(I_sm))
    p0 = [0.0, float(np.nanmedian(I)),
          k1g, 0.03, A_max * 0.5,
          k2g, 0.03, A_max * 0.5]

    try:
        popt, _ = curve_fit(_two_lorentzians, k, I, p0=p0, maxfev=5000)
        kF_L, kF_R = sorted([popt[2], popt[5]])
        if k_range[0] < kF_L < kF_R < k_range[1]:
            return kF_L, kF_R
    except RuntimeError:
        pass
    return np.nan, np.nan


def find_gamma_mdc(da_k_pi, fs_window_ev=0.01, k_range=(-0.6, 0.6),
                   n_ky_scan=15, ky_range=(-0.15, 0.15),
                   n_kx_scan=15, kx_range=(-0.05, 0.25)):
    """
    Detecte le point Gamma par la methode du milieu des kF.

    Principe : la MDC a EF montre deux pics a Gamma-kF et Gamma+kF.
    Leur milieu donne Gamma. On scanne en kx et ky pour obtenir
    une estimation stable (mediane).

    Parametres
    ----------
    da_k_pi : xr.DataArray
        Donnees en k-space (dims: kx, ky, eV), axes en pi/a.
    fs_window_ev : float
        Demi-fenetre autour de EF pour integrer la FS.
    k_range : tuple
        Fenetre en k pour la recherche des pics.
    n_ky_scan, ky_range : int, tuple
        Nombre et plage de coupes ky pour estimer Gamma_kx.
    n_kx_scan, kx_range : int, tuple
        Nombre et plage de coupes kx pour estimer Gamma_ky.

    Retourne
    --------
    gamma : dict
        {'kx': float, 'ky': float,
         'kf_pairs_kx': list, 'kf_pairs_ky': list,
         'gamma_kx_list': list, 'gamma_ky_list': list}
    """
    _fs_ef = da_k_pi.sel(eV=slice(-fs_window_ev, fs_window_ev)).mean('eV')
    kx_arr = _fs_ef.kx.values
    ky_arr = _fs_ef.ky.values

    # Scan en kx : MDC le long de kx pour differents ky
    ky_candidates = np.linspace(ky_range[0], ky_range[1], n_ky_scan)
    gamma_kx_list = []
    kf_pairs_kx = []

    for ky_val in ky_candidates:
        mdc = _fs_ef.sel(ky=ky_val, method='nearest').values
        kF_L, kF_R = _fit_kf_pair(kx_arr, mdc, k_range=k_range)
        if np.isfinite(kF_L):
            gamma_kx_list.append((kF_L + kF_R) / 2)
            kf_pairs_kx.append((ky_val, kF_L, kF_R))

    # Scan en ky : MDC le long de ky pour differents kx
    kx_candidates = np.linspace(kx_range[0], kx_range[1], n_kx_scan)
    gamma_ky_list = []
    kf_pairs_ky = []

    for kx_val in kx_candidates:
        mdc = _fs_ef.sel(kx=kx_val, method='nearest').values
        kF_L, kF_R = _fit_kf_pair(ky_arr, mdc, k_range=k_range)
        if np.isfinite(kF_L):
            gamma_ky_list.append((kF_L + kF_R) / 2)
            kf_pairs_ky.append((kx_val, kF_L, kF_R))

    gamma_kx = float(np.nanmedian(gamma_kx_list)) if gamma_kx_list else np.nan
    gamma_ky = float(np.nanmedian(gamma_ky_list)) if gamma_ky_list else np.nan

    print(f"Gamma par milieu des kF :")
    print(f"  kx = {gamma_kx:+.4f} pi/a  ({len(gamma_kx_list)} fits)")
    print(f"  ky = {gamma_ky:+.4f} pi/a  ({len(gamma_ky_list)} fits)")

    return dict(
        kx=gamma_kx, ky=gamma_ky,
        kf_pairs_kx=kf_pairs_kx, kf_pairs_ky=kf_pairs_ky,
        gamma_kx_list=gamma_kx_list, gamma_ky_list=gamma_ky_list,
    )


def plot_gamma_detection(gamma_result, da_k_pi, fs_window_ev=0.01, cmap='viridis'):
    """
    Figure 3 panneaux : FS + paires kF, stabilite Gamma_kx, stabilite Gamma_ky.
    """
    gr = gamma_result
    _fs = da_k_pi.sel(eV=slice(-fs_window_ev, fs_window_ev)).mean('eV')
    kx_arr = _fs.kx.values
    ky_arr = _fs.ky.values
    I_plot = _fs.values
    if I_plot.shape != (len(ky_arr), len(kx_arr)):
        I_plot = I_plot.T
    vm = float(np.nanpercentile(np.abs(I_plot), 98))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panneau 1 : FS + Gamma + paires kF
    axes[0].pcolormesh(kx_arr, ky_arr, I_plot, cmap=cmap,
                       vmin=0, vmax=vm, shading='nearest')
    for ky_v, kL, kR in gr['kf_pairs_kx']:
        axes[0].plot([kL, kR], [ky_v, ky_v], 'r.-', ms=5, lw=0.8)
    for kx_v, kL, kR in gr['kf_pairs_ky']:
        axes[0].plot([kx_v, kx_v], [kL, kR], 'c.-', ms=5, lw=0.8)
    axes[0].plot(gr['kx'], gr['ky'], '*', color='red', ms=16, zorder=6,
                 label=f"Gamma ({gr['kx']:+.3f}, {gr['ky']:+.3f})")
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('kx (pi/a)'); axes[0].set_ylabel('ky (pi/a)')
    axes[0].set_title('FS + paires kF')
    axes[0].legend(fontsize=8)

    # Panneau 2 : stabilite Gamma_kx
    if gr['gamma_kx_list']:
        ky_vals = [p[0] for p in gr['kf_pairs_kx']]
        axes[1].plot(ky_vals, gr['gamma_kx_list'], 'ro-', ms=4)
        axes[1].axhline(gr['kx'], color='red', ls='--', lw=1,
                        label=f"mediane = {gr['kx']:+.4f}")
        axes[1].set_xlabel('ky de la coupe (pi/a)')
        axes[1].set_ylabel('Gamma_kx (pi/a)')
        axes[1].set_title('Stabilite Gamma_kx')
        axes[1].legend(fontsize=8)

    # Panneau 3 : stabilite Gamma_ky
    if gr['gamma_ky_list']:
        kx_vals = [p[0] for p in gr['kf_pairs_ky']]
        axes[2].plot(kx_vals, gr['gamma_ky_list'], 'co-', ms=4)
        axes[2].axhline(gr['ky'], color='cyan', ls='--', lw=1,
                        label=f"mediane = {gr['ky']:+.4f}")
        axes[2].set_xlabel('kx de la coupe (pi/a)')
        axes[2].set_ylabel('Gamma_ky (pi/a)')
        axes[2].set_title('Stabilite Gamma_ky')
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig, axes


def estimate_gamma_bm_mdc(
    data_cut,
    kpar,
    ev_arr,
    ev_range=(-0.45, -0.05),
    k_range=(-1.0, 1.0),
    center_guess=0.0,
    center_window=0.35,
    smooth_sigma=2.0,
    prominence=0.04,
    min_separation=0.08,
    max_asymmetry=0.35,
    min_points=3,
    verbose=True,
):
    """
    Estime Gamma dans une band map 2D par symetrie des MDC.

    Pour chaque energie de la fenetre, on detecte une paire de pics gauche/droite
    et on prend son milieu. Le Gamma final est la mediane robuste de ces milieux.
    C'est plus rigoureux qu'une valeur de theta seule quand la BM contient des
    branches visibles autour de Gamma.
    """
    data = np.asarray(data_cut, dtype=float)
    k = np.asarray(kpar, dtype=float)
    ev = np.asarray(ev_arr, dtype=float)
    if data.shape != (len(k), len(ev)):
        data = np.squeeze(data)
        if data.shape == (len(ev), len(k)):
            data = data.T
    if data.shape != (len(k), len(ev)):
        raise ValueError(f"Shape data invalide: {data.shape}, attendu ({len(k)}, {len(ev)})")

    ev_lo, ev_hi = sorted(ev_range)
    k_lo, k_hi = sorted(k_range)
    e_idx = np.where((ev >= ev_lo) & (ev <= ev_hi))[0]
    k_mask = (k >= k_lo) & (k <= k_hi)
    if len(e_idx) == 0:
        raise ValueError(f"Aucune energie dans ev_range={ev_range}")
    if k_mask.sum() < 20:
        raise ValueError(f"Pas assez de points k dans k_range={k_range}")

    kk = k[k_mask]
    centers, pairs, scores = [], [], []
    c_min = center_guess - center_window
    c_max = center_guess + center_window
    dk = abs(float(np.nanmedian(np.diff(kk)))) if len(kk) > 1 else 0.01
    distance = max(1, int(min_separation / max(dk, 1e-9)))

    for ie in e_idx:
        mdc = np.nan_to_num(data[k_mask, ie].astype(float), nan=0.0)
        if smooth_sigma and smooth_sigma > 0:
            mdc = gaussian_filter1d(mdc, sigma=float(smooth_sigma))
        lo, hi = np.nanpercentile(mdc, [5, 99])
        if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
            continue
        y = np.clip((mdc - lo) / (hi - lo), 0, None)
        pk, _ = find_peaks(y, prominence=prominence, distance=distance)
        if len(pk) < 2:
            continue

        pk_k = kk[pk]
        pk_y = y[pk]
        left = np.where((pk_k < center_guess - min_separation / 2) & (pk_k >= k_lo))[0]
        right = np.where((pk_k > center_guess + min_separation / 2) & (pk_k <= k_hi))[0]
        if len(left) == 0 or len(right) == 0:
            continue

        candidates = []
        for il in left:
            for ir in right:
                kl, kr = float(pk_k[il]), float(pk_k[ir])
                center = 0.5 * (kl + kr)
                if not (c_min <= center <= c_max):
                    continue
                sep_l = abs(center - kl)
                sep_r = abs(kr - center)
                asym = abs(sep_l - sep_r) / max(sep_l + sep_r, 1e-12)
                if asym > max_asymmetry:
                    continue
                amp_score = float(pk_y[il] + pk_y[ir])
                sep_score = min(sep_l, sep_r)
                candidates.append((amp_score + 0.2 * sep_score - asym, center, kl, kr, asym))
        if not candidates:
            continue
        _, center, kl, kr, asym = max(candidates, key=lambda x: x[0])
        centers.append(center)
        pairs.append((float(ev[ie]), kl, kr, center))
        scores.append(1.0 - asym)

    centers_arr = np.asarray(centers, dtype=float)
    if centers_arr.size < min_points:
        gamma = np.nan
        mad = np.nan
    else:
        gamma = float(np.nanmedian(centers_arr))
        mad = float(1.4826 * np.nanmedian(np.abs(centers_arr - gamma)))

    if verbose:
        print("Gamma BM par milieux de MDC :")
        print(f"  gamma = {gamma:+.4f} pi/a  |  n={centers_arr.size}  |  MAD={mad:.4f}")

    return {
        "gamma": gamma,
        "center": gamma,
        "mad": mad,
        "n": int(centers_arr.size),
        "centers": centers_arr,
        "pairs": pairs,
        "scores": np.asarray(scores, dtype=float),
        "ev_range": tuple(ev_range),
        "k_range": tuple(k_range),
    }


# =============================================================================
#  2. Extraction de coupe E(k)
# =============================================================================

def extract_cut(da_k_pi, k_start, k_end,
                ev_range=(-0.7, 0.05), npts=400, width=0.05, n_perp=11):
    """
    Interpole une coupe E(k) le long d'un chemin (k_start -> k_end).

    Parametres
    ----------
    da_k_pi : xr.DataArray
        Donnees 3D (kx, ky, eV).
    k_start, k_end : array-like (2,)
        Points de depart et d'arrivee en pi/a.
    ev_range : tuple
        (ev_min, ev_max) en eV.
    npts : int
        Nombre de points le long du chemin.
    width : float
        Largeur d'integration perpendiculaire (pi/a).
    n_perp : int
        Nombre de lignes perpendiculaires pour l'integration.

    Retourne
    --------
    data_cut : np.ndarray (npts, ne)
    kpar : np.ndarray (npts,)     k// centre sur Gamma (milieu du chemin)
    ev_arr : np.ndarray (ne,)
    """
    k_start = np.asarray(k_start, dtype=float)
    k_end = np.asarray(k_end, dtype=float)

    da_ev = da_k_pi.sel(eV=slice(ev_range[0], ev_range[1]))
    ev_arr = da_ev.eV.values
    kx_arr = da_ev.kx.values
    ky_arr = da_ev.ky.values
    I3d = da_ev.transpose('kx', 'ky', 'eV').values.astype(float)

    rgi = RegularGridInterpolator(
        (kx_arr, ky_arr, ev_arr), I3d,
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    dk = k_end - k_start
    L = float(np.linalg.norm(dk))
    khat = dk / L if L > 1e-6 else np.array([1.0, 0.0])
    kperp = np.array([-khat[1], khat[0]])

    t = np.linspace(0, L, npts)
    kpar = t - L / 2
    p = np.linspace(-width / 2, width / 2, max(n_perp, 1))

    T, P = np.meshgrid(t, p)
    kx_q = (k_start[0] + T * khat[0] + P * kperp[0]).ravel()
    ky_q = (k_start[1] + T * khat[1] + P * kperp[1]).ravel()
    ne = len(ev_arr)

    pts = np.column_stack([
        np.repeat(kx_q, ne),
        np.repeat(ky_q, ne),
        np.tile(ev_arr, len(kx_q)),
    ])
    vals = rgi(pts).reshape(len(p), npts, ne)
    data_cut = np.nanmean(vals, axis=0)

    return data_cut, kpar, ev_arr


# =============================================================================
#  3. Waterfall de MDCs
# =============================================================================

def mdc_waterfall(
    data_cut, kpar, ev_arr,
    ev_start=-0.5, ev_end=0.0, delta_ev=0.05,
    smooth_sigma=1.5, offset_scale=1.0, normalize="each",
    ax=None, cmap="coolwarm_r", lw=1.2,
    fill=True, fill_alpha=0.25, title=None,
):
    """
    Waterfall de MDCs : une courbe I(k) par tranche en energie.

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    kpar : np.ndarray (nk,)
    ev_arr : np.ndarray (ne,)
    ev_start, ev_end : float
        Fenetre en energie.
    delta_ev : float
        Pas entre deux MDCs.
    smooth_sigma : float
        Lissage gaussien 1D le long de k (pixels).
    offset_scale : float
        Facteur sur le decalage vertical.
    normalize : str
        "each", "global" ou "none".
    ax : Axes ou None
    cmap : str
    lw, fill, fill_alpha, title : display params

    Retourne
    --------
    fig, ax, energies
    """
    energies = []
    ev = ev_start
    while ev <= ev_end + 1e-9:
        energies.append(ev)
        ev += delta_ev
    if not energies:
        raise ValueError(f"Aucune energie dans [{ev_start}, {ev_end}]")

    global_max = 1.0
    if normalize == "global":
        global_max = float(np.nanmax(np.abs(data_cut))) or 1.0

    cmap_fn = plt.get_cmap(cmap)
    n = len(energies)
    colors = [cmap_fn(i / max(n - 1, 1)) for i in range(n)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    for i, ev_i in enumerate(energies):
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc = data_cut[:, ie].astype(float)

        if smooth_sigma > 0:
            mdc = gaussian_filter1d(mdc, sigma=smooth_sigma)

        if normalize == "each":
            mdc_range = mdc.max() - mdc.min()
            if mdc_range > 0:
                mdc = (mdc - mdc.min()) / mdc_range
        elif normalize == "global":
            mdc = mdc / global_max

        offset = i * delta_ev * offset_scale
        y = mdc + offset
        c = colors[i]

        ax.plot(kpar, y, color=c, lw=lw, label=f"{ev_arr[ie]:.3f} eV", zorder=n - i)
        if fill:
            ax.fill_between(kpar, offset, y, color=c,
                            alpha=fill_alpha, zorder=n - i)
        ax.axhline(offset, color=c, lw=0.4, ls='--', alpha=0.4, zorder=0)

    tick_step = max(1, n // 10)
    tick_idx = list(range(0, n, tick_step))
    if tick_idx[-1] != n - 1:
        tick_idx.append(n - 1)
    ax.set_yticks([i * delta_ev * offset_scale for i in tick_idx])
    ax.set_yticklabels([f"{energies[i]:.3f}" for i in tick_idx], fontsize=8)

    ax.set_xlabel("k// (pi/a)", fontsize=11)
    ax.set_ylabel("E - EF (eV)", fontsize=11)
    ax.set_xlim(kpar[0], kpar[-1])
    ax.set_title(title or f"Waterfall MDC  [{ev_start:.2f} -> {ev_end:.2f} eV]",
                 fontsize=11)

    if created_fig:
        plt.tight_layout()

    return fig, ax, energies


# =============================================================================
#  4. Seconde derivee et courbure 2D (Zhang et al., 2011)
# =============================================================================

def secdev_curvature(data_cut, kpar, ev_arr,
                     sigma_k=2.0, sigma_e=2.0,
                     c0_fraction=0.05, border_clip=3):
    """
    Calcule la seconde derivee (-d2I/dE2) et la courbure 2D.

    La courbure 2D (Zhang et al., Rev. Sci. Instrum. 2011) est definie :
      curv = -[(C0 + (dI/dE)^2) * d2I/dk2 - dI/dk * dI/dE * d2I/dkdE]
              / (C0 + (dI/dk)^2 + (dI/dE)^2)^(3/2)

    Parametres
    ----------
    data_cut : np.ndarray (nk, ne)
    kpar, ev_arr : np.ndarray 1D
    sigma_k, sigma_e : float
        Lissage gaussien (pixels) avant derivation.
    c0_fraction : float
        Fraction du max des gradients pour le terme de regularisation.
    border_clip : int
        Nombre de pixels de bord a ignorer pour le calcul de vmax.

    Retourne
    --------
    dict avec cles: 'smoothed', 'secdev', 'curvature'
    """
    d = data_cut.astype(float)
    # Remplacer NaN par interpolation locale (médiane des voisins valides)
    # pour éviter les discontinuités NaN→0 qui font exploser C0
    nan_mask = np.isnan(d)
    if nan_mask.any():
        d = d.copy()
        d[nan_mask] = np.nanmedian(d)
    I = gaussian_filter(d, sigma=[sigma_k, sigma_e])

    dI_dE = np.gradient(I, ev_arr, axis=1)
    d2I_dE2 = np.gradient(dI_dE, ev_arr, axis=1)
    dI_dk = np.gradient(I, kpar, axis=0)
    d2I_dk2 = np.gradient(dI_dk, kpar, axis=0)
    d2I_dkdE = np.gradient(dI_dk, ev_arr, axis=1)

    secdev = -d2I_dE2

    # C0 calculé sur la région intérieure (sans les bords distordus par le filtre)
    bc = max(0, int(border_clip))
    interior = (slice(bc, -bc or None), slice(bc, -bc or None))
    C0 = c0_fraction * (np.abs(dI_dk[interior]).max()**2 + np.abs(dI_dE[interior]).max()**2)
    numer = (C0 + dI_dE**2) * d2I_dk2 - dI_dk * dI_dE * d2I_dkdE
    denom = (C0 + dI_dk**2 + dI_dE**2)**1.5
    curv2d = -numer / (denom + 1e-30)

    # Remettre NaN sur les pixels de bord (artefacts de filtre)
    if bc > 0:
        for arr in (I, secdev, curv2d):
            arr[:bc,  :] = np.nan;  arr[-bc:, :] = np.nan
            arr[:,  :bc] = np.nan;  arr[:, -bc:] = np.nan

    return dict(smoothed=I, secdev=secdev, curvature=curv2d)


# =============================================================================
#  5. Fit MDC Peak Pairs — méthode Igor band_finder_v2
#     Paires de pics symétriques autour d'un centre (typiquement Γ)
# =============================================================================

def _make_peak_pairs_model(n_pairs, width_mode='independent'):
    """
    Fabrique un modele MDC a N paires de Lorentziennes symétriques.

    Chaque paire i fitte deux pics à ±k0_i + xg (centre commun).

    width_mode :
        'independent' — chaque pic a sa propre largeur (2 largeurs/paire)
        'global'      — une seule largeur partagée par toutes les paires
        'symmetric'   — w1=w2 au sein de chaque paire, mais varie entre paires

    Vecteur de paramètres p :
        p[0] : bg_a  (fond linéaire pente)
        p[1] : bg_b  (fond linéaire offset)
        p[2] : xg    (centre commun de toutes les paires, souvent ~0)
        Puis pour chaque paire i :
          independent : k0_i, A1_i, w1_i, A2_i, w2_i   (5 params)
          symmetric   : k0_i, A1_i, A2_i, w_i           (4 params)
          global      : k0_i, A1_i, A2_i                 (3 params) + w_global à la fin
    """
    def model(k, *p):
        p  = np.asarray(p, dtype=float)
        bg = p[0] * k + p[1]
        xg = p[2]
        res = bg.copy()

        if width_mode == 'global':
            w_global = p[-1]
            for i in range(n_pairs):
                k0 = p[3 + 3*i]
                A1 = p[3 + 3*i + 1]
                A2 = p[3 + 3*i + 2]
                w  = w_global
                res += _lor_peak(k, -k0 + xg, A1, w)
                res += _lor_peak(k, +k0 + xg, A2, w)
        elif width_mode == 'symmetric':
            for i in range(n_pairs):
                k0 = p[3 + 4*i]
                A1 = p[3 + 4*i + 1]
                A2 = p[3 + 4*i + 2]
                w  = p[3 + 4*i + 3]
                res += _lor_peak(k, -k0 + xg, A1, w)
                res += _lor_peak(k, +k0 + xg, A2, w)
        else:  # independent
            for i in range(n_pairs):
                k0 = p[3 + 5*i]
                A1 = p[3 + 5*i + 1]
                w1 = p[3 + 5*i + 2]
                A2 = p[3 + 5*i + 3]
                w2 = p[3 + 5*i + 4]
                res += _lor_peak(k, -k0 + xg, A1, w1)
                res += _lor_peak(k, +k0 + xg, A2, w2)
        return res

    # Nombre de params par paire selon le mode
    n_pp = {'independent': 5, 'symmetric': 4, 'global': 3}[width_mode]
    n_extra = 1 if width_mode == 'global' else 0  # w_global à la fin
    return model, n_pp, n_extra


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
    scan_direction='down',
    width_mode='symmetric',
    k_min=None,
    k_max=None,
    k0_max=None,
    verbose=False,
):
    """
    Fit MDC par paires de Lorentziennes symétriques (méthode Igor Peak Pairs).

    Chaque paire produit deux pics symétriques autour d'un centre xg (≈ Γ).
    Plus stable que le fit indépendant quand les bandes sont symétriques.

    Paramètres
    ----------
    data_cut    : np.ndarray (nk, ne)
    kpar, ev_arr: np.ndarray 1D
    n_pairs     : int — nombre de paires (1 paire = 2 kF symétriques)
    ev_start/end: float — fenêtre en énergie
    smooth_fit  : float — lissage pour le fit
    smooth_detect: float — lissage pour la détection initiale
    gamma_init  : float — largeur initiale (eV)
    gamma_max   : float — largeur maximale (eV)
    kF_init     : list de float ou None — positions k0 initiales (une par paire).
                  IMPORTANT : doit correspondre à la position des pics à l'énergie
                  de départ du scan (ev_end si scan_direction='down', ev_start sinon).
                  Lire sur le waterfall à cette énergie. Si None, détection auto.
    center_init : float — position initiale de xg (défaut 0 = Γ)
    xg_range    : float — demi-fenêtre de contrainte autour de center_init.
                  xg est contraint dans [center_init-xg_range, center_init+xg_range].
                  Évite la dérive du centre. Défaut 0.12 pi/a.
    min_amplitude: float — seuil d'amplitude pour valider un pic (défaut 0.02).
                  Réduire si un côté est supprimé par les effets de matrice.
    max_jump    : float — seuil (pi/a) au-delà duquel un saut de k0 est rejeté.
                  Augmenter si la dispersion est rapide (bande très courbée).
    scan_direction: 'down' (défaut) | 'up'
                  'down' : scan de ev_end vers ev_start (de EF vers les basses E).
                  kF_init doit correspondre à la position à EF.
                  'up'   : scan de ev_start vers ev_end (ordre croissant d'énergie).
                  kF_init doit correspondre à la position à ev_start.
                  Pour une bande-trou centrée en Γ, utiliser 'down'.
    width_mode  : 'independent' | 'symmetric' | 'global'
    k_min, k_max: float ou None — plage k utilisée pour le fit (équivalent curseurs Igor).
                  CRUCIAL : restreindre au voisinage des pics d'intérêt pour éviter
                  que le fit accroche des features plus brillantes hors de la région.
                  Exemple : k_min=-0.6, k_max=0.6 pour un pocket α autour de Γ.
                  Si None, toute l'axe k est utilisée.

    Retourne
    --------
    dict avec clés :
        'kF_minus' : list de n_pairs arrays — position du pic gauche (-k0+xg)
        'kF_plus'  : list de n_pairs arrays — position du pic droit  (+k0+xg)
        'k0'       : list de n_pairs arrays — demi-séparation k0
        'xg'       : array — centre Γ à chaque énergie
        'e_fitted' : array des énergies convergées
        'I_smoothed': array lissé sur toute l'axe k (pour affichage)
        'kpar', 'ev_arr': axes
    """
    model, n_pp, n_extra = _make_peak_pairs_model(n_pairs, width_mode)

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

    lo = [-np.inf, -np.inf, center_init - xg_range]   # bg_a, bg_b, xg
    hi = [ np.inf,  np.inf, center_init + xg_range]
    for pi in range(n_pairs):
        k0_hi = k0_hi_list[pi]
        if width_mode == 'independent':
            lo += [k0_lo,  0.0,       0.001, 0.0,       0.001]
            hi += [k0_hi,  np.inf, gamma_max, np.inf, gamma_max]
        elif width_mode == 'symmetric':
            lo += [k0_lo,  0.0,  0.0,  0.001]
            hi += [k0_hi,  np.inf, np.inf, gamma_max]
        else:  # global
            lo += [k0_lo,  0.0,  0.0]
            hi += [k0_hi,  np.inf, np.inf]
    if width_mode == 'global':
        lo += [0.001]; hi += [gamma_max]

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

    print(f'fit_mdc_peak_pairs : {len(wf_energies)} tranches dans [{ev_lo:.3f}, {ev_hi:.3f}] eV '
          f'(scan {scan_direction!r} depuis ev_for_init={ev_for_init:.3f} eV)')
    if len(wf_energies) == 0:
        import warnings
        warnings.warn(
            'fit_mdc_peak_pairs : aucune énergie trouvée dans la fenêtre '
            f'[{ev_lo:.3f}, {ev_hi:.3f}] eV. '
            f'Vérifier ev_arr (plage [{float(ev_arr[0]):.3f}, {float(ev_arr[-1]):.3f}] eV), '
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
            print(f'  Auto-détection kF_init à E={ev_arr[ie_init]:+.3f} eV → {[f"{k:.3f}" for k in kF_init]}'
                  f'  ({"pics trouvés" if len(pk_idx) >= n_pairs else "fallback intensité max"})')

    kF_minus_list = [[] for _ in range(n_pairs)]
    kF_plus_list  = [[] for _ in range(n_pairs)]
    k0_list       = [[] for _ in range(n_pairs)]
    xg_list       = []
    e_fitted      = []
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
        return p

    for ev_i in wf_energies:
        ie = int(np.argmin(np.abs(ev_arr - ev_i)))
        # MDC restreinte à la plage [k_min, k_max] — comme les curseurs Igor
        mdc_full = I_fit[:, ie]
        mdc      = mdc_full[k_mask]
        mx = mdc.max()
        if mx <= 0:
            if verbose:
                print(f'  E={ev_arr[ie]:+.3f} eV : MDC max=0, tranche ignorée')
            continue
        mdc_n = mdc / mx

        if prev_popt is not None:
            p0 = list(prev_popt)
        else:
            p0 = _make_p0(prev_k0)

        try:
            popt, _ = curve_fit(model, kpar_fit, mdc_n, p0=p0,
                                bounds=(lo, hi), maxfev=8000)
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

                # Détection de saut : si k0 dévie trop du dernier k0 valide
                if abs(k0_fit - prev_k0[i]) > max_jump:
                    jumped = True

                km = (-k0_fit + xg_fit) if A1 > min_amplitude else np.nan
                kp = (+k0_fit + xg_fit) if A2 > min_amplitude else np.nan
                kF_minus_list[i].append(km if not jumped else np.nan)
                kF_plus_list[i].append(kp if not jumped else np.nan)
                k0_list[i].append(
                    k0_fit if (not jumped and (A1>min_amplitude or A2>min_amplitude))
                    else np.nan
                )
                if (np.isfinite(km) or np.isfinite(kp)) and not jumped:
                    converged = True

                if verbose:
                    status = 'OK' if (converged and not jumped) else ('JUMP' if jumped else 'LOW_A')
                    print(f'  E={ev_arr[ie]:+.3f} eV | k0={k0_fit:.3f} '
                          f'A-={A1:.3f} A+={A2:.3f} xg={xg_fit:.3f} '
                          f'ref_k0={prev_k0[i]:.3f} [{status}]')

            if converged:
                xg_list.append(xg_fit)
                e_fitted.append(ev_arr[ie])
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
                prev_popt = None   # reset warm-start si saut ou amplitude trop faible
                # prev_k0 conservé intentionnellement : on garde le dernier k0 valide
                # comme référence pour la détection de saut des prochaines tranches

        except Exception as exc:
            if verbose:
                print(f'  E={ev_arr[ie]:+.3f} eV : exception → {exc}')
            for i in range(n_pairs):
                kF_minus_list[i].append(np.nan)
                kF_plus_list[i].append(np.nan)
                k0_list[i].append(np.nan)
            xg_list.append(np.nan)
            e_fitted.append(ev_arr[ie])
            prev_popt = None

    # Réordonner en énergie croissante (indépendant du sens du scan)
    e_arr_out = np.array(e_fitted)
    sort_idx  = np.argsort(e_arr_out)
    return dict(
        kF_minus   =[np.array(x)[sort_idx] for x in kF_minus_list],
        kF_plus    =[np.array(x)[sort_idx] for x in kF_plus_list],
        k0         =[np.array(x)[sort_idx] for x in k0_list],
        xg         =np.array(xg_list)[sort_idx],
        e_fitted   =e_arr_out[sort_idx],
        I_smoothed =I_fit,
        kpar       =kpar,
        ev_arr     =ev_arr,
        n_pairs    =n_pairs,
        width_mode =width_mode,
    )


# =============================================================================
#  5b. Diagnostics MDC — visualisation des fits slice par slice
# =============================================================================

def debug_mdc_fit(
    data_cut, kpar, ev_arr,
    energy,
    n_pairs=1,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.20,
    kF_init=None,
    center_init=0.0,
    xg_range=0.12,
    k_min=None,
    k_max=None,
    k0_max=None,
    width_mode='symmetric',
    ax=None,
    title=None,
    verbose=True,
    show_residual=True,
):
    """
    Fit et affiche UNE MDC a l'energie donnee, avec le modele ajuste superpose.

    Utilise la meme logique que fit_mdc_peak_pairs() mais sur un seul slice.
    Permet de verifier visuellement si les parametres (k_min, k_max, kF_init,
    gamma_init…) sont corrects avant de lancer le scan complet.

    Parametres
    ----------
    data_cut   : np.ndarray (nk, ne)
    kpar       : np.ndarray 1D  — axe k (pi/a)
    ev_arr     : np.ndarray 1D  — axe energie (eV, 0 = EF)
    energy     : float          — energie du slice a diagnostiquer (ex: -0.05)
    n_pairs    : int            — nombre de paires (doit correspondre a fit_mdc_peak_pairs)
    k_min/max  : float|None     — meme contrainte que fit_mdc_peak_pairs
    kF_init    : list|None      — guess initial pour k0 de chaque paire
    width_mode : str            — 'symmetric' | 'independent' | 'global'
    ax         : matplotlib.Axes ou None — si None, cree une figure

    Retourne
    --------
    dict:
        'success'  : bool
        'popt'     : array des parametres optimises (ou None)
        'k0'       : list de float — demi-separations ajustees
        'xg'       : float — centre ajuste
        'gamma'    : list de float — largeurs ajustees
        'residual' : float — norme des residus normalises
        'ax'       : matplotlib Axes
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    # --- preparation des donnees ---
    I_fit    = gaussian_filter1d(data_cut.astype(float), sigma=smooth_fit,    axis=0)
    I_detect = gaussian_filter1d(data_cut.astype(float), sigma=smooth_detect, axis=0)

    ie = int(np.argmin(np.abs(ev_arr - energy)))
    energy_actual = float(ev_arr[ie])

    k_lo = k_min if k_min is not None else float(kpar[0])
    k_hi = k_max if k_max is not None else float(kpar[-1])
    k_mask   = (kpar >= k_lo) & (kpar <= k_hi)
    kpar_fit = kpar[k_mask]

    mdc_full   = I_fit[:, ie]
    mdc        = mdc_full[k_mask]
    mdc_detect = I_detect[k_mask, ie]
    mx = mdc.max()
    if mx <= 0:
        return {'success': False, 'popt': None, 'ax': ax}
    mdc_n = mdc / mx

    # --- guess initial ---
    dk = abs(float(kpar[1] - kpar[0]))
    k0_lo = max(dk * 2, 0.02)
    k0_hi_auto = (k_hi - center_init) * 0.95

    if k0_max is None:
        k0_hi_list = [k0_hi_auto] * n_pairs
    elif np.isscalar(k0_max):
        k0_hi_list = [float(k0_max)] * n_pairs
    else:
        k0_hi_list = list(k0_max)
        while len(k0_hi_list) < n_pairs:
            k0_hi_list.append(k0_hi_auto)

    if kF_init is None:
        mdc_d_n = mdc_detect / (mdc_detect.max() or 1)
        pk_idx, _ = find_peaks(mdc_d_n, prominence=0.05,
                               distance=max(1, int(0.05 / dk)))
        pos_k = kpar_fit[pk_idx]
        pos_k_pos = pos_k[pos_k > center_init]
        if len(pos_k_pos) >= n_pairs:
            kF_use = sorted(pos_k_pos)[:n_pairs]
        else:
            kF_use = list(np.linspace(0.05, k0_hi_auto * 0.8, n_pairs))
    else:
        kF_use = list(kF_init)

    # --- construction du modele et des bornes ---
    model, n_pp, _ = _make_peak_pairs_model(n_pairs, width_mode)

    lo = [-np.inf, -np.inf, center_init - xg_range]
    hi = [ np.inf,  np.inf, center_init + xg_range]
    for pi in range(n_pairs):
        k0_hi = k0_hi_list[pi]
        if width_mode == 'independent':
            lo += [k0_lo,  0.0,       0.001, 0.0,       0.001]
            hi += [k0_hi,  np.inf, gamma_max, np.inf, gamma_max]
        elif width_mode == 'symmetric':
            lo += [k0_lo, 0.0, 0.0, 0.001]
            hi += [k0_hi, np.inf, np.inf, gamma_max]
        else:
            lo += [k0_lo, 0.0, 0.0]
            hi += [k0_hi, np.inf, np.inf]
    if width_mode == 'global':
        lo += [0.001]; hi += [gamma_max]

    p0 = [0.0, float(np.percentile(mdc_n, 10)), center_init]
    for k0g in kF_use:
        A_g = max(float(np.interp(abs(k0g) + center_init, kpar_fit, mdc_n)), 0.05)
        if width_mode == 'independent':
            p0 += [abs(k0g), A_g, gamma_init, A_g, gamma_init]
        elif width_mode == 'symmetric':
            p0 += [abs(k0g), A_g, A_g, gamma_init]
        else:
            p0 += [abs(k0g), A_g, A_g]
    if width_mode == 'global':
        p0 += [gamma_init]

    # --- fit ---
    popt = None
    success = False
    try:
        popt, pcov = curve_fit(model, kpar_fit, mdc_n, p0=p0,
                               bounds=(lo, hi), maxfev=12000)
        success = True
    except Exception as exc:
        if verbose:
            print(f"[debug_mdc_fit] ECHEC du fit a E={energy_actual:.4f} eV : {exc}")

    # --- figure ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # donnees brutes (toute la plage k) en gris leger
    ax.plot(kpar, mdc_full / mx, color='#cccccc', lw=1, label='hors plage')
    # donnees dans la plage de fit
    ax.plot(kpar_fit, mdc_n, 'k-', lw=1.5, label='MDC')
    # guess initial
    mdc_p0 = model(kpar_fit, *p0)
    ax.plot(kpar_fit, mdc_p0, 'b--', lw=1, alpha=0.7, label='p0')
    # positions du guess
    for k0g in kF_use:
        ax.axvline(center_init - k0g, color='blue', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(center_init + k0g, color='blue', lw=0.8, ls=':', alpha=0.5)

    k0_out = []
    xg_out = center_init
    gamma_out = []

    if success:
        mdc_fit = model(kpar_fit, *popt)
        ax.plot(kpar_fit, mdc_fit, 'r-', lw=2, label='fit')
        xg_out = popt[2]
        ax.axvline(xg_out, color='red', lw=0.8, ls='--', alpha=0.7)

        for i in range(n_pairs):
            k0_i = popt[3 + n_pp*i]
            k0_out.append(k0_i)
            km = xg_out - k0_i
            kp = xg_out + k0_i
            ax.axvline(km, color='green',  lw=1.5, ls='-')
            ax.axvline(kp, color='orange', lw=1.5, ls='-')
            if width_mode == 'independent':
                gamma_out.append((popt[3 + n_pp*i + 2], popt[3 + n_pp*i + 4]))
            elif width_mode == 'symmetric':
                gamma_out.append(popt[3 + n_pp*i + 3])
            else:
                gamma_out.append(popt[-1] if width_mode == 'global' else gamma_init)

        residual = float(np.sqrt(np.mean((mdc_n - mdc_fit)**2)))
        # Résidu en remplissage (pas de twinx — compatible panel)
        if show_residual:
            ax.fill_between(kpar_fit, mdc_n, mdc_fit,
                            alpha=0.2, color='red')
    else:
        residual = np.nan

    # limites de la plage de fit
    ax.axvline(k_lo, color='purple', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(k_hi, color='purple', lw=0.8, ls='--', alpha=0.5)

    # Titre court
    if title is not None:
        ttl = title
    else:
        status = 'OK' if success else 'FAIL'
        ttl = f'E = {energy_actual:+.3f} eV  [{status}]'
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel('k (π/a)', fontsize=8)
    ax.set_ylabel('I norm.', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(kpar[0]), float(kpar[-1]))

    # Annotation compacte (remplace la légende dans le panel)
    if success:
        info = (f"xg={xg_out:.3f}\n"
                f"k0={[f'{v:.3f}' for v in k0_out]}\n"
                f"rms={residual:.3f}")
        ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=6,
                va='top', bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='lightyellow', alpha=0.8))

    # Légende seulement en mode standalone (pas dans le panel)
    if show_residual:   # proxy : show_residual=True = mode standalone
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    if verbose and success:
        print(f"  E={energy_actual:+.4f} eV | xg={xg_out:.4f} | "
              f"k0={[round(v,4) for v in k0_out]} | rms={residual:.4f}")

    return dict(success=success, popt=popt, k0=k0_out, xg=xg_out,
                gamma=gamma_out, residual=residual, ax=ax,
                energy=energy_actual)


def plot_mdc_fit_panel(
    data_cut, kpar, ev_arr,
    energies,
    n_pairs=1,
    smooth_fit=1.5,
    smooth_detect=3.0,
    gamma_init=0.05,
    gamma_max=0.20,
    kF_init=None,
    center_init=0.0,
    xg_range=0.12,
    k_min=None,
    k_max=None,
    k0_max=None,
    width_mode='symmetric',
    ncols=3,
    figsize=None,
    suptitle='Diagnostic MDC fits',
):
    """
    Affiche une grille de diagnostics MDC pour une liste d'energies.

    Chaque panneau montre la MDC + guess initial + fit converge (ou echec)
    pour l'energie correpondante. Ideal pour identifier a quelle energie
    le fit commence a derailler et pourquoi.

    Parametres
    ----------
    energies : list de float — energies a diagnostiquer (eV relatif a EF)
               Exemple: np.linspace(-0.15, -0.01, 12)
    ncols    : int — nombre de colonnes dans la grille
    figsize  : tuple ou None — taille de la figure

    Retourne
    --------
    fig, axes, results : liste des dicts retournes par debug_mdc_fit()
    """
    n = len(energies)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    results = []
    for idx, (en, ax) in enumerate(zip(energies, axes_flat)):
        r = debug_mdc_fit(
            data_cut, kpar, ev_arr,
            energy=en,
            n_pairs=n_pairs,
            smooth_fit=smooth_fit,
            smooth_detect=smooth_detect,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            kF_init=kF_init,
            center_init=center_init,
            xg_range=xg_range,
            k_min=k_min,
            k_max=k_max,
            k0_max=k0_max,
            width_mode=width_mode,
            ax=ax,
        )
        results.append(r)
        # indicateur succes/echec dans le coin
        status = 'OK' if r['success'] else 'FAIL'
        color  = 'limegreen' if r['success'] else 'red'
        ax.text(0.02, 0.96, status, transform=ax.transAxes,
                fontsize=9, fontweight='bold', color=color, va='top')

    # cacher les axes inutilises
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, axes, results


def plot_mdc_waterfall_with_fit(
    fit_result,
    data_cut=None,
    vmin=None, vmax=None,
    cmap='Blues',
    figsize=(12, 8),
    title='MDC Peak Pairs — resultat du fit',
):
    """
    Affiche la carte 2D E(k) avec les kF ajustes superposes,
    PLUS un waterfall des MDCs avec les positions kF marquees.

    Parametres
    ----------
    fit_result : dict retourne par fit_mdc_peak_pairs()
    data_cut   : np.ndarray (nk, ne) — donnees brutes (si None, utilise I_smoothed)
    """
    kpar   = fit_result['kpar']
    ev_arr = fit_result['ev_arr']
    e_fit  = fit_result['e_fitted']
    I_sm   = fit_result['I_smoothed']

    img = data_cut if data_cut is not None else I_sm

    fig, (ax_map, ax_wf) = plt.subplots(1, 2, figsize=figsize)

    # --- Carte 2D ---
    if vmin is None: vmin = np.nanpercentile(img, 2)
    if vmax is None: vmax = np.nanpercentile(img, 98)
    ax_map.pcolormesh(kpar, ev_arr, img.T, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading='auto')
    colors_minus = ['cyan', 'lime', 'yellow']
    colors_plus  = ['red', 'orange', 'magenta']
    for i in range(fit_result['n_pairs']):
        kF_m = fit_result['kF_minus'][i]
        kF_p = fit_result['kF_plus'][i]
        ax_map.scatter(kF_m, e_fit, s=6, color=colors_minus[i % 3],
                       label=f'kF- pair{i+1}', zorder=5)
        ax_map.scatter(kF_p, e_fit, s=6, color=colors_plus[i % 3],
                       label=f'kF+ pair{i+1}', zorder=5)
    ax_map.set_xlabel('k (π/a)'); ax_map.set_ylabel('E - EF (eV)')
    ax_map.legend(fontsize=8); ax_map.set_title('Carte 2D + kF fits')

    # --- Waterfall ---
    # Selectionne un sous-ensemble pour ne pas surcharger
    n_wf = min(30, len(e_fit))
    idx_sel = np.round(np.linspace(0, len(e_fit) - 1, n_wf)).astype(int)
    spacing = 0.4 / n_wf  # ecart vertical entre MDCs

    cmap_wf = plt.cm.RdYlBu_r
    for rank, ii in enumerate(idx_sel):
        ev_i  = e_fit[ii]
        ie    = int(np.argmin(np.abs(ev_arr - ev_i)))
        mdc   = I_sm[:, ie]
        mx    = mdc.max()
        if mx <= 0: continue
        mdc_n = mdc / mx
        offset = rank * spacing
        color  = cmap_wf(rank / n_wf)
        ax_wf.plot(kpar, mdc_n + offset, lw=0.8, color=color)
        # points kF sur le waterfall
        for i in range(fit_result['n_pairs']):
            km = fit_result['kF_minus'][i][ii]
            kp = fit_result['kF_plus'][i][ii]
            mdc_at_km = float(np.interp(km, kpar, mdc_n)) if np.isfinite(km) else np.nan
            mdc_at_kp = float(np.interp(kp, kpar, mdc_n)) if np.isfinite(kp) else np.nan
            if np.isfinite(km):
                ax_wf.plot(km, mdc_at_km + offset, 'o', ms=3,
                           color=colors_minus[i % 3], zorder=5)
            if np.isfinite(kp):
                ax_wf.plot(kp, mdc_at_kp + offset, 'o', ms=3,
                           color=colors_plus[i % 3], zorder=5)

    ax_wf.set_xlabel('k (π/a)'); ax_wf.set_ylabel('MDC offset')
    ax_wf.set_title('Waterfall MDCs + positions kF')
    ax_wf.set_xlim(float(kpar[0]), float(kpar[-1]))

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, (ax_map, ax_wf)


# =============================================================================
#  6. Fit MDC multi-Lorentziennes + Hungarian matching (méthode générale)
# =============================================================================

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

def remove_grid_artifact(data, kpar=None, method='fft',
                          grid_freq=None, notch_width=3, notch_sigma=1.0,
                          ev_ref_range=None, plot=False):
    """
    Supprime l'artefact périodique de la grille de retard du détecteur DA30.

    L'artefact apparaît comme des stries périodiques le long de l'axe k
    (même fréquence à toutes les énergies). Deux méthodes disponibles :

    method='fft'     : filtre coupe-bande dans l'espace de Fourier (1D le long de k).
                       Efficace si la période de la grille est bien définie.
    method='profile' : divise chaque tranche en énergie par le profil angulaire
                       moyen (lissé). Plus simple, supprime aussi le bruit fixe.

    Paramètres
    ----------
    data         : np.ndarray (nk, ne)
    kpar         : np.ndarray 1D — axe k (pour affichage uniquement)
    method       : 'fft' | 'profile'
    grid_freq    : float ou None — fréquence de la grille en cycles/pixel.
                   Si None, auto-détection sur le spectre moyen.
    notch_width  : int — demi-largeur du filtre notch en bins FFT (method='fft')
    notch_sigma  : float — lissage gaussien du filtre notch (0 = porte, >0 = doux)
    ev_ref_range : tuple (ev_lo, ev_hi) ou None — fenêtre en énergie pour estimer
                   le profil de référence. Si None, toutes les énergies sont utilisées.
    plot         : bool — afficher le profil Fourier et le filtre appliqué

    Retourne
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
                            label=f'grille f={grid_freq:.4f} (T={grid_period_px:.1f} px)')
            axes[0].set(xlabel='Fréquence (cycles/px)', ylabel='Puissance',
                        title='Spectre Fourier du profil moyen')
            axes[0].legend()
            axes[1].plot(freqs, filt, 'b-', lw=1.5)
            axes[1].set(xlabel='Fréquence', ylabel='Filtre', title='Filtre notch appliqué',
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
                    label='profil brut')
            ax.plot(kpar, ref_smooth / ref_smooth.max(), 'r-', lw=1.5,
                    label='profil lissé (diviseur)')
            ax.set(xlabel='k// (pi/a)', ylabel='I norm.',
                   title='Profil angulaire moyen utilisé pour la correction')
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

    Les donnees mesurent N(k,E) = A(k,E) × f(E,T).
    La division retire la coupure thermique et donne acces a A(k,E)
    jusque cutoff_ev au-dessus de EF.

    Attention : amplifie le bruit au-dessus de EF — utiliser sur
    des donnees suffisamment lisses ou a basse temperature.

    Parametres
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
        raise ValueError("Aucune coupe valide (verifier hv, work_function, V0).")

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
