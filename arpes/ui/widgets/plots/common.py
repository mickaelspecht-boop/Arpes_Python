"""Shared ARPES analysis helpers used by plot widgets/controllers."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit

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


def auto_ef_window(ev_arr, edc, half_width=0.15, search=(-0.5, 0.2)):
    """Centre une fenêtre de fit sur le gradient max de l'EDC.

    Retourne (e_lo, e_hi). Utile quand EF n'est pas déjà proche de zéro.
    """
    ev_arr = np.asarray(ev_arr, dtype=float)
    edc    = np.asarray(edc,    dtype=float)
    mask = (ev_arr >= search[0]) & (ev_arr <= search[1])
    if mask.sum() < 10:
        return (-half_width, half_width)
    e = ev_arr[mask]
    I = np.where(np.isfinite(edc[mask]), edc[mask], 0.0)
    I_sm = gaussian_filter1d(I, sigma=2.0)
    grad = np.gradient(I_sm, e)
    ec = float(e[int(np.argmax(np.abs(grad)))])
    return (ec - half_width, ec + half_width)


def fit_fermi_edge_per_column(
    data, kpar, ev_arr,
    temperature_K=28.0,
    half_width=0.15,
    sigma_resolution_init=0.025,
    poly_deg=2,
    auto_window=True,
    ef_search=(-0.5, 0.2),
    min_amplitude_ratio=0.15,
    max_ef_err_ev=0.020,
    verbose=False,
):
    """Fit du bord de Fermi colonne par colonne (k) puis lissage polynomial.

    `data` : (n_k, n_E). `kpar` : (n_k,). `ev_arr` : (n_E,).

    Retourne dict :
        ef_per_col   : EF brut fitté par colonne (np.nan si échec)
        ef_smooth    : EF lissé via polynôme degré `poly_deg`
        poly_coefs   : coefficients du polynôme (np.polyfit, degré décroissant)
        kpar         : axe k associé
        mean_ef      : moyenne pondérée des EF retenus
        mean_fwhm    : FWHM moyen retenu
        rms          : écart-type des résidus colonne par colonne
        n_valid      : nombre de colonnes avec fit valide
        window       : (e_lo, e_hi) effectivement utilisée
    """
    data  = np.asarray(data, dtype=float)
    kpar  = np.asarray(kpar, dtype=float)
    ev    = np.asarray(ev_arr, dtype=float)
    n_k   = data.shape[0]

    edc_avg = np.nanmean(data, axis=0)
    if auto_window:
        win = auto_ef_window(ev, edc_avg, half_width=half_width, search=ef_search)
    else:
        win = (-half_width, half_width)

    ef_per_col   = np.full(n_k, np.nan)
    fwhm_per_col = np.full(n_k, np.nan)
    err_per_col  = np.full(n_k, np.nan)

    amp_max = float(np.nanmax(edc_avg)) if np.any(np.isfinite(edc_avg)) else 0.0
    # Axe matplotlib jetable (hors pyplot) pour éviter d'accumuler des figures
    from matplotlib.figure import Figure as _Fig
    _dummy_ax = _Fig().add_subplot(111)
    for i in range(n_k):
        col = data[i]
        if amp_max > 0 and np.nanmax(col) < min_amplitude_ratio * amp_max:
            continue
        try:
            _dummy_ax.clear()
            r = fit_fermi_edge(
                ev, col,
                temperature_K=temperature_K,
                fit_range=win,
                sigma_resolution_init=sigma_resolution_init,
                fix_kBT=True,
                units="binding",
                ax=_dummy_ax,
                verbose=False,
            )
            if not r.get("success"):
                continue
            err = float(r.get("EF_err", np.nan))
            if not np.isfinite(err) or err > max_ef_err_ev:
                continue
            ef_per_col[i]   = float(r["EF"])
            fwhm_per_col[i] = float(r["fwhm_res"])
            err_per_col[i]  = err
        except Exception:
            continue

    valid = np.isfinite(ef_per_col)
    n_valid = int(valid.sum())
    if n_valid < max(3, poly_deg + 2):
        # Pas assez de colonnes valides — repli sur EF scalaire
        ef_smooth = np.full(n_k, np.nanmedian(ef_per_col)) if n_valid else np.zeros(n_k)
        coefs = np.array([np.nanmedian(ef_per_col) if n_valid else 0.0])
        rms = float(np.nanstd(ef_per_col)) if n_valid else 0.0
    else:
        # Pondération par 1/err²
        w = np.zeros(n_k)
        w[valid] = 1.0 / np.clip(err_per_col[valid], 1e-4, np.inf) ** 2
        coefs = np.polyfit(kpar[valid], ef_per_col[valid], deg=poly_deg, w=w[valid])
        ef_smooth = np.polyval(coefs, kpar)
        rms = float(np.sqrt(np.mean((ef_per_col[valid] - ef_smooth[valid]) ** 2)))

    mean_ef = float(np.nansum(ef_per_col[valid]) / max(n_valid, 1)) if n_valid else 0.0
    mean_fwhm = float(np.nanmedian(fwhm_per_col[valid])) if n_valid else float("nan")

    if verbose:
        print(f"fit_fermi_edge_per_column : {n_valid}/{n_k} valides | "
              f"<EF>={mean_ef*1000:+.1f} meV | FWHM≈{mean_fwhm*1000:.0f} meV | "
              f"rms_poly={rms*1000:.1f} meV")

    return dict(
        ef_per_col=ef_per_col,
        ef_smooth=ef_smooth,
        poly_coefs=np.asarray(coefs, dtype=float),
        kpar=kpar.copy(),
        mean_ef=mean_ef,
        mean_fwhm=mean_fwhm,
        rms=rms,
        n_valid=n_valid,
        window=win,
        fwhm_per_col=fwhm_per_col,
        err_per_col=err_per_col,
    )


def apply_ef_correction_per_column(data, kpar, ev_arr, ef_smooth):
    """Décale chaque colonne k pour que `ef_smooth(k)` tombe à E=0.

    Implémentation par interpolation linéaire 1D colonne par colonne sur la
    grille `ev_arr` originale. Renvoie un nouveau tableau (n_k, n_E).
    Les bords sortant de la grille sont remplis avec NaN.
    """
    data  = np.asarray(data, dtype=float)
    kpar  = np.asarray(kpar, dtype=float)
    ev    = np.asarray(ev_arr, dtype=float)
    ef_smooth = np.asarray(ef_smooth, dtype=float)
    n_k, n_e = data.shape
    out = np.empty_like(data)
    for i in range(n_k):
        # data_corr(E) = data_orig(E + ef_smooth[i])
        src = ev + float(ef_smooth[i])
        col = data[i]
        # Bord remplit avec la valeur extrême pour éviter les NaN qui cassent
        # gaussian_filter (utilisé par SecDev/Curvature côté affichage).
        out[i] = np.interp(ev, src, col, left=col[0], right=col[-1])
    return out


# =============================================================================
#  1. Detection de Gamma par milieu des kF (MDC midpoint)
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
