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
    Fit the Fermi-Dirac distribution convolved with a Gaussian
    (instrumental resolution) on an averaged EDC.

    Model:
        I(E) = A x [FD(E; EF, kBT) convolved with Gauss(sigma_res)] + slope*E + bg

    Can be used on:
    - An EDC from a gold file (absolute reference), best precision
    - An EDC averaged over all sample angles, auto-calibration

    Parameters
    ----------
    ev_arr : np.ndarray 1D
        Energy axis.
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
        print(f"{'OK' if success else 'FAILED'} Fit EF   "
              f"EF = {EF_fit:+.4f} eV  ±{EF_err*1000:.1f} meV  "
              f"| FWHM_res = {fwhm_fit*1000:.0f} meV  "
              f"| T_eff = {T_eff:.0f} K  (T_nominal={temperature_K:.0f} K)  "
              f"| residual = {residual:.4f}")
        if units == 'binding' and abs(EF_fit) > 0.05:
            print(f"  Warning: EF offset = {EF_fit*1000:+.0f} meV - the energy axis is shifted by this value.")
            print(f"    -> Apply ev_arr = ev_arr - ({EF_fit:.4f}) to correct it.")

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
    ax2.set_ylabel('Residual', color='gray', fontsize=8)
    ax2.tick_params(axis='y', colors='gray', labelsize=7)
    ax2.set_ylim(-0.4, 0.4)

    ttl = title or ('EDC EF calibration' + (' [FAILED]' if not success else ''))
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('E (eV)')
    ax.set_ylabel(r'$I/I_{\max}$')
    ax.legend(fontsize=8, loc='center right')
    ax.set_xlim(fit_range)

    # annotations texte
    ax.text(0.02, 0.05,
            f"EF = {EF_fit:+.4f} eV  ±{EF_err*1000:.1f} meV\n"
            f"FWHM = {fwhm_fit*1000:.0f} meV   T_eff = {T_eff:.0f} K\n"
            f"rms residual = {residual:.4f}",
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


def _robust_ef_polyfit(kpar, ef_per_col, err_per_col, poly_deg):
    """Lissage EF(k) robuste, avec poids compatibles `np.polyfit`.

    `np.polyfit(..., w=...)` attend des poids de type 1/sigma. Un poids
    1/sigma^2 amplifie trop les colonnes à incertitude sous-estimée et peut
    produire une parabole sans rapport avec le nuage EF(k).
    """
    kpar = np.asarray(kpar, dtype=float)
    ef_per_col = np.asarray(ef_per_col, dtype=float)
    err_per_col = np.asarray(err_per_col, dtype=float)
    valid = np.isfinite(kpar) & np.isfinite(ef_per_col)
    n_valid = int(valid.sum())
    if n_valid < max(3, poly_deg + 2):
        value = float(np.nanmedian(ef_per_col[valid])) if n_valid else 0.0
        coefs = np.array([value])
        return coefs, np.full_like(kpar, value, dtype=float)

    fit_mask = valid.copy()
    coefs = None
    for _ in range(4):
        err = np.clip(err_per_col[fit_mask], 0.005, 0.050)
        err = np.where(np.isfinite(err), err, 0.050)
        w_fit = 1.0 / err
        deg_fit = min(poly_deg, int(fit_mask.sum()) - 1)
        coefs = np.polyfit(kpar[fit_mask], ef_per_col[fit_mask], deg=deg_fit, w=w_fit)
        resid = ef_per_col[fit_mask] - np.polyval(coefs, kpar[fit_mask])
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = max(1.4826 * mad, 0.003)
        keep_local = np.abs(resid) <= max(4.0 * sigma, 0.015)
        if keep_local.all() or keep_local.sum() < max(3, poly_deg + 2):
            break
        idx = np.where(fit_mask)[0]
        fit_mask[idx[~keep_local]] = False

    if coefs is None:
        coefs = np.array([float(np.nanmedian(ef_per_col[valid]))])
    return np.asarray(coefs, dtype=float), np.polyval(coefs, kpar)


def fit_fermi_edge_per_column(
    data, kpar, ev_arr,
    temperature_K=28.0,
    half_width=0.15,
    sigma_resolution_init=0.025,
    poly_deg=2,
    auto_window=True,
    ef_search=(-0.5, 0.2),
    fit_range=None,
    min_amplitude_ratio=0.15,
    max_ef_err_ev=0.020,
    verbose=False,
):
    """Fit the Fermi edge column by column (k), then polynomial-smooth it.

    `data` : (n_k, n_E). `kpar` : (n_k,). `ev_arr` : (n_E,).

    Returns dict:
        ef_per_col   : raw EF fitted by column (np.nan on failure)
        ef_smooth    : EF smoothed with a polynomial of degree `poly_deg`
        poly_coefs   : polynomial coefficients (np.polyfit, decreasing degree)
        kpar         : associated k axis
        mean_ef      : weighted average of retained EF values
        mean_fwhm    : average retained FWHM
        rms          : standard deviation of column-by-column residuals
        n_valid      : number of columns with a valid fit
        window       : (e_lo, e_hi) actually used
    """
    data  = np.asarray(data, dtype=float)
    kpar  = np.asarray(kpar, dtype=float)
    ev    = np.asarray(ev_arr, dtype=float)
    n_k   = data.shape[0]

    edc_avg = np.nanmean(data, axis=0)
    if fit_range is not None:
        win = tuple(sorted((float(fit_range[0]), float(fit_range[1]))))
    elif auto_window:
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
        coefs, ef_smooth = _robust_ef_polyfit(kpar, ef_per_col, err_per_col, poly_deg)
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
#  5. Fit MDC Peak Pairs — méthode Igor band_finder_v2
#     Paires de pics symétriques autour d'un centre (typiquement Γ)
# =============================================================================
