"""MDC diagnostic plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import _make_peak_pairs_model

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
