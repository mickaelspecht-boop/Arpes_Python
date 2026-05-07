"""Peak-pair MDC fitting routines."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .fit_overlay import _make_peak_pairs_model, _resolution_correct_gamma

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
    dE_eV=0.0,
    dk_inv_a=0.0,
    resolution_source="",
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
    gamma_floor = min(max(0.001, float(dk_inv_a or 0.0)), float(gamma_max) * 0.95)
    for pi in range(n_pairs):
        k0_hi = k0_hi_list[pi]
        if width_mode == 'independent':
            lo += [k0_lo,  0.0, gamma_floor, 0.0, gamma_floor]
            hi += [k0_hi,  np.inf, gamma_max, np.inf, gamma_max]
        elif width_mode == 'symmetric':
            lo += [k0_lo,  0.0,  0.0, gamma_floor]
            hi += [k0_hi,  np.inf, np.inf, gamma_max]
        else:  # global
            lo += [k0_lo,  0.0,  0.0]
            hi += [k0_hi,  np.inf, np.inf]
    if width_mode == 'global':
        lo += [gamma_floor]; hi += [gamma_max]

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
    gamma_list    = [[] for _ in range(n_pairs)]
    sigma_kF_minus_list = [[] for _ in range(n_pairs)]
    sigma_kF_plus_list  = [[] for _ in range(n_pairs)]
    sigma_gamma_list    = [[] for _ in range(n_pairs)]
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
            popt, pcov = curve_fit(model, kpar_fit, mdc_n, p0=p0,
                                bounds=(lo, hi), maxfev=8000)
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

                # Détection de saut : si k0 dévie trop du dernier k0 valide
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
                    g1 = popt[3 + n_pp*i + 2]
                    g2 = popt[3 + n_pp*i + 4]
                    gamma_fit = float(np.nanmean([g1, g2]))
                    sg1 = float(sigma_full[3 + n_pp * i + 2])
                    sg2 = float(sigma_full[3 + n_pp * i + 4])
                    sigma_gamma_val = float(0.5 * np.sqrt(sg1 ** 2 + sg2 ** 2))
                elif width_mode == 'symmetric':
                    gamma_fit = float(popt[3 + n_pp*i + 3])
                    sigma_gamma_val = float(sigma_full[3 + n_pp * i + 3])
                else:
                    gamma_fit = float(popt[-1])
                    sigma_gamma_val = float(sigma_full[-1])
                gamma_list[i].append(gamma_fit if not jumped else np.nan)
                sigma_gamma_list[i].append(
                    sigma_gamma_val if not jumped else np.nan
                )

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
                gamma_list[i].append(np.nan)
                sigma_kF_minus_list[i].append(np.nan)
                sigma_kF_plus_list[i].append(np.nan)
                sigma_gamma_list[i].append(np.nan)
            xg_list.append(np.nan)
            e_fitted.append(ev_arr[ie])
            prev_popt = None

    # Réordonner en énergie croissante (indépendant du sens du scan)
    e_arr_out = np.array(e_fitted)
    sort_idx  = np.argsort(e_arr_out)
    k0_out = [np.array(x)[sort_idx] for x in k0_list]
    gamma_brut = [np.array(x)[sort_idx] for x in gamma_list]
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
        kpar       =kpar,
        ev_arr     =ev_arr,
        n_pairs    =n_pairs,
        width_mode =width_mode,
        gamma       =gamma_brut,
        gamma_brut  =gamma_brut,
        gamma_min   =gamma_min,
        gamma_corrige=gamma_corrige,
        resolution  ={
            "dE_eV": float(dE_eV or 0.0),
            "dE_meV": float(dE_eV or 0.0) * 1000.0,
            "dk_inv_a": float(dk_inv_a or 0.0),
            "source": str(resolution_source or ""),
        },
    )


# =============================================================================
#  5b. Diagnostics MDC — visualisation des fits slice par slice
# =============================================================================
