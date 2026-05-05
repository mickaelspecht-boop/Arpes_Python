"""Fermi-surface plotting and cut helpers."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator

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

    print("Gamma par milieu des kF :")
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
