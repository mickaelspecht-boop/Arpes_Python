"""Band-map specific analysis helpers."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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
