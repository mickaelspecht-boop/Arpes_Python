"""Fermi-surface autocorrelation C(q) for ARPES nesting analysis.

C(q) = Σ_k A(k, E_F) A(k+q, E_F) is the geometric self-overlap of the Fermi
surface translated by q. It is NOT the full susceptibility χ(q) (it ignores
interactions and orbital matrix elements) but it is a robust, model-free measure
of FS connectivity: peaks flag candidate nesting / folding vectors.

Computed via the Wiener–Khinchin theorem (autocorrelation = IFFT|FFT|²) so it is
O(N log N). Pure numpy — no PyQt, headless-testable.
"""
from __future__ import annotations

import numpy as np


def fs_autocorrelation(
    fs: np.ndarray, *, subtract_mean: bool = False, normalize: bool = True
) -> np.ndarray:
    """Centered FS autocorrelation, same shape as ``fs``.

    ``fs[iy, ix]`` is the Fermi-surface intensity map (NaNs treated as 0).
    ``subtract_mean`` removes the q=0 background so weak nesting peaks stand out
    (the "connected" autocorrelation). ``normalize`` scales the peak to 1.
    The output is fftshift-centered: the middle pixel is q=0.
    """
    a = np.nan_to_num(np.asarray(fs, dtype=float), nan=0.0)
    if subtract_mean:
        a = a - a.mean()
    spec = np.fft.rfft2(a)
    ac = np.fft.irfft2(spec * np.conj(spec), s=a.shape)
    ac = np.fft.fftshift(ac)
    if normalize:
        peak = np.nanmax(np.abs(ac))
        if peak > 0:
            ac = ac / peak
    return ac


def autocorrelation_q_axes(
    kx: np.ndarray, ky: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """q axes (qx, qy) matching :func:`fs_autocorrelation`'s centered output.

    Same pixel spacing as the input k axes, centered on q=0. Lengths follow the
    fftshift convention (center index = ``n // 2``).
    """
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    nx, ny = kx.size, ky.size
    dkx = float(np.mean(np.diff(kx))) if nx > 1 else 1.0
    dky = float(np.mean(np.diff(ky))) if ny > 1 else 1.0
    qx = (np.arange(nx) - nx // 2) * dkx
    qy = (np.arange(ny) - ny // 2) * dky
    return qx, qy


def autocorrelation_peaks(
    ac: np.ndarray, qx: np.ndarray, qy: np.ndarray,
    *, min_radius_frac: float = 0.08, n_peaks: int = 3,
) -> list[dict]:
    """Return the strongest off-center C(q) peaks as ``{qx, qy, q, value}``.

    Peaks within ``min_radius_frac`` of the full q-range from q=0 are skipped so
    the trivial self-overlap maximum at the origin is excluded.
    """
    ac = np.asarray(ac, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    qxx, qyy = np.meshgrid(qx, qy)
    qr = np.hypot(qxx, qyy)
    q_span = max(float(np.ptp(qx)), float(np.ptp(qy)), 1e-9)
    mask = qr >= float(min_radius_frac) * q_span
    vals = np.where(mask, ac, -np.inf)
    out: list[dict] = []
    work = vals.copy()
    for _ in range(int(n_peaks)):
        idx = int(np.argmax(work))
        if not np.isfinite(work.flat[idx]):
            break
        iy, ix = np.unravel_index(idx, work.shape)
        out.append({
            "qx": float(qxx[iy, ix]), "qy": float(qyy[iy, ix]),
            "q": float(qr[iy, ix]), "value": float(ac[iy, ix]),
        })
        # suppress a small neighborhood so the next peak is distinct
        rr = max(work.shape) // 20 + 1
        y0, y1 = max(0, iy - rr), min(work.shape[0], iy + rr + 1)
        x0, x1 = max(0, ix - rr), min(work.shape[1], ix + rr + 1)
        work[y0:y1, x0:x1] = -np.inf
    return out
