"""Manual DFT alignment from user-placed high-symmetry points.

The user picks a high-symmetry label (Γ, X, M, …) and clicks where it sits on the
BM band map. Each placed anchor yields a pair ``(u, k_data)`` where:

- ``u`` = the label's *branch-local* k coordinate, i.e. exactly the quantity the
  overlay renderer multiplies by ``k_scale`` before adding ``k_shift``
  (``display_k = u * k_scale + k_shift``, see ``selection.displayed_k_axis``).
- ``k_data`` = the displayed k (π/a) where the user dropped it.

With ≥2 anchors a linear least-squares fit gives ``k_scale`` and ``k_shift`` so the
DFT maps onto the chosen points. With a single anchor the current scale is kept
and only the shift is solved (an anchor, like the manual Γ center generalised to
any label). Pure numpy — no PyQt, headless-testable.
"""
from __future__ import annotations

import numpy as np

from arpes.theory.data import TheoryBandData, TheoryOverlayConfig
from arpes.theory.selection import _branch_local_k


def _norm_label(value) -> str:
    return str(value or "").strip().upper().replace("GAMMA", "Γ")


def local_k_for_label(data: dict, config: dict, label: str) -> float | None:
    """Branch-local k coordinate of ``label`` (pre-scale), or None.

    Returns None when the label is unknown or falls outside the current segment
    (``_branch_local_k`` is NaN off-branch) — that anchor cannot be used.
    """
    labels = (data or {}).get("labels") or []
    target = _norm_label(label)
    raw_k = None
    for item in labels:
        if _norm_label(item.get("label")) == target and item.get("k") is not None:
            try:
                raw_k = float(item["k"])
            except (TypeError, ValueError):
                raw_k = None
            break
    if raw_k is None:
        return None
    d = TheoryBandData.from_dict(data)
    c = TheoryOverlayConfig.from_dict(config or {})
    k_full = np.asarray(d.k_distance, dtype=float)
    if k_full.size == 0:
        return None
    loc = _branch_local_k(d, c, k_full)
    idx = int(np.argmin(np.abs(k_full - raw_k)))
    if idx >= loc.size:
        return None
    u = float(loc[idx])
    return u if np.isfinite(u) else None


def fit_scale_shift(
    pairs, *, current_scale: float = 1.0, scale_bounds=(0.1, 5.0)
) -> tuple[float, float] | None:
    """Solve ``k_data = scale * u + shift`` for the placed anchors.

    ``pairs`` is an iterable of ``(u, k_data)``. ≥2 valid pairs → least squares;
    exactly 1 → keep ``current_scale`` and anchor the shift. Returns
    ``(scale, shift)`` (scale clamped to ``scale_bounds``) or None if no usable
    pair / degenerate (all u identical with ≥2 points).
    """
    pts = [
        (float(u), float(k))
        for u, k in pairs
        if u is not None and np.isfinite(u) and np.isfinite(k)
    ]
    if not pts:
        return None
    if len(pts) == 1:
        u, k = pts[0]
        scale = float(np.clip(current_scale, *scale_bounds))
        return scale, float(k - scale * u)
    u = np.asarray([p[0] for p in pts], dtype=float)
    k = np.asarray([p[1] for p in pts], dtype=float)
    if float(np.ptp(u)) < 1e-9:
        return None  # all anchors at the same DFT coordinate → scale undefined
    coeffs = np.linalg.lstsq(np.vstack([u, np.ones_like(u)]).T, k, rcond=None)[0]
    scale = float(np.clip(coeffs[0], *scale_bounds))
    return scale, float(coeffs[1])
