"""Source de vérité unique pour la décision « comment appliquer Γ ».

P2 — fonction pure `resolve(raw, ref, hv, phi, entry_azi) → ResolvedGamma`
qui prend l'état brut et retourne la décision à appliquer. Aucune mutation,
aucun side-effect, testable headless.

Le controller UI (`gamma_controller.GammaController`) délègue à cette
fonction et utilise le résultat via un single-setter (`apply_gamma`).

Invariant idempotence prouvable :
    apply(resolve(raw, ref, …)) puis resolve(raw, ref, …) → ResolvedGamma
    avec `axis_shift_delta == 0` (rien à faire de plus).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from arpes.physics.gamma import (
    gamma_reference_to_bm_center,
    project_gamma_by_azi,
)


GammaMode = Literal[
    "none",          # rien à faire (pas de raw, pas de ref valide)
    "loader_baked",  # loader a déjà appliqué l'offset angulaire
    "axis_shifted",  # déplacer l'axe k de raw (delta peut être 0 si déjà à jour)
]


@dataclass(frozen=True)
class ResolvedGamma:
    """Décision d'application Γ pour un raw_data + ref donnés.

    Sémantique des champs :
    - ``mode`` : famille d'action (cf `GammaMode`).
    - ``display_center`` : valeur à pousser dans `sp_cx` (centre fit affiché).
    - ``fit_center_init`` : valeur à pousser dans `entry.fit_params.center_init`.
    - ``axis_shift_target`` : shift absolu (π/a) cible pour l'axe k.
    - ``axis_shift_delta`` : delta à appliquer MAINTENANT
      (=`target - current_shift_in_meta`).
    - ``fs_marker_kx/ky`` : coord à pousser dans `fs_controls.set_center`
      quand raw est FS (sinon NaN).
    - ``is_fs`` : raw est une FS (oriente l'usage du marker).
    - ``same_ref_path`` : la ref Γ vient du même fichier que raw.
    - ``reason`` : message human-readable pour statusbar (peut être "").
    - ``warning`` : message d'avertissement non bloquant (peut être "").
    """
    mode: GammaMode
    display_center: float
    fit_center_init: float
    axis_shift_target: float
    axis_shift_delta: float
    fs_marker_kx: float
    fs_marker_ky: float
    is_fs: bool
    same_ref_path: bool
    reason: str
    warning: str = ""


_NONE = ResolvedGamma(
    mode="none",
    display_center=0.0,
    fit_center_init=0.0,
    axis_shift_target=0.0,
    axis_shift_delta=0.0,
    fs_marker_kx=float("nan"),
    fs_marker_ky=float("nan"),
    is_fs=False,
    same_ref_path=False,
    reason="",
)


def _current_axis_shift(meta: dict | None) -> float:
    if not meta:
        return 0.0
    if not meta.get("bm_gamma_axis_centered") and not meta.get("fs_gamma_axis_centered"):
        return 0.0
    try:
        return float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _same_path(a, b) -> bool:
    if a is None or b is None:
        return False
    return str(a) == str(b)


def resolve(
    raw_data: dict | None,
    ref: dict | None,
    *,
    work_func: float,
    bm_hv: float | None = None,
    entry_azi: float | None = None,
    warn_collector: list | None = None,
) -> ResolvedGamma:
    """Décide quoi faire pour appliquer Γ sur ``raw_data``.

    Args:
        raw_data: dict loader (clés ``metadata``, ``kpar``, ``hv``, ``path``).
        ref: ``session.gamma_reference`` (peut être {} ou None).
        work_func: φ (eV) pour la projection FS→BM polar.
        bm_hv: surcharge du hv du raw (sinon lu depuis raw_data['hv']).
        entry_azi: azi du fichier courant (sinon lu depuis ref/meta).
        warn_collector: liste optionnelle où pousser warnings physics.

    Returns:
        `ResolvedGamma` immutable. Pas de mutation de raw_data ni de ref.
    """
    if raw_data is None:
        return _NONE
    meta = raw_data.get("metadata", {}) or {}
    is_fs = meta.get("fs_data") is not None
    hv = float(bm_hv) if bm_hv is not None else raw_data.get("hv")
    raw_path = raw_data.get("path")
    current_shift = _current_axis_shift(meta)

    # Cas 1 : loader-baked
    if meta.get("angle_offsets_applied"):
        return ResolvedGamma(
            mode="loader_baked",
            display_center=0.0,
            fit_center_init=0.0,
            axis_shift_target=0.0,
            axis_shift_delta=0.0,
            fs_marker_kx=0.0 if is_fs else float("nan"),
            fs_marker_ky=0.0 if is_fs else float("nan"),
            is_fs=is_fs,
            same_ref_path=False,
            reason="Γ offset angulaire loader actif",
        )

    # Cas 2 : pas de ref → on ne fait rien
    if not ref or not isinstance(ref, dict):
        return _NONE

    same_path = _same_path(ref.get("path"), raw_path)

    if is_fs:
        # Cas FS : marker = projection par azi (ou direct si même fichier).
        if same_path:
            kx_target = float(ref.get("kx", 0.0) or 0.0)
            ky_target = float(ref.get("ky", 0.0) or 0.0)
            warn = ""
        else:
            kx_target, ky_target = project_gamma_by_azi(
                ref, entry_azi,
                on_warn=(warn_collector.append if warn_collector is not None else None),
                warn_label="Γ référence → FS",
            )
            warn = ""
        if not np.isfinite(kx_target) or not np.isfinite(ky_target):
            return _NONE
        target_shift = float(kx_target)
        delta = target_shift - current_shift
        return ResolvedGamma(
            mode="axis_shifted",
            display_center=0.0,
            fit_center_init=0.0,
            axis_shift_target=target_shift,
            axis_shift_delta=delta,
            fs_marker_kx=float(kx_target),
            fs_marker_ky=float(ky_target),
            is_fs=True,
            same_ref_path=same_path,
            reason=(
                ""
                if same_path
                else f"Γ FS propagé par azimut : kx={kx_target:+.4f}, ky={ky_target:+.4f} π/a"
            ),
            warning=warn,
        )

    # Cas BM
    gamma_bm, correction = gamma_reference_to_bm_center(
        ref,
        bm_metadata=meta,
        bm_hv=hv,
        work_func=work_func,
        bm_azi=entry_azi,
        on_warn=(warn_collector.append if warn_collector is not None else None),
    )
    if not np.isfinite(gamma_bm):
        return _NONE
    target_shift = float(gamma_bm)
    delta = target_shift - current_shift
    return ResolvedGamma(
        mode="axis_shifted",
        display_center=0.0,
        fit_center_init=0.0,
        axis_shift_target=target_shift,
        axis_shift_delta=delta,
        fs_marker_kx=float("nan"),
        fs_marker_ky=float("nan"),
        is_fs=False,
        same_ref_path=same_path,
        reason=f"Γ mémorisé appliqué : {gamma_bm:+.4f} π/a  correction={correction:+.4f}",
    )
