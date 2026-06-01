"""Cycle de vie Γ (reset + badge état) — free functions prenant `ctrl` en arg.

Extrait de `gamma_controller.py` pour rester sous le plafond 700 LOC.
Pattern free-function + thin wrapper documenté dans CLAUDE.md.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QMessageBox


def format_badge_text(ctrl) -> str:
    """Texte court pour le badge statusbar (état Γ courant)."""
    ref = ctrl._stored_gamma_reference()
    meta = (ctrl._raw_data or {}).get("metadata", {}) or {}
    if meta.get("angle_offsets_applied"):
        ao = meta.get("angle_offsets_applied") or {}
        try:
            theta = float(ao.get("theta0_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            theta = 0.0
        return f"Γ loader-offset θ0={theta:+.3f}°"
    if not ref:
        return "Γ ∅"
    try:
        kx = float(ref.get("kx", 0.0) or 0.0)
        ky = float(ref.get("ky", 0.0) or 0.0)
    except (TypeError, ValueError):
        kx, ky = 0.0, 0.0
    src = ref.get("source", "?")
    same = ctrl._same_path(ref.get("path"), (ctrl._raw_data or {}).get("path")) \
        if ctrl._raw_data else False
    if meta.get("bm_gamma_axis_centered") or meta.get("fs_gamma_axis_centered"):
        state = "appliqué" if same else "propagé"
    else:
        state = "stocké"
    return f"Γ kx={kx:+.3f} ky={ky:+.3f} · {src} · {state}"


def update_badge(ctrl) -> None:
    lbl = getattr(ctrl._parent, "_gamma_status_label", None)
    if lbl is None:
        return
    try:
        lbl.setText(format_badge_text(ctrl))
    except Exception:
        pass


def forget_with_confirm(ctrl, gamma_meta_keys: tuple) -> None:
    """Dialog de confirmation puis appel `forget(ctrl)`."""
    n_files = len(getattr(ctrl._session, "files", {}) or {})
    reply = QMessageBox.question(
        ctrl._parent, "Oublier Γ",
        f"Effacer la référence Γ session et restaurer l'axe brut ?\n"
        f"Impact : {n_files} fichier(s) de la session. fit_result sera remappé "
        f"en sens inverse pour rester aligné avec l'axe brut.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        forget(ctrl, gamma_meta_keys)


def forget(ctrl, gamma_meta_keys: tuple) -> None:
    """Réinitialise tout l'état Γ (session + entry + raw_data).

    Porte de sortie aux gardes `_is_axis_locked` : inverse le shift d'axe en
    utilisant la valeur courante de `bm_gamma_axis_shift`, remap `fit_result`
    en sens inverse, puis efface tous les flags Γ + références session.
    """
    meta = ctrl._raw_data.get("metadata", {}) if ctrl._raw_data else {}
    try:
        previous_shift = float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0)
    except (TypeError, ValueError):
        previous_shift = 0.0
    was_centered = bool(
        meta.get("bm_gamma_axis_centered") or meta.get("fs_gamma_axis_centered")
    )

    if ctrl._raw_data is not None and was_centered and abs(previous_shift) > 1e-12:
        kpar = np.asarray(ctrl._raw_data.get("kpar"), dtype=float)
        if kpar.size:
            ctrl._raw_data["kpar"] = kpar + previous_shift
        if meta.get("fs_data") is not None:
            fs_kx = meta.get("fs_kx")
            if fs_kx is not None:
                meta["fs_kx"] = np.asarray(fs_kx, dtype=float) + previous_shift
            try:
                previous_ky = float(meta.get("fs_gamma_axis_shift_ky", 0.0) or 0.0)
            except (TypeError, ValueError):
                previous_ky = 0.0
            if abs(previous_ky) > 1e-12:
                fs_ky = meta.get("fs_ky")
                if fs_ky is not None:
                    meta["fs_ky"] = np.asarray(fs_ky, dtype=float) + previous_ky
        ctrl._remap_fit_results_by_delta(-previous_shift)
        if hasattr(ctrl, "_sel_k"):
            ctrl._sel_k = float(ctrl._sel_k + previous_shift)

    if ctrl._raw_data is not None:
        for k in gamma_meta_keys:
            meta.pop(k, None)

    ctrl._session.gamma_reference = {}
    ctrl._session.angle_offsets = {}

    entry = ctrl._current_entry()
    if entry is not None:
        entry.meta_gamma_state = {}
        entry.fs_center_kx = None
        entry.fs_center_ky = None
        entry.fit_params.center_init = 0.0

    sp_cx = getattr(ctrl._params, "sp_cx", None)
    if sp_cx is not None and hasattr(sp_cx, "setValue"):
        had_block = sp_cx.blockSignals(True) if hasattr(sp_cx, "blockSignals") else False
        try:
            sp_cx.setValue(0.0)
        finally:
            if hasattr(sp_cx, "blockSignals"):
                try:
                    sp_cx.blockSignals(had_block)
                except Exception:
                    pass

    if hasattr(ctrl, "_fs_controls"):
        try:
            ctrl._fs_controls.set_center(0.0, 0.0)
        except Exception:
            pass

    try:
        ctrl._session.save()
    except Exception as exc:
        ctrl._status(f"Attention: sauvegarde après reset Γ échouée : {exc}")

    ctrl._status("Γ réinitialisé : références session, axes et flags effacés.")
    update_badge(ctrl)
    try:
        ctrl._draw_current_view()
    except Exception:
        pass
