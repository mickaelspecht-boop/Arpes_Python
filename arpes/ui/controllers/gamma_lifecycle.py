"""Γ lifecycle helpers for reset, persistence, and status-badge updates."""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QMessageBox


def format_badge_text(ctrl) -> str:
    """Short text for the status-bar badge (current Γ state)."""
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
        state = "applied" if same else "propagated"
    else:
        state = "stored"
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
    """Show confirmation dialog, then call `forget(ctrl)`."""
    n_files = len(getattr(ctrl._session, "files", {}) or {})
    reply = QMessageBox.question(
        ctrl._parent, "Forget Γ",
        f"Clear the session Γ reference and restore the raw axis?\n"
        f"Impact: {n_files} session file(s). fit_result will be remapped "
        f"in the opposite direction to stay aligned with the raw axis.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        forget(ctrl, gamma_meta_keys)


def forget(ctrl, gamma_meta_keys: tuple) -> None:
    """Reset the full Γ state (session + entry + raw_data).

    Escape hatch for `_is_axis_locked` guards: reverse the axis shift using
    the current `bm_gamma_axis_shift`, remap `fit_result` in the opposite
    direction, then clear all Γ flags and session references.
    """
    meta = ctrl._raw_data.get("metadata", {}) if ctrl._raw_data else {}
    try:
        previous_shift = float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0)
    except (TypeError, ValueError):
        previous_shift = 0.0
    was_centered = bool(
        meta.get("bm_gamma_axis_centered") or meta.get("fs_gamma_axis_centered")
    )
    # Loader-baked Γ (offset applied inside the loader at read time) cannot be
    # undone by axis arithmetic here: it needs a reload (handled at the end).
    loader_baked = bool(meta.get("angle_offsets_applied"))

    if ctrl._raw_data is not None and was_centered and abs(previous_shift) > 1e-12:
        raw_kpar = ctrl._raw_data.get("kpar")
        # Guard: np.asarray(None) yields a 0-d NaN array whose .size is 1, so the
        # old `if kpar.size` corrupted kpar into a NaN scalar for FS-only / kpar-less
        # raws. Only shift a real 1-D axis.
        kpar = np.asarray(raw_kpar, dtype=float) if raw_kpar is not None else None
        if kpar is not None and kpar.ndim >= 1 and kpar.size:
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
        ctrl._status(f"Warning: save after Γ reset failed: {exc}")

    # Loader-baked Γ: the angular offset was applied inside the loader at read
    # time, so the in-memory kpar is still shifted and `angle_offsets_applied`
    # is still set — the axis-arithmetic path above cannot undo it and
    # `_is_axis_locked` would stay True (permanent lock; "Forget Γ" was a no-op).
    # The session refs are now cleared, so `angle_offsets_for_load` returns no
    # offsets; reloading rebuilds a clean raw axis. (Bake source = stored Γ ref /
    # session.angle_offsets, both cleared above — reload does NOT re-bake.)
    if loader_baked:
        path = (ctrl._raw_data or {}).get("path") if ctrl._raw_data else None
        if not path:
            ctrl._status(
                "Γ reset: session refs cleared, but cannot reload to restore the "
                "raw axis (no file path). Re-open the file to clear the loader offset."
            )
            update_badge(ctrl)
            return
        ctrl._status("Γ reset: reloading file to clear the loader angular offset…")
        try:
            ctrl._reload_current_no_cache()
        except Exception as exc:
            ctrl._status(
                f"Γ reset: reload failed — loader offset NOT cleared: {exc}. "
                "Re-open the file manually."
            )
            return
        update_badge(ctrl)
        return

    ctrl._status("Γ reset: session references, axes, and flags cleared.")
    update_badge(ctrl)
    try:
        ctrl._draw_current_view()
    except Exception:
        pass
