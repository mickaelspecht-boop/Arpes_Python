"""Controller UI pour interactions souris/spinbox + scheduling redraws.

Sort de ArpesExplorer toute la logique d'interaction temps-réel :
- ROI fit (cliquer-glisser un rectangle sur la carte BM/MDC)
- click sur carte → MDC/EDC à (k, E)
- spinbox énergie → resync sélection
- view changed → redessine
- debouncers (`_redraw_timer`, `_fit_redraw_timer`) → redraws séquentiels
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog, QToolTip
from matplotlib.patches import Rectangle

from arpes.core.undo import UndoFrame


class InteractionController:
    def __init__(self, parent):
        self._parent = parent

    @property
    def _params(self):
        return self._parent._params

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ---------------------------------------------------------------- callbacks
    def _on_view_changed(self):
        p = self._parent
        if hasattr(p, "_cmb_view_fit") and p._cmb_view_fit.currentText() != p._cmb_view.currentText():
            p._cmb_view_fit.blockSignals(True)
            p._cmb_view_fit.setCurrentText(p._cmb_view.currentText())
            p._cmb_view_fit.blockSignals(False)
        mode = p._cmb_view.currentText()
        if p._current_path:
            entry = p._session.get_or_create(p._session.key_for_path(p._current_path))
            entry.view_mode = mode
            entry.edcnorm = mode == "EDCnorm"
        p._update_display_data()
        p._draw_bm()
        if p._tabs.currentIndex() == 1:
            p._draw_mdc_edc()

    def _on_view_fit_changed(self):
        p = self._parent
        if p._cmb_view.currentText() != p._cmb_view_fit.currentText():
            p._cmb_view.blockSignals(True)
            p._cmb_view.setCurrentText(p._cmb_view_fit.currentText())
            p._cmb_view.blockSignals(False)
        self._on_view_changed()

    def _on_ev_spinbox_changed(self, val: float):
        p = self._parent
        if p._raw_data is None:
            return
        ev_arr = p._raw_data["ev_arr"]
        p._sel_ev = float(np.clip(val, ev_arr.min(), ev_arr.max()))
        p._draw_bm()
        p._draw_mdc_edc()
        if hasattr(p, "_mdc_fit_tabs") and p._tabs.currentIndex() == 1 and p._mdc_fit_tabs.currentIndex() == 1:
            p._draw_mdc_waterfall()

    def _schedule_model_redraw(self, _=None):
        self._parent._redraw_timer.start(120)

    def _schedule_fit_only_redraw(self, _=None):
        self._parent._fit_redraw_timer.start(120)

    def _on_model_changed(self, _=None):
        p = self._parent
        p._update_display_data()
        p._draw_bm()
        if p._tabs.currentIndex() == 1:
            p._draw_mdc_edc()
            if hasattr(p, "_mdc_fit_tabs") and p._mdc_fit_tabs.currentIndex() == 1:
                p._draw_mdc_waterfall()

    def _on_fit_only_changed(self, _=None):
        p = self._parent
        if p._tabs.currentIndex() != 1:
            return
        p._draw_mdc_edc()
        if hasattr(p, "_mdc_fit_tabs") and p._mdc_fit_tabs.currentIndex() == 1:
            p._draw_mdc_waterfall()

    # ---------------------------------------------------------------- ROI fit
    def _set_fit_roi_pick_mode(self, active: bool):
        p = self._parent
        active = bool(active)
        if not active and p._fit_roi_rect is not None:
            try:
                canvas = p._fit_roi_rect.figure.canvas
                p._fit_roi_rect.remove()
                canvas.draw_idle()
            except Exception:
                pass
        p._fit_roi_active = active
        p._fit_roi_start = None
        p._fit_roi_ax = None
        p._fit_roi_rect = None
        self._params.set_fit_roi_active(active)
        for canv in (getattr(p, "_bm_canvas", None), getattr(p, "_mdc_map_canvas", None)):
            if canv is None or not hasattr(canv, "canvas"):
                continue
            if active:
                canv.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                canv.canvas.unsetCursor()
        if active:
            if p._tabs.currentIndex() not in (0, 1):
                p._tabs.setCurrentIndex(1)
            self._status("Sélection zone fit : cliquer-glisser un rectangle sur la carte.")

    def _on_fit_roi_press(self, event):
        p = self._parent
        if not p._fit_roi_active:
            return
        if event.inaxes not in (p._bm_canvas.ax, p._mdc_map_canvas.ax):
            return
        button = getattr(event.button, "value", event.button)
        if button != 1 or event.xdata is None or event.ydata is None:
            return
        p._fit_roi_start = (float(event.xdata), float(event.ydata))
        p._fit_roi_ax = event.inaxes
        if p._fit_roi_rect is not None:
            try:
                p._fit_roi_rect.remove()
            except Exception:
                pass
        p._fit_roi_rect = Rectangle(
            p._fit_roi_start, 0.0, 0.0,
            fill=False, edgecolor="#38bdf8", linewidth=1.4,
            linestyle="-", alpha=0.95, zorder=20,
        )
        event.inaxes.add_patch(p._fit_roi_rect)
        event.canvas.draw_idle()

    def _on_fit_roi_motion(self, event):
        p = self._parent
        if not p._fit_roi_active or p._fit_roi_start is None or p._fit_roi_rect is None:
            return
        if event.inaxes is not p._fit_roi_ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = p._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        p._fit_roi_rect.set_x(min(x0, x1))
        p._fit_roi_rect.set_y(min(y0, y1))
        p._fit_roi_rect.set_width(abs(x1 - x0))
        p._fit_roi_rect.set_height(abs(y1 - y0))
        event.canvas.draw_idle()

    def _on_fit_roi_release(self, event):
        p = self._parent
        if not p._fit_roi_active or p._fit_roi_start is None:
            return
        if event.inaxes is not p._fit_roi_ax or event.xdata is None or event.ydata is None:
            self._set_fit_roi_pick_mode(False)
            return
        x0, y0 = p._fit_roi_start
        x1, y1 = float(event.xdata), float(event.ydata)
        if abs(x1 - x0) < 1e-4 or abs(y1 - y0) < 1e-4:
            self._set_fit_roi_pick_mode(False)
            return
        self._apply_fit_roi_from_bounds(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._set_fit_roi_pick_mode(False)

    def _apply_fit_roi_from_bounds(self, k0: float, k1: float, e0: float, e1: float):
        p = self._parent
        if p._raw_data is None:
            return
        d = p._raw_data
        k0 = float(np.clip(k0, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        k1 = float(np.clip(k1, np.nanmin(d["kpar"]), np.nanmax(d["kpar"])))
        e0 = float(np.clip(e0, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        e1 = float(np.clip(e1, np.nanmin(d["ev_arr"]), np.nanmax(d["ev_arr"])))
        if k1 <= k0 or e1 <= e0:
            return
        for sp, val in (
            (self._params.sp_kmin, k0), (self._params.sp_kmax, k1),
            (self._params.sp_evs, e0), (self._params.sp_eve, e1),
        ):
            sp.blockSignals(True)
            sp.setValue(float(val))
            sp.blockSignals(False)
        p._sel_k = float((k0 + k1) * 0.5)
        p._sel_ev = float((e0 + e1) * 0.5)
        self._sync_ev_spinbox()
        self._params.params_changed.emit()
        p._draw_bm()
        p._draw_mdc_edc()
        self._status(
            f"Zone fit : k={k0:+.3f}→{k1:+.3f} π/a, "
            f"E={e0:+.3f}→{e1:+.3f} eV"
        )

    # ----------------------------------------------------- selection points fit
    _DRAG_THRESHOLD_PX = 4.0
    _PICK_RADIUS_PX = 12.0
    _UNDO_STACK_MAX = 20

    def _on_fit_select_press(self, event):
        p = self._parent
        if getattr(p, "_fit_roi_active", False):
            return
        if event.inaxes not in (
            getattr(p._bm_canvas, "ax", None),
            getattr(p._mdc_map_canvas, "ax", None),
        ):
            return
        button = getattr(event.button, "value", event.button)
        if button != 1 or event.xdata is None or event.ydata is None:
            return
        p._fit_select_press_xy = (float(event.xdata), float(event.ydata))
        p._fit_select_press_ax = event.inaxes
        p._fit_select_consumed_click = False
        p._fit_select_modifier = bool(getattr(event, "key", "") or "") and "shift" in str(event.key).lower()

    def _on_fit_annotate_press(self, event):
        p = self._parent
        if getattr(p, "_fit_roi_active", False):
            return
        if event.inaxes not in (
            getattr(p._bm_canvas, "ax", None),
            getattr(p._mdc_map_canvas, "ax", None),
        ):
            return
        button = getattr(event.button, "value", event.button)
        if button != 3 or event.xdata is None or event.ydata is None:
            return
        if p._fit_res is None or not getattr(p, "_current_path", None):
            return
        e_fit = np.asarray(p._fit_res.get("e_fitted", []), dtype=float)
        if e_fit.size == 0:
            return
        try:
            click_disp = event.inaxes.transData.transform(
                (float(event.xdata), float(event.ydata))
            )
        except Exception:
            return
        nearest = self._find_nearest_kf_point(
            p._fit_res, e_fit, event.inaxes, click_disp,
            pixel_radius=self._PICK_RADIUS_PX,
        )
        if nearest is None:
            return
        branch, pair_idx, point_idx = nearest
        current = self._annotation_text_for_point(branch, pair_idx, point_idx)
        text, ok = QInputDialog.getMultiLineText(
            p,
            "Annotation point fit",
            f"Note pour {branch}, paire {pair_idx + 1}, point {point_idx}:",
            current,
        )
        if not ok:
            return
        text = text.strip()
        if not text:
            self._status("Annotation vide ignorée.")
            return
        key = p._session.key_for_path(p._current_path)
        entry = p._session.get_or_create(key)
        annotations = dict(entry.annotations or {})
        branch_notes = list(annotations.get(branch, []))
        note = {
            "pair": int(pair_idx),
            "index": int(point_idx),
            "text": text,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        replaced = False
        for idx, existing in enumerate(branch_notes):
            if existing.get("pair") == pair_idx and existing.get("index") == point_idx:
                branch_notes[idx] = note
                replaced = True
                break
        if not replaced:
            branch_notes.append(note)
        annotations[branch] = branch_notes
        entry.annotations = annotations
        p._session.save()
        self._status("Annotation enregistrée.")
        p._draw_bm()
        if getattr(p, "_tabs", None) is not None and p._tabs.currentIndex() == 1:
            p._draw_mdc_edc()

    def _on_fit_annotation_motion(self, event):
        p = self._parent
        if event.inaxes not in (
            getattr(p._bm_canvas, "ax", None),
            getattr(p._mdc_map_canvas, "ax", None),
        ):
            QToolTip.hideText()
            return
        if event.xdata is None or event.ydata is None or p._fit_res is None:
            QToolTip.hideText()
            return
        e_fit = np.asarray(p._fit_res.get("e_fitted", []), dtype=float)
        if e_fit.size == 0:
            QToolTip.hideText()
            return
        try:
            click_disp = event.inaxes.transData.transform(
                (float(event.xdata), float(event.ydata))
            )
        except Exception:
            QToolTip.hideText()
            return
        nearest = self._find_nearest_kf_point(
            p._fit_res, e_fit, event.inaxes, click_disp,
            pixel_radius=self._PICK_RADIUS_PX,
        )
        if nearest is None:
            QToolTip.hideText()
            return
        branch, pair_idx, point_idx = nearest
        text = self._annotation_text_for_point(branch, pair_idx, point_idx)
        if not text:
            QToolTip.hideText()
            return
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is None or not hasattr(gui_event, "globalPosition"):
            return
        QToolTip.showText(gui_event.globalPosition().toPoint(), text, event.canvas)

    def _annotation_text_for_point(self, branch: str, pair_idx: int, point_idx: int) -> str:
        p = self._parent
        if not getattr(p, "_current_path", None):
            return ""
        key = p._session.key_for_path(p._current_path)
        entry = p._session.get_or_create(key)
        for note in (entry.annotations or {}).get(branch, []):
            if note.get("pair") == pair_idx and note.get("index") == point_idx:
                return str(note.get("text", ""))
        return ""

    def _on_fit_select_motion(self, event):
        p = self._parent
        if getattr(p, "_fit_roi_active", False):
            return
        start = getattr(p, "_fit_select_press_xy", None)
        ax = getattr(p, "_fit_select_press_ax", None)
        if start is None or ax is None or event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        try:
            d0 = ax.transData.transform(start)
            d1 = ax.transData.transform((float(event.xdata), float(event.ydata)))
        except Exception:
            return
        dx, dy = d1[0] - d0[0], d1[1] - d0[1]
        if (dx * dx + dy * dy) ** 0.5 < self._DRAG_THRESHOLD_PX:
            return
        rect = getattr(p, "_fit_select_rect", None)
        if rect is None:
            rect = Rectangle(start, 0.0, 0.0, fill=False, edgecolor="#fbbf24",
                             linewidth=1.2, linestyle="--", alpha=0.95, zorder=22)
            ax.add_patch(rect)
            p._fit_select_rect = rect
        x0, y0 = start
        x1, y1 = float(event.xdata), float(event.ydata)
        rect.set_x(min(x0, x1))
        rect.set_y(min(y0, y1))
        rect.set_width(abs(x1 - x0))
        rect.set_height(abs(y1 - y0))
        event.canvas.draw_idle()

    def _on_fit_select_release(self, event):
        p = self._parent
        if getattr(p, "_fit_roi_active", False):
            return
        start = getattr(p, "_fit_select_press_xy", None)
        ax = getattr(p, "_fit_select_press_ax", None)
        rect = getattr(p, "_fit_select_rect", None)
        p._fit_select_press_xy = None
        p._fit_select_press_ax = None
        if start is None or ax is None:
            return
        if event.xdata is None or event.ydata is None or event.inaxes is not ax:
            self._discard_select_rect()
            return
        try:
            d0 = ax.transData.transform(start)
            d1 = ax.transData.transform((float(event.xdata), float(event.ydata)))
        except Exception:
            self._discard_select_rect()
            return
        moved = ((d1[0] - d0[0]) ** 2 + (d1[1] - d0[1]) ** 2) ** 0.5
        additive = bool(event.key and "shift" in str(event.key).lower())
        if moved < self._DRAG_THRESHOLD_PX:
            self._discard_select_rect()
            self._handle_single_click_selection(ax, d1, additive=additive)
        else:
            x0, y0 = start
            x1, y1 = float(event.xdata), float(event.ydata)
            self._discard_select_rect()
            self._handle_rect_selection(ax, sorted((x0, x1)), sorted((y0, y1)), additive=additive)
        p._draw_bm()

    def _discard_select_rect(self) -> None:
        p = self._parent
        rect = getattr(p, "_fit_select_rect", None)
        if rect is not None:
            try:
                canvas = rect.figure.canvas
                rect.remove()
                canvas.draw_idle()
            except Exception:
                pass
        p._fit_select_rect = None

    def _handle_single_click_selection(self, ax, click_disp, *, additive: bool) -> None:
        p = self._parent
        if p._fit_res is None:
            return
        e_fit = np.asarray(p._fit_res.get("e_fitted", []), dtype=float)
        if e_fit.size == 0:
            return
        nearest = self._find_nearest_kf_point(
            p._fit_res, e_fit, ax, click_disp, pixel_radius=self._PICK_RADIUS_PX,
        )
        if nearest is None:
            if not additive:
                if p._fit_selected:
                    p._fit_selected = []
                    self._status("Sélection vidée.")
            return
        if not additive:
            if nearest in p._fit_selected and len(p._fit_selected) == 1:
                p._fit_selected = []
            else:
                p._fit_selected = [nearest]
        else:
            if nearest in p._fit_selected:
                p._fit_selected.remove(nearest)
            else:
                p._fit_selected.append(nearest)
        self._status(f"{len(p._fit_selected)} point(s) sélectionné(s). Suppr pour retirer.")

    def _handle_rect_selection(self, ax, xs, ys, *, additive: bool) -> None:
        p = self._parent
        if p._fit_res is None:
            return
        fr = p._fit_res
        e_fit = np.asarray(fr.get("e_fitted", []), dtype=float)
        if e_fit.size == 0:
            return
        x0, x1 = xs
        y0, y1 = ys
        hits: list[tuple[str, int, int]] = []
        for branch in ("kF_minus", "kF_plus"):
            for pair_idx, raw in enumerate(fr.get(branch) or []):
                arr = np.asarray(raw, dtype=float)
                n = min(arr.size, e_fit.size)
                if n == 0:
                    continue
                k_arr = arr[:n]
                e_arr = e_fit[:n]
                mask = (
                    np.isfinite(k_arr) & np.isfinite(e_arr)
                    & (k_arr >= x0) & (k_arr <= x1)
                    & (e_arr >= y0) & (e_arr <= y1)
                )
                for idx in np.flatnonzero(mask):
                    hits.append((branch, pair_idx, int(idx)))
        if not additive:
            p._fit_selected = list(hits)
        else:
            existing = set(p._fit_selected)
            for h in hits:
                if h in existing:
                    existing.remove(h)
                else:
                    existing.add(h)
            p._fit_selected = list(existing)
        self._status(f"{len(p._fit_selected)} point(s) sélectionné(s). Suppr pour retirer.")

    def _delete_selected_fit_points(self) -> None:
        p = self._parent
        sel = list(getattr(p, "_fit_selected", []) or [])
        if not sel or p._fit_res is None:
            return
        fr = p._fit_res
        before = self._fit_branch_snapshot(fr)
        for branch, pair_idx, point_idx in sel:
            arr = list(fr[branch][pair_idx])
            if 0 <= point_idx < len(arr):
                arr[point_idx] = float("nan")
            fr[branch][pair_idx] = arr
        after = self._fit_branch_snapshot(fr)
        p._undo_stack.push(UndoFrame(
            action="fit_delete",
            data={"n_points": len(sel)},
            undo=lambda before=before: self._restore_fit_branches(before),
            redo=lambda after=after: self._restore_fit_branches(after),
        ))
        p._fit_selected = []
        self._persist_fit_result(fr)
        self._params.set_fit_undo_enabled(p._undo_stack.can_undo())
        self._status(f"{len(sel)} point(s) supprimé(s). « Annuler » pour restaurer.")
        p._draw_bm()

    def _undo_fit_delete(self) -> None:
        p = self._parent
        if p._fit_res is None:
            self._params.set_fit_undo_enabled(False)
            return
        frame = p._undo_stack.undo()
        self._params.set_fit_undo_enabled(p._undo_stack.can_undo())
        self._status("Suppression annulée." if frame else "Aucune action à annuler.")
        p._draw_bm()

    def _redo_fit_delete(self) -> None:
        p = self._parent
        if p._fit_res is None:
            return
        frame = p._undo_stack.redo()
        self._params.set_fit_undo_enabled(p._undo_stack.can_undo())
        self._status("Suppression réappliquée." if frame else "Aucune action à rétablir.")
        p._draw_bm()

    def _fit_branch_snapshot(self, fr: dict) -> dict:
        return {b: [list(arr) for arr in (fr.get(b) or [])] for b in ("kF_minus", "kF_plus")}

    def _restore_fit_branches(self, snapshot: dict) -> None:
        p = self._parent
        if p._fit_res is None:
            return
        for branch, arrays in snapshot.items():
            p._fit_res[branch] = [list(a) for a in arrays]
        p._fit_selected = []
        self._persist_fit_result(p._fit_res)

    def _find_nearest_kf_point(self, fr, e_fit, ax, click_disp, *, pixel_radius: float = 12.0):
        best = None
        best_dist2 = float(pixel_radius) ** 2
        for branch in ("kF_minus", "kF_plus"):
            arrays = fr.get(branch) or []
            for pair_idx, raw in enumerate(arrays):
                arr = np.asarray(raw, dtype=float)
                n = min(arr.size, e_fit.size)
                if n == 0:
                    continue
                pts_data = np.column_stack([arr[:n], e_fit[:n]])
                valid = np.all(np.isfinite(pts_data), axis=1)
                if not valid.any():
                    continue
                pts_disp = ax.transData.transform(pts_data[valid])
                d2 = np.sum((pts_disp - click_disp) ** 2, axis=1)
                local_idx = int(np.argmin(d2))
                global_idx = int(np.flatnonzero(valid)[local_idx])
                if d2[local_idx] < best_dist2:
                    best_dist2 = float(d2[local_idx])
                    best = (branch, pair_idx, global_idx)
        return best

    def _persist_fit_result(self, fr: dict) -> None:
        p = self._parent
        p._fit_res = fr
        path = getattr(p, "_current_path", None)
        if path:
            key = p._session.key_for_path(path)
            entry = p._session.get_or_create(key)
            entry.fit_result = fr
            p._session.save()
        results = getattr(p, "_results", None)
        if results is not None and hasattr(results, "refresh_physics_only"):
            try:
                results.refresh_physics_only()
            except Exception:
                pass

    def _reset_fit_roi_range(self):
        p = self._parent
        if p._raw_data is None:
            return
        d = p._raw_data
        self._apply_fit_roi_from_bounds(
            float(np.nanmin(d["kpar"])), float(np.nanmax(d["kpar"])),
            float(np.nanmin(d["ev_arr"])), float(np.nanmax(d["ev_arr"])),
        )

    def _on_map_click(self, event):
        p = self._parent
        if p._fit_roi_active:
            return
        if event.inaxes not in (p._bm_canvas.ax, p._mdc_map_canvas.ax):
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._is_click_on_fit_point(event):
            return
        d = p._raw_data
        p._sel_ev = float(np.clip(event.ydata, d["ev_arr"].min(), d["ev_arr"].max()))
        p._sel_k = float(np.clip(event.xdata, d["kpar"].min(), d["kpar"].max()))
        self._sync_ev_spinbox()
        p._draw_bm()
        p._draw_mdc_edc()

    def _is_click_on_fit_point(self, event) -> bool:
        p = self._parent
        if p._fit_res is None:
            return False
        e_fit = np.asarray(p._fit_res.get("e_fitted", []), dtype=float)
        if e_fit.size == 0:
            return False
        try:
            disp = event.inaxes.transData.transform(
                (float(event.xdata), float(event.ydata))
            )
        except Exception:
            return False
        return self._find_nearest_kf_point(
            p._fit_res, e_fit, event.inaxes, disp, pixel_radius=self._PICK_RADIUS_PX,
        ) is not None

    def _sync_ev_spinbox(self):
        self._params.sp_ev.blockSignals(True)
        self._params.sp_ev.setValue(self._parent._sel_ev)
        self._params.sp_ev.blockSignals(False)
