"""FS Explorer controller: ARPEST-style browsing of an FS volume.

Single responsibility: drive the FS Explorer tab (iso-E map + cut line +
extracted BM + sweep animation). All sampling math lives headless in
arpes/physics/fs_explorer_compute.py.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QTimer

from arpes.physics.fs import _robust_norm
from arpes.physics.fs_explorer_compute import (
    downsample_volume,
    extract_bm_cut,
    extract_iso_e_slice,
    free_cut_allowed,
    native_cut,
    snap_to_native,
    volume_from_meta,
)

_ANIM_INTERVAL_MS = 66    # ~15 fps
_DRAG_SUBSAMPLE = 4       # k-space subsampling while dragging/animating
_DRAG_E_SUBSAMPLE = 2     # energy-axis subsampling while dragging/animating
_DRAG_THROTTLE_MS = 40    # cap live cut redraws at ~25 fps during drag
_SETTLE_MS = 400          # idle delay before the one full-res redraw
_CUT_N_PTS = 400


class FSExplorerController:
    """Owns the FS Explorer tab state (line, mode, energy, animation)."""

    def __init__(self, parent):
        self._parent = parent
        self._line = [0.0, 0.0, 0.0, 1.0]   # cx, cy, angle_deg, length
        self._mode = "free"
        self._dragging = False
        self._speed = 1.0
        self._anim_dir = 1.0
        self._vol_id = None                  # id() of fs_data → reset on change
        self._vol_cache = None               # (vol f32, kx, ky, e_ax) reuse
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(_ANIM_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._animation_step)
        # Mouse-move events arrive faster than we can extract+draw a cut:
        # keep only the latest line and apply it at most every 40 ms.
        self._pending_line = None
        self._throttle = QTimer()
        self._throttle.setSingleShot(True)
        self._throttle.setInterval(_DRAG_THROTTLE_MS)
        self._throttle.timeout.connect(self._apply_pending_line)
        # Spinbox clicks (angle/length) get a fast preview cut; the full-res
        # cut renders once, after the user stops clicking. A synchronous
        # full-res draw per click (~235 ms) queued up the clicks and kept
        # "playing" them long after the last one.
        self._settle = QTimer()
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._on_settle)

    # ----------------------------------------------------- proxies parent
    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    @property
    def _map_view(self):
        return getattr(self._parent, "_fs_explorer_map", None)

    @property
    def _cut_view(self):
        return getattr(self._parent, "_fs_explorer_cut", None)

    @property
    def _bar(self):
        return getattr(self._parent, "_fs_explorer_bar", None)

    # ------------------------------------------------------------ dispatch
    def _fs_explorer_action(self, verb: str, payload: dict | None = None):
        """Single PROXY_MAP entry. Verbs: draw, line_changed, drag_state,
        energy_changed, line_params, mode_changed, play_toggle,
        speed_changed, file_changed, tab_activated."""
        p = payload or {}
        if verb == "file_changed":
            return self._on_file_changed()
        if verb == "tab_activated":
            if not p.get("active", True):
                return self._stop_animation()
            return self._draw_all()
        if verb == "draw":
            return self._draw_all()
        if self._resolve_volume() is None:
            return None
        if verb == "line_changed":
            self._pending_line = [p["cx"], p["cy"], p["angle"], p["length"]]
            if self._dragging:
                if not self._throttle.isActive():
                    self._throttle.start()
                return None
            return self._apply_pending_line()
        if verb == "drag_state":
            self._dragging = bool(p.get("dragging"))
            if not self._dragging:
                self._throttle.stop()
                if self._pending_line is not None:
                    self._apply_pending_line()   # full-res (dragging False)
                else:
                    self._draw_cut(fast=False)
            return None
        if verb == "energy_changed":
            return self._draw_all(keep_line=True)
        if verb == "line_params":
            self._line[2] = float(p["angle"])
            self._line[3] = float(p["length"])
            self._push_line_to_map()
            self._pending_line = list(self._line)
            if not self._throttle.isActive():
                self._throttle.start()
            self._settle.start()
            return None
        if verb == "mode_changed":
            return self._on_mode_changed(str(p.get("mode", "free")))
        if verb == "play_toggle":
            return self._on_play_toggle(bool(p.get("play")))
        if verb == "speed_changed":
            self._speed = float(p.get("speed", 1.0))
            return None
        return None

    # ------------------------------------------------------------- volume
    def _resolve_volume(self, *, announce: bool = False):
        """(vol, kx, ky, e_ax, meta) or None; placeholders on refusal."""
        d = getattr(self._parent, "_raw_data", None)
        meta = (d or {}).get("metadata", {}) or {}
        if d is None or self._map_view is None:
            if announce:
                self._show_placeholders("No data loaded.")
            return None
        if meta.get("axes_raw_view"):
            if announce:
                self._show_placeholders(
                    "FS Explorer not available in browse-only mode "
                    "(raw θ/E axes).\nSet φ, a and hν via Samples… first.")
                self._status("⚠ FS Explorer — not available on raw axes.")
            return None
        fs_obj = meta.get("fs_data")
        if fs_obj is None:
            if announce:
                self._show_placeholders(
                    "No FS volume in this file.\n"
                    "Load a Fermi-surface scan (folder/fast map).")
            return None
        if id(fs_obj) == self._vol_id and self._vol_cache is not None:
            vol, kx, ky, e_ax = self._vol_cache
        else:
            try:
                # volume_from_meta casts to float32 (copy for float64 input):
                # do it once per volume, then reuse — never per drag frame.
                out = volume_from_meta(meta)
            except ValueError as exc:
                if announce:
                    self._show_placeholders(str(exc))
                    self._status(f"⚠ FS Explorer — {exc}")
                return None
            vol, kx, ky, e_ax = out
            if np.asarray(fs_obj).dtype != np.float32:
                self._status(
                    f"FS Explorer: volume cast to float32 for display "
                    f"({vol.nbytes / 1e6:.0f} MB)."
                )
            self._vol_id = id(fs_obj)
            self._vol_cache = (vol, kx, ky, e_ax)
            self._reset_line(kx, ky)
            if self._bar is not None:
                self._bar.set_energy_axis(e_ax)
            if not free_cut_allowed(meta) and self._mode == "free":
                self._on_mode_changed("native", announce=False)
                if self._bar is not None:
                    idx = self._bar.cmb_mode.findData("native")
                    self._bar.cmb_mode.blockSignals(True)
                    self._bar.cmb_mode.setCurrentIndex(idx)
                    self._bar.cmb_mode.blockSignals(False)
                self._status(
                    "FS Explorer: free cuts disabled — the two map axes are "
                    "in different units (fs_kind="
                    f"{meta.get('fs_kind', '?')}). Native BMs only.")
        return vol, kx, ky, e_ax, meta

    def _reset_line(self, kx, ky) -> None:
        self._stop_animation()
        self._line = [
            float(0.5 * (kx[0] + kx[-1])),
            float(0.5 * (ky[0] + ky[-1])),
            0.0,
            float(abs(kx[-1] - kx[0])),
        ]
        self._sync_bar_from_line()

    def _on_file_changed(self) -> None:
        self._stop_animation()
        self._vol_id = None
        self._vol_cache = None
        tabs = getattr(self._parent, "_tabs", None)
        if tabs is not None and tabs.currentWidget() is not None and (
            getattr(self._parent, "_fs_explorer_tab", None) is tabs.currentWidget()
        ):
            self._draw_all()

    # ------------------------------------------------------------- drawing
    def _show_placeholders(self, text: str) -> None:
        if self._map_view is not None:
            self._map_view.show_placeholder(text)
        if self._cut_view is not None:
            self._cut_view.show_placeholder("")

    def _axis_labels(self, meta) -> tuple[str, str, str]:
        if str(meta.get("fs_kind", "")) == "kxky":
            return "kx (π/a)", "ky (π/a)", "k along cut (π/a)"
        return "kx (axis units)", "scan axis", "kx (axis units)"

    def _draw_all(self, *, keep_line: bool = True):
        out = self._resolve_volume(announce=True)
        if out is None:
            return
        vol, kx, ky, e_ax, meta = out
        bar = self._bar
        e_cur = bar.current_energy() if bar is not None else float(e_ax[-1])
        img = _robust_norm(extract_iso_e_slice(vol, e_ax, e_cur))
        xl, yl, _ = self._axis_labels(meta)
        self._map_view.draw_map(
            img, kx, ky, xlabel=xl, ylabel=yl,
            title=f"iso-E map @ E−EF = {e_cur:+.3f} eV",
            equal_aspect=str(meta.get("fs_kind", "")) == "kxky",
        )
        self._push_line_to_map()
        self._draw_cut(fast=False)

    def _push_line_to_map(self) -> None:
        if self._map_view is not None:
            self._map_view.set_line(*self._line)

    def _sync_bar_from_line(self) -> None:
        if self._bar is not None:
            self._bar.set_line_params(self._line[2], self._line[3])

    def _snap_line_native(self) -> None:
        out = self._resolve_volume()
        if out is None:
            return
        _vol, kx, ky, _e_ax, _meta = out
        self._line[1] = float(ky[snap_to_native(ky, self._line[1])])
        self._line[2] = 0.0
        self._line[0] = float(0.5 * (kx[0] + kx[-1]))
        self._line[3] = float(abs(kx[-1] - kx[0]))
        self._push_line_to_map()

    def _apply_pending_line(self):
        if self._pending_line is None:
            return
        self._line = self._pending_line
        self._pending_line = None
        if self._mode == "native":
            self._snap_line_native()
        self._sync_bar_from_line()
        # Fast preview while the user is still interacting (drag in progress
        # or settle window open); the settle timeout does the full-res pass.
        return self._draw_cut(fast=self._dragging or self._settle.isActive())

    def _on_settle(self):
        if self._dragging or self._anim_timer.isActive():
            return  # still interacting / sweeping: stay on fast frames
        self._apply_pending_line()
        self._draw_cut(fast=False)

    def _draw_cut(self, *, fast: bool):
        out = self._resolve_volume()
        if out is None or self._cut_view is None:
            return
        vol, kx, ky, e_ax, meta = out
        cx, cy, ang, length = self._line
        _xl, _yl, cut_xl = self._axis_labels(meta)
        bar = self._bar
        e_cur = bar.current_energy() if bar is not None else None
        e_used = e_ax[::_DRAG_E_SUBSAMPLE] if fast else e_ax
        if self._mode == "native":
            idx = snap_to_native(ky, cy)
            cut = native_cut(vol, kx, e_ax, idx)
            if fast:
                cut.image = cut.image[::2, ::_DRAG_E_SUBSAMPLE]
                cut.k_along = cut.k_along[::2]
            title = f"native BM — step {idx + 1}/{ky.size} @ {ky[idx]:+.3f}"
        else:
            v, kxs, kys = (downsample_volume(vol, kx, ky, _DRAG_SUBSAMPLE)
                           if fast else (vol, kx, ky))
            if fast:
                v = v[:, :, ::_DRAG_E_SUBSAMPLE]
            cut = extract_bm_cut(v, kxs, kys, e_used, cx=cx, cy=cy,
                                 angle_deg=ang, length=length,
                                 n_pts=_CUT_N_PTS // (2 if fast else 1))
            if cut is None:
                self._status("⚠ FS Explorer — zero-length cut line.")
                return
            title = f"cut @ ({cx:+.2f}, {cy:+.2f}), {ang:.0f}°"
            if cut.nan_fraction >= 0.95:
                self._status("⚠ FS Explorer — cut line entirely outside the volume.")
            elif cut.nan_fraction > 0:
                title += f"  ({cut.nan_fraction:.0%} outside volume)"
        self._cut_view.draw_cut(
            _robust_norm(cut.image), cut.k_along, e_used, e_current=e_cur,
            xlabel=cut_xl, ylabel="E − EF (eV)", title=title,
        )

    # ------------------------------------------------------------ modes
    def _on_mode_changed(self, mode: str, *, announce: bool = True) -> None:
        self._mode = "native" if mode == "native" else "free"
        out = self._resolve_volume()
        if out is None:
            return
        _vol, _kx, _ky, _e_ax, meta = out
        if self._mode == "free" and not free_cut_allowed(meta):
            self._mode = "native"
            if self._bar is not None:
                idx = self._bar.cmb_mode.findData("native")
                self._bar.cmb_mode.blockSignals(True)
                self._bar.cmb_mode.setCurrentIndex(idx)
                self._bar.cmb_mode.blockSignals(False)
            self._status(
                "FS Explorer: free cuts need both axes in k (fs_kind=kxky); "
                "this scan only supports Native BMs.")
        if self._mode == "native":
            self._snap_line_native()
            self._sync_bar_from_line()
        if announce:
            self._draw_cut(fast=False)

    # --------------------------------------------------------- animation
    def _on_play_toggle(self, play: bool) -> None:
        if play:
            self._anim_timer.start()
            self._status("▶ FS Explorer — sweeping the cut through the volume.")
        else:
            self._stop_animation()

    def _stop_animation(self) -> None:
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        if self._bar is not None:
            self._bar.stop_play()

    def _animation_step(self) -> None:
        if self._dragging:
            return
        out = self._resolve_volume()
        if out is None:
            self._stop_animation()
            return
        _vol, kx, ky, _e_ax, _meta = out
        cx, cy, ang, _length = self._line
        if self._mode == "native":
            step = max(1, int(round(self._speed)))
            idx = snap_to_native(ky, cy) + int(self._anim_dir) * step
            if idx <= 0 or idx >= ky.size - 1:
                idx = int(np.clip(idx, 0, ky.size - 1))
                self._anim_dir *= -1.0  # bounce at the volume edge
            self._line[1] = float(ky[idx])
        else:
            step = self._speed * float(abs(ky[-1] - ky[0])) / max(ky.size - 1, 1)
            a = np.deg2rad(ang)
            nx_, ny_ = -np.sin(a), np.cos(a)   # unit normal to the line
            cx += self._anim_dir * step * nx_
            cy += self._anim_dir * step * ny_
            in_x = min(kx[0], kx[-1]) <= cx <= max(kx[0], kx[-1])
            in_y = min(ky[0], ky[-1]) <= cy <= max(ky[0], ky[-1])
            if not (in_x and in_y):
                self._anim_dir *= -1.0  # bounce
                return
            self._line[0], self._line[1] = float(cx), float(cy)
        self._push_line_to_map()
        self._draw_cut(fast=True)
