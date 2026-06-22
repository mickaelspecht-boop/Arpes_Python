"""Controller UI pour la correction de distorsion BM (trapèze + parabole).

Pipeline pur dans `arpes/physics/distortion.py`. Ce controller :
- lit/écrit `entry.bm_distortion`
- recompose la config depuis le panneau (`panel.bm_distortion_params`)
- déclenche `_update_display_data` + `_draw_current_view`
- redteam guardrails (FS data, angle_offsets hash, active EF calibration)
- store calibrations partagées dans `~/.config/arpes/distortion_calib.json`
  (clé `(lens_mode, pass_energy, hv)`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from arpes.core import processing_history as ph
from arpes.physics.distortion import (
    angle_offsets_hash,
    auto_detect_parabola,
    auto_detect_trapezoid,
    calib_key_for_meta,
    clamp_params,
    gamma_shift_signature,
    get_cfg_summary,
    is_distortion_active,
    signal_bbox,
)


def _calib_store_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "arpes" / "distortion_calib.json"


def _load_calib_store() -> dict:
    p = _calib_store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_calib_store(store: dict) -> None:
    p = _calib_store_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(store, indent=2))
    except Exception:
        pass


def _calib_key_str(meta: dict | None) -> str:
    return "|".join(calib_key_for_meta(meta))


class DistortionController:
    # P3.1: writes through to parent are now allow-listed (fail-loud on typo).
    # Reads stay forwarded (a typo'd read already raises via the parent).
    _OWN_ATTRS = frozenset({"_parent"})
    _PARENT_WRITES = frozenset()

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif name in self._PARENT_WRITES:
            setattr(self._parent, name, value)
        else:
            raise AttributeError(
                f"{type(self).__name__} refuses to write '{name}': missing from "
                "_PARENT_WRITES (typo?). Add it to _PARENT_WRITES "
                "if the parent attribute is legitimate."
            )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ef_calib_in_progress(self) -> bool:
        dlg = getattr(self._parent, "_active_ef_calib_dialog", None)
        try:
            return bool(dlg) and dlg.isVisible()
        except Exception:
            return False

    def _current_meta(self) -> dict:
        if self._raw_data is None:
            return {}
        return self._raw_data.get("metadata", {}) or {}

    # ── apply / reset ───────────────────────────────────────────────────────
    def _apply_bm_distortion(self):
        if self._raw_data is None or not self._current_path:
            QMessageBox.warning(self._parent, "BM distortion",
                                "Load a band map first.")
            return
        meta = self._current_meta()
        data = np.asarray(self._raw_data.get("data"), dtype=float)
        if data.ndim != 2:
            QMessageBox.warning(self._parent, "BM distortion",
                                "Correction is only available on a 2D band map.")
            return
        if self._ef_calib_in_progress():
            QMessageBox.warning(self._parent, "BM distortion",
                                "EF calibration in progress — close the dialog first.")
            return

        cfg = self._params.bm_distortion_params()
        if not is_distortion_active(cfg):
            QMessageBox.warning(self._parent, "BM distortion",
                                "No active correction (slopes and a = 0).")
            return

        cfg["angle_offsets_hash"] = angle_offsets_hash(self._session.angle_offsets)
        cfg["gamma_shift_at_calib"] = gamma_shift_signature(meta)
        cfg["calib_key"] = list(calib_key_for_meta(meta))
        cfg["source"] = "manual"

        cfg_clamped = clamp_params(cfg, self._raw_data["kpar"], self._raw_data["ev_arr"])
        # Préserve les champs ajoutés (clamp ne les inclut que si présents).
        for k in ("angle_offsets_hash", "gamma_shift_at_calib", "calib_key", "source"):
            cfg_clamped[k] = cfg[k]

        try:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.bm_distortion = cfg_clamped
            self._session.save()
            self._save_to_calib_store(cfg_clamped, meta)
            self._parent._distortion_preview_visible = False
            # Invalidation explicite cache FS volume : si propagate_distortion_to_fs
            # actif, la prochaine visite onglet FS doit re-warper avec la NOUVELLE
            # calibration BM (cache_signature détecte déjà, mais on force pour clarté).
            try:
                self._parent._fs_distortion_cache_invalidate()
            except Exception:
                pass
            self._update_display_data()
            self._draw_current_view()
            msg = get_cfg_summary(cfg_clamped)
            self._params.lbl_distortion.setText(msg)
            self._status(msg)
            ph.log_action(self._parent,
                ph.CAT_DISTORT, "BM distortion on", entry=entry, summary=msg,
            )
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done("BM distortion applied")
        except Exception as exc:
            QMessageBox.warning(self._parent, "BM distortion", str(exc))
            self._status(f"Warning: BM distortion: {exc}")

    def _reset_bm_distortion(self):
        if not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        # Garde une copie nulle bit-exact pour test de réversibilité aval.
        entry.bm_distortion = {}
        self._session.save()
        ph.log_action(self._parent,ph.CAT_DISTORT, "BM distortion off", entry=entry)
        self._params.set_bm_distortion_state({})
        self._parent._distortion_preview_visible = False
        self._update_display_data()
        self._draw_current_view()
        self._params.lbl_distortion.setText("BM distortion: disabled for this file.")
        self._status("BM distortion disabled for this file.")
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done("BM distortion disabled")

    # -- live preview overlay ------------------------------------------------
    def _on_distortion_preview_changed(self):
        """Enable the dotted BM overlay (hidden again after Apply)."""
        if self._raw_data is None:
            return
        cfg = self._params.bm_distortion_params()
        active = (
            (cfg["trapezoid"]["enabled"] and (
                abs(cfg["trapezoid"]["slope_left"]) > 0
                or abs(cfg["trapezoid"]["slope_right"]) > 0))
            or (cfg["parabola"]["enabled"] and abs(cfg["parabola"]["a"]) > 0)
        )
        self._parent._distortion_preview_visible = bool(active)
        timer = getattr(self._parent, "_distortion_preview_timer", None)
        if timer is not None:
            timer.start(100)
            return
        self._redraw_distortion_preview()

    def _redraw_distortion_preview(self):
        """Refresh only the visible BM overlay, without redrawing MDC/FS."""
        tabs = getattr(self._parent, "_tabs", None)
        if tabs is not None and tabs.currentIndex() != 0:
            return
        try:
            self._draw_bm(overlays_only=True)
        except Exception:
            pass

    def _distortion_preview_bbox(self, kpar, ev):
        data = np.asarray(self._raw_data["data"], dtype=float)
        key = (
            self._raw_data.get("path"),
            id(self._raw_data),
            id(self._raw_data.get("data")),
            data.shape,
            id(self._raw_data.get("kpar")),
            id(self._raw_data.get("ev_arr")),
        )
        if getattr(self._parent, "_distortion_preview_bbox_key", None) == key:
            cached = getattr(self._parent, "_distortion_preview_bbox", None)
            if cached is not None:
                return cached
        bbox = signal_bbox(data, kpar, ev, intensity_percentile=50.0)
        self._parent._distortion_preview_bbox_key = key
        self._parent._distortion_preview_bbox = bbox
        return bbox

    def _draw_distortion_preview_overlay(self, ax):
        """Trace en pointillé les contours du trapèze + de l'iso-énergie
        parabolique tels qu'ils apparaîtront avant correction. Caché si
        ``_distortion_preview_visible`` est False ou si une calibration
        a déjà été appliquée à la donnée affichée."""
        if not getattr(self._parent, "_distortion_preview_visible", False):
            return
        if self._raw_data is None:
            return
        cfg = self._params.bm_distortion_params()
        kpar = np.asarray(self._raw_data["kpar"], dtype=float)
        ev = np.asarray(self._raw_data["ev_arr"], dtype=float)
        if kpar.size < 2 or ev.size < 2:
            return
        # Bbox du signal effectif (intensité > p50). Sinon fallback fenêtre.
        bbox = self._distortion_preview_bbox(kpar, ev)
        k_min, k_max = bbox["k_min"], bbox["k_max"]
        ev_min, ev_max = bbox["ev_min"], bbox["ev_max"]

        trap = cfg.get("trapezoid") or {}
        para = cfg.get("parabola") or {}

        if trap.get("enabled") and (abs(float(trap.get("slope_left", 0.0) or 0.0)) > 0
                                    or abs(float(trap.get("slope_right", 0.0) or 0.0)) > 0):
            slope_l = float(trap["slope_left"])
            slope_r = float(trap["slope_right"])
            pivot = float(trap.get("pivot_ev")
                          if trap.get("pivot_ev") is not None
                          else 0.5 * (ev_min + ev_max))
            e_samples = np.linspace(ev_min, ev_max, 60)
            d_e = e_samples - pivot
            left_src = k_min - slope_l * d_e
            right_src = k_max + slope_r * d_e
            ax.plot(left_src, e_samples, "--", color="cyan", lw=1.3, alpha=0.85,
                    zorder=8, label="trap L (preview)")
            ax.plot(right_src, e_samples, "--", color="cyan", lw=1.3, alpha=0.85,
                    zorder=8)

        if para.get("enabled") and abs(float(para.get("a", 0.0) or 0.0)) > 0:
            a = float(para["a"])
            k0 = float(para["k0"])
            pivot_e = float(trap.get("pivot_ev")
                            if (trap and trap.get("pivot_ev") is not None)
                            else 0.5 * (ev_min + ev_max))
            k_samples = np.linspace(k_min, k_max, 200)
            # Overlay convention cohérente avec apply : e_src = E + a*(K-k0)².
            # Band source à E_peak(K) = pivot + a*(K-k0)² → overlay = pivot + a*(K-k0)².
            # Si a<0, parabole ouvre vers le bas (dispersion trou type cuprate).
            e_curve = pivot_e + a * (k_samples - k0) ** 2
            mask = (e_curve >= ev_min) & (e_curve <= ev_max)
            if mask.any():
                ax.plot(k_samples[mask], e_curve[mask], ":", color="magenta",
                        lw=1.6, alpha=0.9, zorder=8, label="parabole (preview)")

    # ── auto-detect ──────────────────────────────────────────────────────────
    def _auto_bm_distortion(self):
        if self._raw_data is None:
            QMessageBox.warning(self._parent, "BM distortion auto", "Load a band map first.")
            return
        meta = self._current_meta()
        if np.asarray(self._raw_data.get("data"), dtype=float).ndim != 2:
            QMessageBox.warning(self._parent, "BM distortion auto",
                                "Auto-detect is only available on a 2D band map.")
            return
        data = np.asarray(self._raw_data["data"], dtype=float)
        kpar = np.asarray(self._raw_data["kpar"], dtype=float)
        ev = np.asarray(self._raw_data["ev_arr"], dtype=float)
        trap = auto_detect_trapezoid(data, kpar, ev)
        para = auto_detect_parabola(data, kpar, ev)
        if trap is None and para is None:
            QMessageBox.information(self._parent, "BM distortion auto",
                                    "Insufficient dispersion or n_kpar < 16 — auto-detect refused. "
                                    "Enter the slopes / a, k0 manually.")
            return
        cfg = self._params.bm_distortion_params()
        if trap is not None:
            cfg.setdefault("trapezoid", {}).update({
                "enabled": True,
                "slope_left": float(trap["slope_left"]),
                "slope_right": float(trap["slope_right"]),
                "pivot_ev": float(trap["pivot_ev"]),
            })
        if para is not None:
            cfg.setdefault("parabola", {}).update({
                "enabled": True,
                "a": float(para["a"]),
                "k0": float(para["k0"]),
            })
        cfg["enabled"] = True
        self._params.set_bm_distortion_state(cfg)
        bits = []
        if trap:
            bits.append(f"trap L={trap['slope_left']:+.3f} R={trap['slope_right']:+.3f} "
                        f"(R²={trap['r2_left']:.2f}/{trap['r2_right']:.2f})")
        if para:
            bits.append(f"parabole a={para['a']:+.3f} k0={para['k0']:+.3f} "
                        f"(n={para['n_points']})")
        self._status("Auto-detected distortion: " + " | ".join(bits))
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done("auto-detect distortion")

    # ── calib partagée ──────────────────────────────────────────────────────
    def _save_to_calib_store(self, cfg: dict, meta: dict) -> None:
        """Sauvegarde la calibration sous `~/.config/arpes/distortion_calib.json`
        keyée par `(lens_mode, pass_energy, hv)`. Réutilisable par autres
        fichiers de même géométrie analyseur."""
        store = _load_calib_store()
        store[_calib_key_str(meta)] = {
            "trapezoid": dict(cfg.get("trapezoid") or {}),
            "parabola": dict(cfg.get("parabola") or {}),
            "enabled": True,
            "source": "calib",
            "ts_meta": {
                "lens_mode": meta.get("lens_mode"),
                "pass_energy": meta.get("pass_energy"),
                "hv": meta.get("hv"),
            },
        }
        _save_calib_store(store)

    def _load_calib_for_current(self) -> dict | None:
        meta = self._current_meta()
        if not meta:
            return None
        store = _load_calib_store()
        return store.get(_calib_key_str(meta))

    def _apply_calib_for_current_if_any(self) -> None:
        """On file load, offer to apply a shared calibration if available.

        This applies when the local entry has no distortion but the
        `(lens, Ep, hv)` key has a stored calibration. It emits statusbar info
        and does not mutate without an explicit request, avoiding silent double
        application.
        """
        if self._raw_data is None:
            return
        meta = self._current_meta()
        if not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        if entry.bm_distortion:
            return
        calib = self._load_calib_for_current()
        if not calib:
            return
        if hasattr(self._params, "lbl_distortion"):
            self._params.lbl_distortion.setText(
                f"Calibration available (lens={meta.get('lens_mode','?')}, "
                f"hν={meta.get('hv')}). Click 'Apply' to load it."
            )
        self._status(f"Distortion calibration found for {_calib_key_str(meta)}")

    def _import_calib_to_current(self):
        """Explicit import of the shared calibration into the current file's entry."""
        calib = self._load_calib_for_current()
        if not calib or self._raw_data is None or not self._current_path:
            QMessageBox.information(self._parent, "BM distortion",
                                    "No shared calibration available for this geometry.")
            return
        meta = self._current_meta()
        cfg = {
            "enabled": True,
            "trapezoid": dict(calib.get("trapezoid") or {}),
            "parabola": dict(calib.get("parabola") or {}),
            "calib_key": list(calib_key_for_meta(meta)),
            "source": "calib_imported",
            "angle_offsets_hash": angle_offsets_hash(self._session.angle_offsets),
            "gamma_shift_at_calib": gamma_shift_signature(meta),
        }
        self._params.set_bm_distortion_state(cfg)
        self._apply_bm_distortion()

    # -- load-time guardrails ------------------------------------------------
    def _check_distortion_consistency_on_load(self) -> None:
        """On load, check angle_offsets hash versus stored calibration."""
        if self._raw_data is None or not self._current_path:
            return
        entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
        cfg = entry.bm_distortion
        if not cfg:
            return
        current_hash = angle_offsets_hash(self._session.angle_offsets)
        saved_hash = cfg.get("angle_offsets_hash")
        if saved_hash and saved_hash != current_hash:
            self._status(
                "Warning: angle_offsets changed since distortion calibration - "
                "recompute or disable the correction."
            )
        self._params.set_bm_distortion_state(cfg)
        if hasattr(self._params, "lbl_distortion"):
            self._params.lbl_distortion.setText(get_cfg_summary(cfg))
