"""Gamma-related UI controller for ArpesExplorer."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from arpes.physics.fs import FermiSurfaceCanvas, FSControlPanel
from arpes.physics.gamma import (
    apply_bm_gamma_axis_shift as _gamma_apply_bm_axis_shift,
    build_gamma_reference as _gamma_build_reference,
    gamma_reference_to_bm_center as _gamma_ref_to_bm_center,
    k_to_angle_offset_deg as _gamma_k_to_angle_offset_deg,
    angle_offsets_from_k_center as _gamma_angle_offsets_from_k_center,
    project_gamma_by_azi as _gamma_project_by_azi,
    stored_gamma_reference as _gamma_stored_reference,
)


class GammaController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _store_fs_center_reference(self, kx: float, ky: float, *, source: str):
        if self._raw_data is None:
            return
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        self._session.gamma_reference = _gamma_build_reference(
            kx=kx, ky=ky,
            metadata=meta,
            hv=self._raw_data.get("hv"),
            path=self._raw_data.get("path"),
            azi=azi_ref,
            source=source,
            direction=(entry_now.meta.direction if entry_now else None),
        )
        offsets = self._angle_offsets_from_k_center(
            float(kx), float(ky),
            hv=self._raw_data.get("hv"),
            source=source,
            ref_path=self._raw_data.get("path"),
            azi=azi_ref,
        )
        if offsets:
            self._session.angle_offsets = offsets
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fs_center_kx = float(kx)
            entry.fs_center_ky = float(ky)
        self._session.save()

    def _k_to_angle_offset_deg(self, k_pi_a: float, *, hv: float | None = None) -> float | None:
        """Convertit un decalage k (pi/a) en offset angulaire CLS (wrapper UI)."""
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return None
        return _gamma_k_to_angle_offset_deg(k_pi_a, hv=hv_val, work_func=work_func)

    def _angle_offsets_from_k_center(
        self,
        kx: float,
        ky: float = 0.0,
        *,
        hv: float | None = None,
        source: str = "",
        ref_path: str | None = None,
        azi: float | None = None,
    ) -> dict:
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return {}
        return _gamma_angle_offsets_from_k_center(
            kx, ky,
            hv=hv_val, work_func=work_func,
            source=source, ref_path=ref_path, azi=azi,
        )

    def _project_gamma_by_azi(
        self,
        ref: dict,
        azi_target: float | None,
        *,
        warn_label: str = "Γ",
    ) -> tuple[float, float]:
        """Projette le Γ de référence dans le repère du fichier courant (wrapper UI)."""
        return _gamma_project_by_azi(
            ref, azi_target,
            on_warn=self._status,
            warn_label=warn_label,
        )

    def _set_fs_center_pick_mode(self, active: bool):
        active = bool(active)
        self._fs_pick_center_active = active
        if hasattr(self, "_fs_controls") and hasattr(self._fs_controls, "set_manual_center_active"):
            self._fs_controls.set_manual_center_active(active)
        if hasattr(self, "_fs_canvas") and hasattr(self._fs_canvas, "canvas"):
            if active:
                self._fs_canvas.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._fs_canvas.canvas.unsetCursor()
        if active:
            if self._tabs.currentIndex() != 3:
                self._tabs.setCurrentIndex(3)
            self._status("Centrage manuel Γ : clique sur le centre à viser dans la carte FS.")

    def _on_fs_map_click(self, event):
        if not getattr(self, "_fs_pick_center_active", False):
            return
        if self._raw_data is None or not self._current_is_fs():
            self._set_fs_center_pick_mode(False)
            QMessageBox.warning(self._parent, "Centrage manuel Γ", "Charge d'abord une FS.")
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        if not hasattr(self, "_fs_canvas") or event.inaxes is not getattr(self._fs_canvas, "ax", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        params = self._fs_controls.params()
        kx = float(params.kx_center + event.xdata)
        ky = float(params.ky_center + event.ydata)
        self._fs_controls.set_center(kx, ky)
        self._store_fs_center_reference(kx, ky, source="fs_manual")
        self._set_fs_center_pick_mode(False)
        self._draw_fs_tab()
        msg = f"Gamma FS manuel : kx={kx:+.4f}, ky={ky:+.4f} π/a"
        self._status(msg)
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"Gamma FS manuel mémorisé ({kx:+.4f}, {ky:+.4f})")
        try:
            self._fs_controls.lbl_info.setText(msg)
        except Exception:
            pass

    def _detect_fs_gamma(self):
        if FermiSurfaceCanvas is None or FSControlPanel is None:
            return
        try:
            params = self._fs_controls.params()
            res = self._fs_canvas.detect_gamma(self._raw_data, params)
            self._fs_controls.set_center(res["kx"], res["ky"])
            self._store_fs_center_reference(res["kx"], res["ky"], source="fs_auto")
            self._draw_fs_tab()
            msg = (f"Gamma FS détecté : kx={res['kx']:+.4f}, ky={res['ky']:+.4f} π/a "
                   f"| {len(res.get('gamma_kx_list', []))} coupes kx, "
                   f"{len(res.get('gamma_ky_list', []))} coupes ky")
            self._status(msg)
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"Gamma FS détecté ({res['kx']:+.4f}, {res['ky']:+.4f})")
            try:
                self._fs_controls.lbl_info.setText(msg)
            except Exception:
                pass
        except Exception as exc:
            QMessageBox.warning(self._parent, "Détection Gamma", str(exc))

    def _stored_gamma_reference(self) -> dict:
        return _gamma_stored_reference(self._session.gamma_reference)

    def _gamma_reference_to_bm_center(self, ref: dict) -> tuple[float, float]:
        """Wrapper UI : délègue à `arpes_gamma.gamma_reference_to_bm_center`."""
        if self._raw_data is None:
            return np.nan, 0.0
        meta = self._raw_data.get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_bm = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        return _gamma_ref_to_bm_center(
            ref,
            bm_metadata=meta,
            bm_hv=self._raw_data.get("hv"),
            work_func=float(self._params.sp_phi.value()),
            bm_azi=azi_bm,
            on_warn=self._status,
        )

    def _center_current_bm_axis_on_gamma(self, gamma_bm: float, ref: dict | None = None) -> bool:
        """Wrapper UI : délègue à `arpes_gamma.apply_bm_gamma_axis_shift` puis
        synchronise la sélection MDC `_sel_k`."""
        if self._raw_data is None:
            return False
        applied = _gamma_apply_bm_axis_shift(self._raw_data, gamma_bm, ref=ref)
        if applied and hasattr(self, "_sel_k"):
            self._sel_k = float(self._sel_k - float(gamma_bm))
        return applied

    def _apply_stored_gamma_to_current_file(self, *, save_entry: bool = False):
        if self._raw_data is None:
            return

        meta = self._raw_data.get("metadata", {}) or {}
        is_fs = meta.get("fs_data") is not None

        if is_fs and FSControlPanel is not None and hasattr(self, "_fs_controls"):
            entry = self._current_entry()
            if meta.get("angle_offsets_applied"):
                self._fs_controls.set_center(0.0, 0.0)
                if save_entry and entry is not None:
                    entry.fs_center_kx = 0.0
                    entry.fs_center_ky = 0.0
                    self._session.save()
                return
            if entry is not None and entry.fs_center_kx is not None and entry.fs_center_ky is not None:
                self._fs_controls.set_center(float(entry.fs_center_kx), float(entry.fs_center_ky))
                return

        ref = self._stored_gamma_reference()
        if not ref:
            return

        if is_fs and FSControlPanel is not None and hasattr(self, "_fs_controls"):
            entry = self._current_entry()
            if self._same_path(ref.get("path"), self._raw_data.get("path")):
                kx_fs = float(ref["kx"])
                ky_fs = float(ref.get("ky", 0.0) or 0.0)
            else:
                azi_fs = entry.meta.azi if (entry and entry.meta.azi is not None) else None
                kx_fs, ky_fs = self._project_gamma_by_azi(
                    ref, azi_fs, warn_label="Γ référence → FS"
                )
                if not np.isfinite(kx_fs) or not np.isfinite(ky_fs):
                    return
            self._fs_controls.set_center(float(kx_fs), float(ky_fs))
            if save_entry and entry is not None:
                entry.fs_center_kx = float(kx_fs)
                entry.fs_center_ky = float(ky_fs)
                self._session.save()
            if not self._same_path(ref.get("path"), self._raw_data.get("path")):
                self._status(f"Γ FS propagé par azimut : kx={kx_fs:+.4f}, ky={ky_fs:+.4f} π/a")
            return

        if meta.get("angle_offsets_applied"):
            self._params.sp_cx.setValue(0.0)
            if save_entry and self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = 0.0
                self._session.save()
            self._status("Γ mémorisé appliqué par offset angulaire : centre fit=0")
            return

        gamma_bm, correction = self._gamma_reference_to_bm_center(ref)
        if not np.isfinite(gamma_bm):
            return

        axis_centered = self._center_current_bm_axis_on_gamma(float(gamma_bm), ref)
        self._params.sp_cx.setValue(0.0 if axis_centered else float(gamma_bm))
        if save_entry and self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fit_params.center_init = 0.0 if axis_centered else float(gamma_bm)
            self._session.save()
        axis_msg = "axe k recentré" if axis_centered else "centre fit seul"
        self._status(
            f"Γ mémorisé appliqué : {gamma_bm:+.4f} π/a  correction={correction:+.4f}  |  {axis_msg}"
        )

    def _estimate_gamma_bm(self):
        try:
            import arpes.ui.widgets.plots as ap
        except Exception:
            self._status("Attention: arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        try:
            res = ap.estimate_gamma_bm_mdc(
                data, kpar, ev,
                ev_range=(self._params.sp_evs.value(), self._params.sp_eve.value()),
                k_range=(self._params.sp_kmin.value(), self._params.sp_kmax.value()),
                center_guess=self._params.sp_cx.value(),
                center_window=max(self._params.sp_xg.value() * 2.0, 0.25),
                smooth_sigma=self._params.sp_sfd.value(),
                verbose=False,
            )
            gamma = float(res["gamma"])
            if not np.isfinite(gamma):
                QMessageBox.warning(
                    self, "Auto Γ BM",
                    "Impossible d'estimer Γ : pas assez de paires MDC valides. "
                    "Ajuste la plage d'énergie, k_min/k_max ou centre_init."
                )
                return
            self._params.sp_cx.setValue(gamma)
            entry_now = self._current_entry()
            azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
            meta_now = self._raw_data.get("metadata", {}) or {}
            self._session.gamma_reference = _gamma_build_reference(
                kx=gamma, ky=0.0,
                metadata=meta_now,
                hv=self._raw_data.get("hv"),
                path=self._raw_data.get("path"),
                azi=azi_ref,
                source="bm",
                direction=(entry_now.meta.direction if entry_now else None),
            )
            if not meta_now.get("angle_offsets_applied") and not meta_now.get("bm_gamma_axis_centered"):
                offsets = self._angle_offsets_from_k_center(
                    float(gamma), 0.0,
                    hv=self._raw_data.get("hv"),
                    source="bm_auto",
                    ref_path=self._raw_data.get("path"),
                    azi=azi_ref,
                )
                if offsets:
                    self._session.angle_offsets = offsets
            if self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = float(gamma)
            self._session.save()
            self._params.lbl_res.setText(
                f"Γ BM = {gamma:+.4f} π/a\n"
                f"n={res['n']}  MAD={res['mad']:.4f}"
            )
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"Auto Γ BM appliqué ({gamma:+.4f} π/a)")
            self._draw_current_view()
            self._status(f"Γ BM estimé : {gamma:+.4f} π/a  n={res['n']}  MAD={res['mad']:.4f}")
        except Exception as exc:
            QMessageBox.warning(self._parent, "Auto Γ BM", str(exc))
            self._status(f"Attention: Auto Γ BM : {exc}")

    def _apply_gamma_reference_to_bm(self):
        ref = self._stored_gamma_reference()
        if not ref:
            QMessageBox.warning(self._parent, "Γ FS → BM", "Aucun Γ de référence. Va sur l'onglet FS et clique d'abord sur Détecter Γ FS.")
            return
        if self._raw_data is None:
            return
        meta = self._raw_data.get("metadata", {}) or {}
        if meta.get("angle_offsets_applied"):
            self._params.sp_cx.setValue(0.0)
            if self._current_path:
                entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
                entry.fit_params.center_init = 0.0
                self._session.save()
            self._params.lbl_res.setText("Γ déjà appliqué par offset angulaire loader")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done("Γ FS appliqué par offset loader")
            self._draw_current_view()
            self._status("Γ FS appliqué : offset angulaire loader déjà actif")
            return
        gamma_bm, correction = self._gamma_reference_to_bm_center(ref)
        if not np.isfinite(gamma_bm):
            QMessageBox.warning(self._parent, "Γ FS → BM", "La référence Γ stockée est invalide.")
            return
        angular_applied = bool(meta.get("angle_offsets_applied"))
        axis_centered = False if angular_applied else self._center_current_bm_axis_on_gamma(float(gamma_bm), ref)
        self._params.sp_cx.setValue(0.0 if (axis_centered or angular_applied) else gamma_bm)
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fit_params.center_init = 0.0 if (axis_centered or angular_applied) else float(gamma_bm)
            self._session.save()
        mode_msg = "offset angulaire loader" if angular_applied else ("axe k recentré" if axis_centered else "centre fit seul")
        self._params.lbl_res.setText(
            f"Γ FS→BM = {gamma_bm:+.4f} π/a\n"
            f"corr polar={correction:+.4f}\n"
            f"{mode_msg}"
        )
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"Γ FS appliqué à la BM ({gamma_bm:+.4f} π/a)")
        self._update_display_data()
        self._draw_current_view()
        self._status(
            f"Γ FS appliqué à la BM : {gamma_bm:+.4f} π/a  correction={correction:+.4f}"
            f"  |  {mode_msg}"
        )
