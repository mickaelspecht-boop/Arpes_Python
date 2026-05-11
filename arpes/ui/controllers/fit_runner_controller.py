"""Controller UI exécutant les fits MDC + calibrations EF + propagation params.

Sort de ArpesExplorer :
- `_get_work_data` (donne données EDC-normalisées au fit)
- `_fit_guess` / `_fit_full` / `_clear_kf`
- `_ef_calibrate` / `_apply_ef_calibration_result` / `_apply_ef_reference_to_current`
- `_refresh_helper_buttons` / `_copy_params`
"""
from __future__ import annotations

import traceback
import warnings
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from arpes.physics.plot_compute import apply_edcnorm
from arpes.physics.ef_calibration import (
    ReferenceError as EFReferenceError,
    already_applied as ef_reference_already_applied,
    apply_reference_to_target as apply_ef_reference_to_target,
    compute_calibration_update as compute_ef_calibration_update,
)
from arpes.physics.fit import MdcFitter
from arpes.ui.widgets.dialogs import EFCalibrationDialog


class FitRunnerController:
    def __init__(self, parent):
        self._parent = parent

    @property
    def _params(self):
        return self._parent._params

    @property
    def _session(self):
        return self._parent._session

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ----------------------------------------------------------------- helpers
    def _get_work_data(self):
        p = self._parent
        if p._raw_data is None:
            return None, None, None
        d = p._raw_data
        if p._cmb_view.currentText() == "EDCnorm":
            p._update_display_data()
            if p._data_disp is not None and np.shape(p._data_disp) == np.shape(d["data"]):
                norm = p._data_disp
            else:
                norm = apply_edcnorm(d["data"])
        else:
            norm = d["data"]
        return norm, d["kpar"], d["ev_arr"]

    # -------------------------------------------------------------------- fit
    def _fit_guess(self):
        p = self._parent
        if p.ap is None:
            self._status("Attention: arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        fp = self._params.get_fit_params()

        ax = p._mdc_edc.axes[0]
        ax.cla(); ax.set_facecolor("#1a1a1a")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kF_init_list = [pp.get("kF_init", 0.30) for pp in (fp.pairs or [])]
                r = p.ap.debug_mdc_fit(
                    data, kpar, ev,
                    energy=p._sel_ev, n_pairs=fp.n_pairs,
                    smooth_fit=fp.smooth_fit, smooth_detect=fp.smooth_detect,
                    gamma_init=fp.gamma_init, gamma_max=fp.gamma_max,
                    kF_init=kF_init_list or None, center_init=fp.center_init,
                    xg_range=fp.xg_range, k_min=fp.k_min, k_max=fp.k_max,
                    k0_max=fp.k0_max, width_mode=fp.width_mode, ax=ax,
                )
            ax.set_title(f"Guess  E={p._sel_ev:.3f} eV", fontsize=8, color="w")
            ax.tick_params(colors="w", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555")
            try:
                leg = ax.get_legend()
                if leg:
                    leg.get_frame().set_facecolor("#333")
                    for t in leg.get_texts():
                        t.set_color("w")
            except Exception:
                pass

            if r["success"]:
                k0s = "  ".join(f"{v:.3f}" for v in r["k0"])
                gamma_vals = r["gamma"] if isinstance(r["gamma"], (list, tuple, np.ndarray)) else [r["gamma"]]
                gammas = "  ".join(f"{float(v):.4f}" for v in gamma_vals)
                self._params.lbl_res.setText(
                    f"OK  E={p._sel_ev:.3f} eV\n"
                    f"kF=[{k0s}] π/a\n"
                    f"γ=[{gammas}]  rms={r['residual']:.4f}\n"
                    f"xg={r['xg']:.4f} π/a")
                self._status(f"Guess OK  kF={k0s}  γ=[{gammas}]")
                if hasattr(self._params, "mark_action_done"):
                    self._params.mark_action_done(f"guess MDC OK à E={p._sel_ev:.3f} eV")
            else:
                self._params.lbl_res.setText("Fit échoué")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    ha="center", va="center", color="tomato", fontsize=8)
            traceback.print_exc()
        p._mdc_edc.fig.tight_layout(pad=0.5)
        p._mdc_edc.redraw()

    def _fit_full(self):
        p = self._parent
        if p.ap is None:
            self._status("Attention: arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        fp = self._params.get_fit_params()

        self._status("Fit complet en cours ...")
        QApplication.processEvents()
        try:
            controller = MdcFitter(p.ap)
            fr = controller.run_full_fit(
                data, kpar, ev, fp,
                resolution_source=getattr(self._params, "_resolution_source_detail", ""),
            )
            p._fit_res = fr

            if p._current_path:
                name = self._session.key_for_path(p._current_path)
                entry = self._session.get_or_create(name)
                controller.update_entry_after_fit(
                    entry, fp,
                    ef_offset=self._params.sp_ef.value(),
                    edcnorm=p._cmb_view.currentText() == "EDCnorm",
                    view_mode=p._cmb_view.currentText(),
                    hv=p._raw_data["hv"],
                )
                self._session.set_fit_result(name, fr)
                p._browser.refresh_item(name)
                self._refresh_helper_buttons()

            summary = controller.summarize(fr)
            self._params.lbl_res.setText(summary.label_text)
            self._params.lbl_res.setToolTip(
                "Résolution instrumentale domine, fit non fiable"
                if summary.resolution_dominates else ""
            )
            try:
                threshold = float(self._params.sp_chi2_threshold.value())
            except Exception:
                threshold = 5.0
            if hasattr(self._params, "update_fit_quality"):
                self._params.update_fit_quality(fr, threshold)
            p._draw_current_view()
            self._status(summary.status_text)
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done("fit complet terminé")
        except Exception as e:
            self._status(f"Attention: Fit complet : {e}")
            traceback.print_exc()

    def _clear_kf(self):
        p = self._parent
        p._fit_res = None
        if p._current_path:
            key = self._session.key_for_path(p._current_path)
            entry = self._session.get_or_create(key)
            entry.fit_result = None
            entry.annotations = {}
            self._session.save()
            if hasattr(p, "_browser"):
                p._browser.refresh_item(key)
        p._fit_selected = []
        if hasattr(self._params, "update_fit_quality"):
            self._params.update_fit_quality(None, 5.0)
        p._draw_current_view()
        self._params.lbl_res.setText("kF effacé")
        results = getattr(p, "_results", None)
        if results is not None and hasattr(results, "refresh"):
            results.refresh()

    # --------------------------------------------------------- EF calibration
    def _ef_calibrate(self):
        p = self._parent
        if p._raw_data is None:
            self._status("Attention: Aucune donnée chargée")
            return
        d = p._raw_data
        entry_now = p._current_entry()
        T_md = (d.get("metadata", {}) or {}).get("temperature")
        try:
            T_md = float(T_md) if T_md is not None else None
        except (TypeError, ValueError):
            T_md = None
        if T_md and np.isfinite(T_md) and T_md > 0:
            T_init = T_md
        elif entry_now and entry_now.meta.temperature and entry_now.meta.temperature > 0:
            T_init = float(entry_now.meta.temperature)
        else:
            T_init = 28.0

        try:
            dlg = EFCalibrationDialog(
                p,
                data=d["data"], kpar=d["kpar"], ev_arr=d["ev_arr"],
                T_init=T_init, half_width_init=0.15,
                source_name=Path(p._current_path).name if p._current_path else "",
                current_offset=self._params.sp_ef.value(),
                metadata=d.get("metadata", {}) or {},
            )
            if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_payload:
                return
            self._apply_ef_calibration_result(dlg.result_payload)
        except Exception as e:
            self._status(f"Attention: Calibration EF : {e}")
            traceback.print_exc()

    def _apply_ef_calibration_result(self, payload: dict):
        p = self._parent
        if not p._current_path:
            return
        key = self._session.key_for_path(p._current_path)
        entry = self._session.get_or_create(key)

        update = compute_ef_calibration_update(
            payload,
            current_ef_offset=float(self._params.sp_ef.value()),
            source_meta=(p._raw_data or {}).get("metadata") or {},
            source_path=str(p._current_path),
        )
        entry.ef_offset = update.new_ef_offset
        entry.ef_correction = update.ef_correction
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(update.new_ef_offset)
        self._params.sp_ef.blockSignals(False)
        msg = update.msg

        if payload.get("save_as_reference"):
            self._session.ef_reference = update.ref_payload
            msg += "  |  référence dossier sauvegardée"

        self._session.save()
        p._load_file(p._current_path)
        self._refresh_helper_buttons()
        self._status(msg)
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done("calibration EF appliquée")

    def _apply_ef_reference_to_current(self):
        p = self._parent
        ref = self._session.ef_reference or {}
        if not ref or not p._current_path:
            self._status("Attention: Aucune référence EF en session - calibrer un Au d'abord")
            return
        key = self._session.key_for_path(p._current_path)
        entry = self._session.get_or_create(key)

        if ef_reference_already_applied(entry.ef_correction):
            ans = QMessageBox.question(
                p,
                "Référence EF déjà appliquée",
                "Une référence EF est déjà appliquée à ce fichier. "
                "L'appliquer à nouveau cumulerait les décalages et serait probablement faux.\n\n"
                "Continuer quand même ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                self._status("Application de la référence EF annulée")
                return

        ref_path = ref.get("source_file", "")
        ref_name = Path(ref_path).name if ref_path else "?"

        try:
            app = apply_ef_reference_to_target(
                ref,
                current_ef_offset=float(self._params.sp_ef.value()),
                target_meta=(p._raw_data or {}).get("metadata") or {},
                ref_path_str=ref_name,
            )
        except EFReferenceError:
            self._status("Attention: Référence EF mal formée")
            return
        entry.ef_offset = app.new_ef_offset
        entry.ef_correction = app.ef_correction
        self._params.sp_ef.blockSignals(True)
        self._params.sp_ef.setValue(app.new_ef_offset)
        self._params.sp_ef.blockSignals(False)
        self._session.save()
        p._load_file(p._current_path)
        self._status(app.msg)
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done("référence EF appliquée")

    # ----------------------------------------------------------- params utils
    def _refresh_helper_buttons(self):
        p = self._parent
        self._params.update_ef_reference_button(self._session.ef_reference or None)
        if not p._current_path:
            self._params.update_copy_params_button(0)
            return
        cur_key = self._session.key_for_path(p._current_path)
        n = sum(
            1 for name, entry in self._session.files.items()
            if entry.fit_result is None and name != cur_key
        )
        self._params.update_copy_params_button(n)

    def _copy_params(self):
        p = self._parent
        if not p._current_path:
            return
        fp = self._params.get_fit_params()
        cur_key = self._session.key_for_path(p._current_path)
        targets: list[str] = []
        for name, entry in self._session.files.items():
            if entry.fit_result is None and name != cur_key:
                entry.fit_params = fp
                targets.append(name)
        self._session.save()
        n = len(targets)
        if n == 0:
            self._status("Aucun fichier cible - tous les autres sont déjà fittés")
        elif n <= 3:
            self._status(f"Params copiés vers {n} fichier(s) : {', '.join(targets)}")
        else:
            preview = ", ".join(targets[:2])
            self._status(f"Params copiés vers {n} fichiers : {preview}, ... (+{n-2})")
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"paramètres propagés vers {n} fichier(s)")
        self._refresh_helper_buttons()
