"""Controller UI exécutant fits MDC, calibrations EF et propagation params."""
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
from arpes.physics.fit import (
    MdcFitter,
    compute_fit_params_hash,
    ensemble_fit,
    imaginary_self_energy,
)
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

    def _current_fit_params_hash(self, entry=None, *, fp=None) -> str:
        """Hash de l'état UI/données courant influant le fit MDC.

        Comparé au hash stocké dans fit_result : si différent → fit
        STALE (params modifiés depuis fit, résultat trompeur).

        ``fp`` peut être passé explicitement pour utiliser un FitParams
        précis (multi-zone) au lieu de relire l'UI (où seules les
        paires 0 sont synchronisées via load_fit_params).
        """
        p = self._parent
        if fp is None:
            fp = self._params.get_fit_params()
        if entry is None and getattr(p, "_current_path", None):
            key = self._session.key_for_path(p._current_path)
            entry = self._session.get_or_create(key)
        hv = None
        try:
            hv = float(p._raw_data["hv"]) if p._raw_data else None
        except Exception:
            hv = None
        return compute_fit_params_hash(
            fp,
            ef_offset=self._params.sp_ef.value(),
            view_mode=p._cmb_view.currentText(),
            hv=hv,
            bm_distortion=getattr(entry, "bm_distortion", None) if entry else None,
            grid_correction=getattr(entry, "grid_correction", None) if entry else None,
            ef_correction=getattr(entry, "ef_correction", None) if entry else None,
        )

    def _update_mdc_tab_label(self, fr: dict | None) -> None:
        """G : titre dynamique de l'onglet Fit MDC = état du fit courant."""
        p = self._parent
        tabs = getattr(p, "_mdc_fit_tabs", None)
        if tabs is None or tabs.count() < 1:
            return
        if not fr:
            tabs.setTabText(0, "Fit MDC")
            return
        # numpy : `or []` planterait sur ndarray non vide (bool ambigu)
        _e = fr.get("e_fitted")
        n_e = 0 if _e is None else len(_e)
        stale = False
        try:
            stale = bool(fr.get("params_hash")
                         and fr["params_hash"] != self._current_fit_params_hash())
        except Exception:
            stale = False
        marker = "•" if stale else "✓"
        suffix = " (stale)" if stale else ""
        tabs.setTabText(0, f"Fit MDC {marker} {n_e}{suffix}")

    def _redraw_all_fit_views(self) -> None:
        """Rafraîchit BM + MDC map + MDC EDC après fit/clear, quel que soit
        l'onglet actif. Corrige : kF n'apparaissait que sur l'onglet courant
        (BM si fit lancé depuis BM), forçait switch+revenir pour voir MDC."""
        p = self._parent
        for name in ("_draw_bm", "_draw_mdc_energy_map", "_draw_mdc_edc",
                     "_draw_mdc_waterfall"):
            fn = getattr(p, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ----------------------------------------------------------------- helpers
    def _get_work_data(self):
        p = self._parent
        if p._raw_data is None:
            return None, None, None
        d = p._raw_data
        view_mode = p._cmb_view.currentText()
        entry = p._current_entry() if hasattr(p, "_current_entry") else None
        from arpes.physics.distortion import is_distortion_active
        from arpes.physics.plot_compute import compute_bandmap_display
        from arpes.physics.norm import remove_grid_artifact as remove_detector_grid_artifact
        grid_cfg = entry.grid_correction if entry and (entry.grid_correction or {}).get("enabled") else None
        bm_dist = getattr(entry, "bm_distortion", None) if entry else None
        dist_cfg = bm_dist if (bm_dist and is_distortion_active(bm_dist)) else None
        if grid_cfg or dist_cfg:
            result = compute_bandmap_display(
                d,
                mode="Raw",
                edc_norm_enabled=False,
                grid_correction=grid_cfg,
                grid_artifact_fn=remove_detector_grid_artifact,
                distortion_correction=dist_cfg,
            )
            data_work = result.data
            kpar_work = np.asarray(result.kpar) if result.kpar is not None else d["kpar"]
            ev_work = np.asarray(result.ev) if result.ev is not None else d["ev_arr"]
            if view_mode == "EDCnorm":
                data_work = apply_edcnorm(data_work)
            return data_work, kpar_work, ev_work
        if view_mode == "EDCnorm":
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

    def _fit_ensemble(self) -> None:
        """I1: refit N fois (perturbe initiaux), agrège kF/Γ médians + σ.

        1 paire = 1 bande (modèle Lorentzien symétrique). L'ensemble
        donne σ statistique fiable (MAD-filtré). Plus lent (× N).
        """
        p = self._parent
        if p.ap is None:
            self._status("Attention: arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        fp = self._params.get_fit_params()
        n = int(getattr(self._params, "sp_ensemble_n", None).value()
                if hasattr(self._params, "sp_ensemble_n") else 30)
        jitter = float(getattr(self._params, "sp_ensemble_jitter", None).value()
                       if hasattr(self._params, "sp_ensemble_jitter") else 0.10)
        self._status(f"Fit ensemble en cours (N={n}, jitter={jitter*100:.0f}%) ...")
        QApplication.processEvents()
        try:
            controller = MdcFitter(p.ap)
            ens = ensemble_fit(
                controller, data, kpar, ev, fp,
                n_runs=n, jitter_pct=jitter,
                resolution_source=getattr(self._params,
                                          "_resolution_source_detail", ""),
            )
            n_ok = int(ens.get("n_ok") or 0)
            if n_ok == 0:
                self._status("Attention: ensemble fit — aucune run convergée.")
                return
            # Compose fit_result final : médianes → kF/Γ, σ dans ensemble.
            fr = {
                "e_fitted": ens["e_fitted"],
                "kF_minus": ens["kF_minus_med"],
                "kF_plus": ens["kF_plus_med"],
                "gamma_corrige": ens["gamma_med"],
                "gamma_brut": ens["gamma_med"],  # référence (pas re-correction)
                "ensemble": ens,
            }
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
                fr["params_hash"] = self._current_fit_params_hash(entry)
                from arpes.physics.distortion import is_distortion_active
                fr["distorted"] = bool(
                    entry.bm_distortion
                    and is_distortion_active(entry.bm_distortion)
                )
                fr["grid_active"] = bool((entry.grid_correction or {}).get("enabled"))
                self._session.set_fit_result(name, fr)
                p._browser.refresh_item(name)
                self._refresh_helper_buttons()
            crystal_a = 0.0
            try:
                crystal_a = float(self._params.sp_crystal_a.value())
            except Exception:
                pass
            summary = controller.summarize(fr, crystal_a=crystal_a)
            self._params.lbl_res.setText(
                summary.label_text +
                f"\nEnsemble: {n_ok}/{n} runs, jitter={jitter*100:.0f}%"
            )
            self._params.lbl_res.setToolTip(
                "Résolution instrumentale domine, fit non fiable"
                if summary.resolution_dominates else
                "kF/Γ = médianes MAD-filtrées; σ dans ensemble."
            )
            try:
                threshold = float(self._params.sp_chi2_threshold.value())
            except Exception:
                threshold = 5.0
            if hasattr(self._params, "update_fit_quality"):
                self._params.update_fit_quality(
                    fr, threshold,
                    current_hash=self._current_fit_params_hash(),
                )
            self._update_mdc_tab_label(fr)
            self._redraw_all_fit_views()
            self._status(
                f"Fit ensemble OK — {n_ok}/{n} runs convergées."
            )
        except Exception as e:
            self._status(f"Attention: Fit ensemble : {e}")
            traceback.print_exc()

    def _calculate_im_self_energy(self) -> None:
        """H: ouvre dialog Im Σ(E) depuis fit_result courant + a."""
        p = self._parent
        fr = getattr(p, "_fit_res", None)
        if not fr:
            self._status("Attention: faire un fit MDC avant Im Σ.")
            return
        try:
            a = float(self._params.sp_crystal_a.value())
        except Exception:
            a = 0.0
        if a <= 0:
            self._status("Attention: renseigne a cristal (Å) > 0 pour Im Σ.")
            return
        wm = str(fr.get("width_mode", "symmetric"))
        # Mode 'independent' → exposer γL/γR séparés + moyenne. Sinon mean seul.
        wants_sides = (wm == "independent"
                       and bool(fr.get("gamma_left_corrige"))
                       and bool(fr.get("gamma_right_corrige")))
        if wants_sides:
            payload = {
                "Moyenne": imaginary_self_energy(fr, a, pair_index=0,
                                                  side="mean"),
                "γL (kF-)": imaginary_self_energy(fr, a, pair_index=0,
                                                   side="left"),
                "γR (kF+)": imaginary_self_energy(fr, a, pair_index=0,
                                                   side="right"),
            }
            ref = payload["Moyenne"]
        else:
            payload = imaginary_self_energy(fr, a, pair_index=0)
            ref = payload
        if ref["energy"].size == 0:
            self._status("Attention: Im Σ indisponible (vF/Γ manquants).")
            return
        from arpes.ui.widgets.dialogs import ImagSelfEnergyDialog
        dlg = ImagSelfEnergyDialog(payload, parent=p)
        dlg.exec()
        med = float(np.nanmedian(ref["im_sigma"])) * 1000.0
        suffix = " (mean/γL/γR séparés)" if wants_sides else ""
        self._status(
            f"Im Σ med = {med:.1f} meV  |  vF = {ref['vF_eV_A']:.2f} eV·Å{suffix}"
        )

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
                # F: empreinte des params au moment du fit -> détection stale
                fr["params_hash"] = self._current_fit_params_hash(entry)
                # Tag the axis state used during the fit so the overlay can
                # refuse to plot when the user later toggles grid/distortion
                # off (fit_result kF lives in the *original* warped axis).
                from arpes.physics.distortion import is_distortion_active
                fr["distorted"] = bool(
                    entry.bm_distortion
                    and is_distortion_active(entry.bm_distortion)
                )
                fr["grid_active"] = bool((entry.grid_correction or {}).get("enabled"))
                self._session.set_fit_result(name, fr)
                # Mirror to active zone if one exists (multi-zone workflow).
                zctrl = getattr(p, "_fit_zones_ctrl", None)
                if zctrl is not None and entry.active_zone_id:
                    zctrl.store_result(entry.active_zone_id, fr)
                p._browser.refresh_item(name)
                self._refresh_helper_buttons()

            crystal_a = 0.0
            try:
                crystal_a = float(self._params.sp_crystal_a.value())
            except Exception:
                crystal_a = 0.0
            summary = controller.summarize(fr, crystal_a=crystal_a)
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
                self._params.update_fit_quality(
                    fr, threshold,
                    current_hash=self._current_fit_params_hash(),
                )
            self._update_mdc_tab_label(fr)
            self._redraw_all_fit_views()
            self._status(summary.status_text)
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done("fit complet terminé")
        except Exception as e:
            self._status(f"Attention: Fit complet : {e}")
            traceback.print_exc()

    def _clear_kf(self):
        from arpes.ui.controllers.fit_clear import clear_kf
        return clear_kf(self)

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

    # =================================================================
    # Multi-zone (Phase 1 sync; threadpool deferred)
    # =================================================================
    def _refresh_zones_strip(self) -> None:
        from arpes.ui.controllers.fit_zone_runner import refresh_zones_strip
        return refresh_zones_strip(self)

    def _on_zone_activated(self, zone_id: str) -> None:
        from arpes.ui.controllers.fit_zone_runner import on_zone_activated
        return on_zone_activated(self, zone_id)

    def _fit_run_all_zones(self) -> None:
        from arpes.ui.controllers.fit_zone_runner import fit_run_all_zones
        return fit_run_all_zones(self)
