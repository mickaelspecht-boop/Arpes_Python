"""Panneau paramètres de fit + sélecteur de paires (FitParamsPanel).

La construction des sections (Énergie, EF/Chargement, Utilitaires, DFT/Théorie,
Fit MDC) est déléguée à des modules `params_*` pour rester sous le plafond
700 LOC. Les helpers métier (update_*, get_fit_params, load_fit_params, gestion
des paires, gestion de provenance hν/résolution) restent ici.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from arpes.core.session import FitParams
from arpes.ui.widgets._qt_helpers import PAIR_COLORS  # noqa: F401  re-export


class ClickablePairLabel(QLabel):
    """Label cliquable pour naviguer entre les paires de Lorentziennes.
    Clic gauche → paire suivante.  Clic droit → paire précédente."""
    pair_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._current = 0
        self._n = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "background:#3a3a4a; color:#cde; font-weight:bold;"
            " padding:4px 8px; border-radius:3px; border:1px solid #556;"
        )
        self._update()

    def setup(self, n: int, current: int = 0):
        self._n = max(1, n)
        self._current = max(0, min(current, self._n - 1))
        self._update()

    @property
    def current(self) -> int:
        return self._current

    def _update(self):
        if self._n == 1:
            self.setText("Paire 1 / 1")
        else:
            self.setText(f"<  Paire {self._current + 1} / {self._n}  >")

    def mousePressEvent(self, event):
        if self._n < 2:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = (self._current + 1) % self._n
        elif event.button() == Qt.MouseButton.RightButton:
            self._current = (self._current - 1) % self._n
        else:
            super().mousePressEvent(event)
            return
        self._update()
        self.pair_changed.emit(self._current)


class FitParamsPanel(QScrollArea):
    params_changed = pyqtSignal()
    fit_only_changed = pyqtSignal()
    guess_requested = pyqtSignal()
    full_fit_requested = pyqtSignal()
    clear_kf_requested = pyqtSignal()
    copy_params_requested = pyqtSignal()
    ef_calib_requested = pyqtSignal()
    ef_apply_reference_requested = pyqtSignal()
    logbook_requested = pyqtSignal()
    gamma_bm_requested = pyqtSignal()
    gamma_ref_requested = pyqtSignal()
    grid_requested = pyqtSignal()
    grid_reset_requested = pyqtSignal()
    distortion_apply_requested = pyqtSignal()
    distortion_reset_requested = pyqtSignal()
    distortion_auto_requested = pyqtSignal()
    distortion_import_calib_requested = pyqtSignal()
    distortion_preview_changed = pyqtSignal()
    fit_roi_requested = pyqtSignal(bool)
    fit_roi_reset_requested = pyqtSignal()
    fit_undo_requested = pyqtSignal()
    theory_import_requested = pyqtSignal()
    theory_refresh_requested = pyqtSignal()
    theory_local_import_requested = pyqtSignal()
    theory_clear_requested = pyqtSignal()
    theory_overlay_changed = pyqtSignal()
    theory_compare_requested = pyqtSignal()
    self_energy_requested = pyqtSignal()
    theory_search_requested = pyqtSignal()
    theory_band_picker_requested = pyqtSignal()
    theory_align_requested = pyqtSignal()
    theory_efalign_requested = pyqtSignal()
    crystal_a_changed = pyqtSignal()
    file_tags_changed = pyqtSignal()
    fit_section_toggled = pyqtSignal(str, bool)
    fit_preset_changed = pyqtSignal(str)
    gamma_center_preview = pyqtSignal(float)
    batch_fit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        w = QWidget()
        self._lay = QVBoxLayout(w)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self.setWidget(w)
        self._pair_params: list[dict] = [{"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}]
        self._current_pair: int = 0
        self._resolution_source_lock = False
        self._resolution_source = "default"
        self._resolution_source_detail = "defaut"
        self._fit_sections: dict = {}
        self._theory_overlay_timer = QTimer(self)
        self._theory_overlay_timer.setSingleShot(True)
        self._theory_overlay_timer.setInterval(120)
        self._theory_overlay_timer.timeout.connect(self.theory_overlay_changed.emit)
        self._build()

    def _build(self):
        # Imports locaux pour éviter cycles statiques entre params.py et
        # params_fit.py (qui réimporte ClickablePairLabel).
        from arpes.ui.widgets.params_ef import (
            build_ef_section,
            build_energy_section,
        )
        from arpes.ui.widgets.params_fit import build_fit_controls
        from arpes.ui.widgets.params_utilities import build_utilities_section

        lay = self._lay
        build_energy_section(self, lay)
        build_ef_section(self, lay)
        build_utilities_section(self, lay)
        build_fit_controls(self, lay)
        lay.addStretch()

    # ── accès params ──────────────────────────────────────────────────────────
    def update_ef_reference_button(self, ref: dict | None):
        """Met à jour le label/état du bouton EF réf selon la session."""
        if not ref:
            self.btn_ef_ref.setText("Aucune réf EF")
            self.btn_ef_ref.setEnabled(False)
            self.btn_ef_ref.setToolTip(
                "Aucune référence EF enregistrée dans cette session.\n"
                "Pour en créer une : 'Calibrer EF auto' sur un scan Au, "
                "puis cocher 'Enregistrer comme référence' dans le dialog."
            )
            return
        mode = ref.get("mode", "?")
        src_path = ref.get("source_file", "")
        src_name = Path(src_path).name if src_path else "(source inconnue)"
        if mode == "scalar":
            shift_meV = float(ref.get("ef_shift", 0.0)) * 1000.0
            label = f"Appliquer EF réf ({shift_meV:+.1f} meV)"
        elif mode == "poly":
            n_valid = int(ref.get("n_valid", 0))
            fwhm = float(ref.get("fwhm_res", 0.0)) * 1000.0
            label = f"Appliquer EF réf poly (n={n_valid})"
        else:
            label = "Appliquer EF réf"
        self.btn_ef_ref.setText(label)
        self.btn_ef_ref.setEnabled(True)
        self.btn_ef_ref.setToolTip(
            f"Référence EF enregistrée :\n"
            f"  mode = {mode}\n"
            f"  source = {src_path or '?'}\n"
            f"Applique cette correction au fichier courant."
        )

    def update_hv_source(self, source: str | None):
        """Affiche la provenance de hν : 'file', 'logbook', 'manual', None."""
        labels = {"file": "Fichier", "logbook": "Logbook", "manual": "Manuel", "session": "Session"}
        self.lbl_hv_src.setText(labels.get(source or "", "Inconnu"))

    def _mark_hv_manual_if_user_edit(self):
        if not getattr(self, "_hv_source_lock", False):
            self.update_hv_source("manual")

    def set_hv_value_with_source(self, value: float, source: str):
        """Set la spinbox hν sans déclencher le marquage 'manuel'."""
        self._hv_source_lock = True
        try:
            self.sp_hv.blockSignals(True)
            self.sp_hv.setValue(float(value))
            self.sp_hv.blockSignals(False)
            self.update_hv_source(source)
        finally:
            self._hv_source_lock = False

    def set_file_tags(self, tags: list[str]):
        if not hasattr(self, "txt_file_tags"):
            return
        self.txt_file_tags.blockSignals(True)
        self.txt_file_tags.setText(", ".join(tags or []))
        self.txt_file_tags.blockSignals(False)

    def file_tags_text(self) -> str:
        if not hasattr(self, "txt_file_tags"):
            return ""
        return self.txt_file_tags.text().strip()

    def update_tag_completions(self, tags: list[str]):
        if hasattr(self, "_tag_completer_model"):
            self._tag_completer_model.setStringList(list(tags or []))

    def update_resolution_source(self, source: str | None):
        """Affiche la provenance de la resolution : 'estimated', 'manual', 'default'."""
        self._resolution_source = source or "default"
        self._resolution_source_detail = self._resolution_source
        label = {"estimated": "Estimée", "manual": "Manuelle", "default": "Défaut"}.get(self._resolution_source, "Défaut")
        self.lbl_dE_src.setText(label)
        self.lbl_dk_src.setText(label)

    def mark_action_done(self, text: str):
        self.lbl_action.setText(f"Dernière action : {text}")

    def _mark_resolution_manual_if_user_edit(self):
        if not getattr(self, "_resolution_source_lock", False):
            self.update_resolution_source("manual")
            self._resolution_source_detail = "manual"

    def set_resolution_with_source(self, dE_meV: float, dk_inv_a: float, source: str, detail: str | None = None):
        """Set les spinboxes resolution sans déclencher le marquage manuel."""
        self._resolution_source_lock = True
        try:
            for sp, value in ((self.sp_dE_meV, dE_meV), (self.sp_dk_inv_a, dk_inv_a)):
                sp.blockSignals(True)
                sp.setValue(float(value))
                sp.blockSignals(False)
            self.update_resolution_source(source)
            self._resolution_source_detail = detail or source
        finally:
            self._resolution_source_lock = False

    def update_copy_params_button(self, n_targets: int):
        """Met à jour le label/état du bouton 'Propager fit params'."""
        if n_targets <= 0:
            self.btn_copy.setText("Propager fit params (0 cible)")
            self.btn_copy.setEnabled(False)
            self.btn_copy.setToolTip(
                "Aucun fichier non-fitté dans le dossier (hors fichier courant).\n"
                "Tous les autres ont déjà un fit_result enregistré : ils ne seront pas écrasés."
            )
        else:
            self.btn_copy.setText(f"Propager fit params ({n_targets} cible{'s' if n_targets > 1 else ''})")
            self.btn_copy.setEnabled(True)
            self.btn_copy.setToolTip(
                f"Copie les paramètres de fit MDC actuels vers les {n_targets} "
                f"fichier(s) du dossier qui n'ont pas encore été fittés.\n"
                f"Les fichiers déjà fittés ne sont jamais écrasés."
            )

    def get_fit_params(self) -> FitParams:
        self._save_pair()
        p0 = self._pair_params[0] if self._pair_params else {}
        return FitParams(
            n_pairs=self.sp_np.value(),
            ev_start=self.sp_evs.value(),
            ev_end=self.sp_eve.value(),
            k_min=self.sp_kmin.value(),
            k_max=self.sp_kmax.value(),
            smooth_fit=self.sp_sff.value(),
            smooth_detect=self.sp_sfd.value(),
            gamma_init=p0.get("gamma_init", 0.08),
            gamma_max=p0.get("gamma_max", 0.30),
            xg_range=self.sp_xg.value(),
            center_init=self.sp_cx.value(),
            k0_max=None if self.chk_k0a.isChecked() else self.sp_k0m.value(),
            width_mode=self.cmb_wm.currentText(),
            min_amplitude=self.sp_ma.value(),
            max_jump=self.sp_mj.value(),
            scan_direction=self.cmb_sd.currentText(),
            dE_meV=self.sp_dE_meV.value(),
            dk_inv_a=self.sp_dk_inv_a.value(),
            pairs=[dict(p) for p in self._pair_params],
        )

    def set_fit_controls_visible(self, visible: bool):
        self._fit_controls_widget.setVisible(visible)

    def set_utilities_visible(self, visible: bool):
        self._utils_widget.setVisible(visible)

    def set_fit_roi_active(self, active: bool):
        self.btn_fit_roi.blockSignals(True)
        self.btn_fit_roi.setChecked(bool(active))
        self.btn_fit_roi.blockSignals(False)

    def set_fit_undo_enabled(self, enabled: bool):
        self.btn_fit_undo.setEnabled(bool(enabled))

    def set_context(self, context: str):
        """Adapte le panneau droit à l'onglet actif."""
        is_bm = context == "bm"
        is_mdc = context == "mdc"
        self._energy_widget.setVisible(is_bm)
        self._ef_widget.setVisible(is_bm)
        self._utils_widget.setVisible(is_bm)
        self._theory_widget.setVisible(is_bm)
        if hasattr(self, "_distortion_widget"):
            self._distortion_widget.setVisible(is_bm)
        if hasattr(self, "_utilities_toolbox"):
            self._utilities_toolbox.setVisible(is_bm)
        self._fit_controls_widget.setVisible(is_mdc)
        self._gamma_tools_widget.setVisible(False)
        if not is_mdc:
            self.set_waterfall_controls_visible(False)

    def set_waterfall_controls_visible(self, visible: bool):
        self._waterfall_controls_widget.setVisible(bool(visible))

    def bm_distortion_params(self) -> dict:
        from arpes.ui.widgets.params_distortion import bm_distortion_params as _f
        return _f(self)

    def set_bm_distortion_state(self, cfg: dict | None) -> None:
        from arpes.ui.widgets.params_distortion import set_bm_distortion_state as _f
        _f(self, cfg)

    def grid_params(self) -> dict:
        return {
            "enabled": True,
            "method": "display_fft2mask",
            "grid_period_px": None,
            "grid_freq": None,
            "notch_width": 2,
            "notch_sigma": 0.8,
            "strength": float(self.sp_grid_strength.value()),
            "fft2_center_radius": 18.0,
            "fft2_peak_sensitivity": 2.5,
            "fft2_plane": "display",
        }

    def theory_overlay_config(self) -> dict:
        return {
            "enabled": bool(self.chk_theory.isChecked()),
            "material_id": self.txt_theory_mpid.text().strip(),
            "segment": self.cmb_theory_segment.currentText().strip(),
            "path_convention": self.cmb_theory_convention.currentData() or "mp_bulk",
            "mu_shift": float(self.sp_theory_mu.value()),
            "z_scale": float(self.sp_theory_z.value()),
            "energy_shift": -float(self.sp_theory_mu.value()),
            "k_shift": float(self.sp_theory_dk.value()),
            "k_scale": float(self.sp_theory_kscale.value()),
            "alpha": float(self.sp_theory_alpha.value()),
            "max_bands": int(self.sp_theory_max.value()),
            "mirror_gamma": bool(self.chk_theory_mirror.isChecked()),
            "band_indices": self.txt_theory_bands.text().strip(),
            "ef_window": (
                float(self.sp_theory_efwin.value())
                if self.chk_theory_ef_only.isChecked() else 0.0
            ),
            "color_by_band": bool(self.chk_theory_color.isChecked()),
            "with_projections": bool(self.chk_theory_projections.isChecked()),
            "crystal_a": float(self.sp_crystal_a.value()),
        }

    def set_theory_overlay_state(self, overlay: dict):
        data = overlay.get("data") or {}
        config = overlay.get("config") or {}
        segments = list(overlay.get("segments") or [])
        self.chk_theory.blockSignals(True)
        self.chk_theory.setChecked(bool(overlay.get("enabled", False)))
        self.chk_theory.blockSignals(False)
        if data.get("material_id"):
            self.txt_theory_mpid.setText(str(data.get("material_id")))
        self.cmb_theory_segment.blockSignals(True)
        self.cmb_theory_segment.clear()
        self.cmb_theory_segment.addItem("")
        self.cmb_theory_segment.addItems(segments)
        if config.get("segment"):
            self.cmb_theory_segment.setCurrentText(str(config.get("segment")))
        self.cmb_theory_segment.blockSignals(False)
        self.cmb_theory_convention.blockSignals(True)
        wanted_convention = str(config.get("path_convention") or "mp_bulk")
        idx = self.cmb_theory_convention.findData(wanted_convention)
        self.cmb_theory_convention.setCurrentIndex(max(0, idx))
        self.cmb_theory_convention.blockSignals(False)
        for sp, key, default in (
            (self.sp_theory_mu, "mu_shift", -float(config.get("energy_shift", 0.0) or 0.0)),
            (self.sp_theory_z, "z_scale", 1.0),
            (self.sp_theory_dk, "k_shift", 0.0),
            (self.sp_theory_kscale, "k_scale", 1.0),
            (self.sp_theory_alpha, "alpha", 0.65),
            (self.sp_theory_max, "max_bands", 10),
        ):
            sp.blockSignals(True)
            sp.setValue(config.get(key, default))
            sp.blockSignals(False)
        self.chk_theory_mirror.blockSignals(True)
        self.chk_theory_mirror.setChecked(bool(config.get("mirror_gamma", False)))
        self.chk_theory_mirror.blockSignals(False)
        self.txt_theory_bands.blockSignals(True)
        self.txt_theory_bands.setText(str(config.get("band_indices", "") or ""))
        self.txt_theory_bands.blockSignals(False)
        win = float(config.get("ef_window", 0.0) or 0.0)
        self.sp_theory_efwin.blockSignals(True)
        self.sp_theory_efwin.setValue(win if win > 0 else 0.0)
        self.sp_theory_efwin.blockSignals(False)
        self.chk_theory_ef_only.blockSignals(True)
        self.chk_theory_ef_only.setChecked(win > 0.0)
        self.chk_theory_ef_only.blockSignals(False)
        self.chk_theory_color.blockSignals(True)
        self.chk_theory_color.setChecked(bool(config.get("color_by_band", True)))
        self.chk_theory_color.blockSignals(False)
        self.chk_theory_projections.blockSignals(True)
        self.chk_theory_projections.setChecked(bool(config.get("with_projections", False)))
        self.chk_theory_projections.blockSignals(False)
        self._populate_theory_band_table(
            data.get("band_meta") or [],
            data.get("band_character") or [],
            str(config.get("band_indices", "") or ""),
        )
        warning = overlay.get("warning") or ""
        mpid = data.get("material_id") or ""
        if mpid:
            source = str(data.get("source") or "")
            prefix = "DFT MP" if source == "materials_project" else "DFT locale"
            efermi = data.get("efermi")
            try:
                ef_txt = f" | DFT E_F={float(efermi):.3f} eV (déjà soustrait)"
            except (TypeError, ValueError):
                ef_txt = ""
            txt = f"{prefix} {mpid}.{ef_txt} Guide visuel, alignement manuel requis."
            cs = str(data.get("crystal_system") or "")
            if source == "materials_project":
                cs_txt = f" {cs}" if cs else ""
                txt += (
                    f"\nChemin = ZB BULK 3D{cs_txt} (Setyawan : Γ,X,P,N,Z…). "
                    "L'overlay FS utilise la ZB SURFACE 2D (Γ,X,M,Y,S) : "
                    "noms différents normaux (3D bulk ≠ 2D surface)."
                )
            if warning:
                txt += f" Attention: {warning}"
            comparison = overlay.get("comparison") or []
            if comparison:
                best = comparison[0]
                txt += (
                    f"\nComparaison: bande {best.get('band_index')} "
                    f"{best.get('branch')} paire {int(best.get('pair_index', 0)) + 1}, "
                    f"RMS={float(best.get('rms_e', 0.0)) * 1000:.0f} meV "
                    f"({int(best.get('n_points', 0))} pts)."
                )
        else:
            txt = "Guide visuel uniquement."
        self.lbl_theory_status.setText(txt)

    def _populate_theory_band_table(self, band_meta, band_character, band_indices):
        """Legacy hook kept for old callers; visual picker replaced table."""
        return

    def _on_theory_band_table_toggled(self, _item):
        """Legacy hook kept for sessions/tests from the old table UI."""
        return

    def _on_theory_bands_text_edited(self):
        """Champ texte legacy édité à la main."""
        self._schedule_theory_overlay_changed()

    def _schedule_theory_overlay_changed(self):
        """Coalesce live DFT UI edits into one overlay redraw.

        Spinboxes can emit several changes while the user types or holds an
        arrow key. The DFT overlay redraw is cosmetic, so a short debounce keeps
        the UI responsive without changing the final state that is saved.
        """
        self._theory_overlay_timer.start()

    def _emit_theory_overlay_changed_now(self):
        self._theory_overlay_timer.stop()
        self.theory_overlay_changed.emit()

    def load_fit_params(self, fp: FitParams):
        for sp, val in [
            (self.sp_evs, fp.ev_start), (self.sp_eve, fp.ev_end),
            (self.sp_kmin, fp.k_min), (self.sp_kmax, fp.k_max),
            (self.sp_sff, fp.smooth_fit), (self.sp_sfd, fp.smooth_detect),
            (self.sp_xg, fp.xg_range), (self.sp_cx, fp.center_init),
            (self.sp_ma, fp.min_amplitude), (self.sp_mj, fp.max_jump),
            (self.sp_dE_meV, getattr(fp, "dE_meV", 15.0)),
            (self.sp_dk_inv_a, getattr(fp, "dk_inv_a", 0.005)),
        ]:
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        if fp.k0_max is not None:
            self.chk_k0a.setChecked(False)
            self.sp_k0m.setValue(fp.k0_max)
        else:
            self.chk_k0a.setChecked(True)
        self.cmb_wm.setCurrentText(fp.width_mode)
        self.cmb_sd.setCurrentText(fp.scan_direction)

        n = fp.n_pairs
        raw = list(getattr(fp, "pairs", None) or [])
        if not raw:
            raw = [{"kF_init": 0.30, "gamma_init": fp.gamma_init, "gamma_max": fp.gamma_max}]
        while len(raw) < n:
            raw.append(dict(raw[-1]))
        self._pair_params = raw[:max(n, 1)]
        self._current_pair = 0
        self.sp_np.blockSignals(True)
        self.sp_np.setValue(n)
        self.sp_np.blockSignals(False)
        self._pair_lbl.setup(n, 0)
        self._load_pair(0)

    # ── gestion par paire ─────────────────────────────────────────────────────
    def _save_pair(self):
        i = self._current_pair
        if i < len(self._pair_params):
            self._pair_params[i] = {
                "kF_init": self.sp_kfi.value(),
                "gamma_init": self.sp_gi.value(),
                "gamma_max": self.sp_gm.value(),
            }

    def _load_pair(self, i: int):
        i = max(0, min(i, len(self._pair_params) - 1))
        p = self._pair_params[i]
        for sp, key, default in [
            (self.sp_kfi, "kF_init", 0.30),
            (self.sp_gi, "gamma_init", 0.08),
            (self.sp_gm, "gamma_max", 0.30),
        ]:
            sp.blockSignals(True)
            sp.setValue(p.get(key, default))
            sp.blockSignals(False)

    def _on_n_pairs_changed(self, n: int):
        self._save_pair()
        default = dict(self._pair_params[-1]) if self._pair_params else \
            {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
        while len(self._pair_params) < n:
            self._pair_params.append(dict(default))
        self._pair_params = self._pair_params[:max(n, 1)]
        self._current_pair = min(self._current_pair, n - 1)
        self._pair_lbl.setup(n, self._current_pair)
        self._load_pair(self._current_pair)
        self.params_changed.emit()

    def _on_pair_changed(self, i: int):
        self._save_pair()
        self._current_pair = i
        self._load_pair(i)

    # ── sections collapsibles + presets ──────────────────────────────────────
    def apply_fit_section_states(self, states: dict | None) -> None:
        if not states:
            return
        for key, grp in self._fit_sections.items():
            if key not in states:
                continue
            expanded = bool(states[key])
            grp.blockSignals(True)
            grp.setChecked(expanded)
            grp.blockSignals(False)
            for i in range(grp.layout().count()):
                w = grp.layout().itemAt(i).widget()
                if w is not None:
                    w.setVisible(expanded)

    def fit_section_states(self) -> dict[str, bool]:
        return {k: bool(g.isChecked()) for k, g in self._fit_sections.items()}

    def set_fit_preset_silent(self, name: str) -> None:
        if not hasattr(self, "cmb_fit_preset"):
            return
        self.cmb_fit_preset.blockSignals(True)
        idx = self.cmb_fit_preset.findText(name or "Custom")
        if idx >= 0:
            self.cmb_fit_preset.setCurrentIndex(idx)
        self.cmb_fit_preset.blockSignals(False)

    def _on_preset_chosen(self, name: str) -> None:
        from arpes.ui.widgets.params_fit import MATERIAL_PRESETS
        preset = MATERIAL_PRESETS.get(name)
        if preset:
            mapping = [
                (self.sp_sff, "smooth_fit"),
                (self.sp_sfd, "smooth_detect"),
                (self.sp_gi, "gamma_init"),
                (self.sp_gm, "gamma_max"),
                (self.sp_ma, "min_amplitude"),
                (self.sp_mj, "max_jump"),
            ]
            for sp, key in mapping:
                if key in preset:
                    sp.blockSignals(True)
                    sp.setValue(float(preset[key]))
                    sp.blockSignals(False)
            if "width_mode" in preset:
                self.cmb_wm.blockSignals(True)
                self.cmb_wm.setCurrentText(str(preset["width_mode"]))
                self.cmb_wm.blockSignals(False)
            self._save_pair()
            self._pair_params[self._current_pair]["gamma_init"] = float(
                preset.get("gamma_init", self.sp_gi.value())
            )
            self._pair_params[self._current_pair]["gamma_max"] = float(
                preset.get("gamma_max", self.sp_gm.value())
            )
            self.fit_only_changed.emit()
        self.fit_preset_changed.emit(name)

    def update_fit_quality(self, fit_result: dict | None, chi2_threshold: float) -> None:
        if not hasattr(self, "lbl_fit_quality"):
            return
        if not fit_result:
            self.lbl_fit_quality.setText("")
            self.lbl_fit_quality.setStyleSheet(
                "color:#888;font-family:monospace;font-size:10px;"
            )
            return
        chi2 = fit_result.get("chi2_red") or []
        arr = np.asarray(chi2, dtype=float)
        arr = arr[np.isfinite(arr)]
        n_total = len(fit_result.get("e_fitted") or [])
        if arr.size == 0:
            self.lbl_fit_quality.setText(
                f"{n_total} slices fittées | χ² indisponible"
            )
            self.lbl_fit_quality.setStyleSheet(
                "color:#888;font-family:monospace;font-size:10px;"
            )
            return
        med = float(np.median(arr))
        bad = int(np.sum(arr > float(chi2_threshold)))
        ratio = bad / max(arr.size, 1)
        color = "#8fc" if ratio < 0.3 else "#fc8"
        self.lbl_fit_quality.setText(
            f"χ²_red méd: {med:.2f}  |  {arr.size} slices  |  {bad} douteuses"
        )
        self.lbl_fit_quality.setStyleSheet(
            f"color:{color};font-family:monospace;font-size:11px;"
        )
