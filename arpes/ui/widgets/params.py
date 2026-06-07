"""Fit parameters panel and pair selector (FitParamsPanel).

Section construction (Energy, EF/Loading, Utilities, DFT/Theory,
MDC Fit) is delegated to `params_*` modules to stay under the 700 LOC
limit. Business helpers (update_*, get_fit_params, load_fit_params, pair
management, hν/resolution provenance management) remain here.
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
    """Clickable label for navigating between Lorentzian pairs.
    Left-click → next pair.  Right-click → previous pair."""
    pair_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._current = 0
        self._n = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Active pair: left-click or right arrow = next; right-click or left arrow = previous.")
        self.setAccessibleName("MDC pair selector")
        self.setAccessibleDescription(
            "Changes the active Lorentzian pair for the initial MDC fit parameters."
        )
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
            self.setText("Pair 1 / 1")
        else:
            self.setText(f"<  Pair {self._current + 1} / {self._n}  >")

    def _step_pair(self, delta: int) -> None:
        if self._n < 2:
            return
        self._current = (self._current + int(delta)) % self._n
        self._update()
        self.pair_changed.emit(self._current)

    def mousePressEvent(self, event):
        if self._n < 2:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._step_pair(+1)
        elif event.button() == Qt.MouseButton.RightButton:
            self._step_pair(-1)
        else:
            super().mousePressEvent(event)
            return

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._step_pair(+1)
            return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self._step_pair(-1)
            return
        super().keyPressEvent(event)


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
    propagate_distortion_fs_toggled = pyqtSignal()
    fit_roi_requested = pyqtSignal(bool)
    fit_roi_reset_requested = pyqtSignal()
    fit_undo_requested = pyqtSignal()
    kf_init_drag_changed = pyqtSignal(int, int, float)  # pair_idx, sign(-1/+1), kF (π/a)
    im_self_energy_requested = pyqtSignal()
    fit_ensemble_requested = pyqtSignal()
    theory_import_requested = pyqtSignal()
    theory_refresh_requested = pyqtSignal()
    theory_local_import_requested = pyqtSignal()
    theory_clear_requested = pyqtSignal()
    theory_overlay_changed = pyqtSignal()
    theory_compare_requested = pyqtSignal()
    self_energy_requested = pyqtSignal()
    theory_search_requested = pyqtSignal()
    theory_band_picker_requested = pyqtSignal()
    theory_mu_fit_requested = pyqtSignal()
    theory_align_requested = pyqtSignal()
    theory_efalign_requested = pyqtSignal()
    work_function_changed = pyqtSignal()
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
        """Update the EF reference button label/state from the session."""
        if not ref:
            self.btn_ef_ref.setText("No EF ref")
            self.btn_ef_ref.setEnabled(False)
            self.btn_ef_ref.setToolTip(
                "No EF reference recorded in this session.\n"
                "To create one: run 'Auto EF calibration' on an Au scan, "
                "then check 'Save as reference' in the dialog."
            )
            return
        mode = ref.get("mode", "?")
        src_path = ref.get("source_file", "")
        src_name = Path(src_path).name if src_path else "(unknown source)"
        if mode == "scalar":
            shift_meV = float(ref.get("ef_shift", 0.0)) * 1000.0
            label = f"Apply EF ref ({shift_meV:+.1f} meV)"
        elif mode == "poly":
            n_valid = int(ref.get("n_valid", 0))
            fwhm = float(ref.get("fwhm_res", 0.0)) * 1000.0
            label = f"Apply EF ref poly (n={n_valid})"
        else:
            label = "Apply EF ref"
        self.btn_ef_ref.setText(label)
        self.btn_ef_ref.setEnabled(True)
        self.btn_ef_ref.setToolTip(
            f"Recorded EF reference:\n"
            f"  mode = {mode}\n"
            f"  source = {src_path or '?'}\n"
            f"Applies this correction to the current file."
        )

    def update_hv_source(self, source: str | None):
        """Display the hν provenance: 'file', 'logbook', 'manual', None."""
        labels = {"file": "File", "logbook": "Logbook", "manual": "Manual", "session": "Session"}
        self.lbl_hv_src.setText(labels.get(source or "", "Unknown"))

    def _mark_hv_manual_if_user_edit(self):
        if not getattr(self, "_hv_source_lock", False):
            self.update_hv_source("manual")

    def set_hv_value_with_source(self, value: float, source: str):
        """Set the hv spinbox without triggering the 'manual' marker."""
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
        """Display the resolution provenance: 'estimated', 'manual', 'default'."""
        self._resolution_source = source or "default"
        self._resolution_source_detail = self._resolution_source
        label = {"estimated": "Estimated", "manual": "Manual", "default": "Default"}.get(self._resolution_source, "Default")
        self.lbl_dE_src.setText(label)
        self.lbl_dk_src.setText(label)

    def mark_action_done(self, text: str):
        self.lbl_action.setText(f"Last action: {text}")

    def _mark_resolution_manual_if_user_edit(self):
        if not getattr(self, "_resolution_source_lock", False):
            self.update_resolution_source("manual")
            self._resolution_source_detail = "manual"

    def set_resolution_with_source(self, dE_meV: float, dk_inv_a: float, source: str, detail: str | None = None):
        """Set resolution spinboxes without triggering the manual marker."""
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
        """Update the label/state of the 'Propagate fit params' button."""
        if n_targets <= 0:
            self.btn_copy.setText("Propagate fit params (0 targets)")
            self.btn_copy.setEnabled(False)
            self.btn_copy.setToolTip(
                "No unfitted files in the folder (excluding the current file).\n"
                "All others already have a recorded fit_result and will not be overwritten."
            )
        else:
            self.btn_copy.setText(f"Propagate fit params ({n_targets} target{'s' if n_targets > 1 else ''})")
            self.btn_copy.setEnabled(True)
            self.btn_copy.setToolTip(
                f"Copies the current MDC fit parameters to the {n_targets} "
                f"file(s) in the folder that have not yet been fitted.\n"
                f"Already fitted files are never overwritten."
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
            width_mode=str(self.cmb_wm.currentData() or "independent"),
            min_amplitude=self.sp_ma.value(),
            max_jump=self.sp_mj.value(),
            mdc_energy_window=self.sp_mdc_ewin.value(),
            scan_direction=self.cmb_sd.currentText(),
            dE_meV=self.sp_dE_meV.value(),
            dk_inv_a=self.sp_dk_inv_a.value(),
            pairs=[dict(p) for p in self._pair_params],
            shape=(self.cmb_lineshape.currentData()
                    if hasattr(self, "cmb_lineshape") else "lorentzian"),
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
        """Adapt the right panel to the active tab."""
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
        from arpes.ui.widgets.params_theory_state import theory_overlay_config
        return theory_overlay_config(self)

    def set_theory_overlay_state(self, overlay: dict):
        from arpes.ui.widgets.params_theory_state import set_theory_overlay_state
        return set_theory_overlay_state(self, overlay)

    def _populate_theory_band_table(self, band_meta, band_character, band_indices):
        from arpes.ui.widgets.params_theory_state import populate_theory_band_table
        return populate_theory_band_table(self, band_meta, band_character, band_indices)

    def _on_theory_band_table_toggled(self, _item):
        from arpes.ui.widgets.params_theory_state import on_theory_band_table_toggled
        return on_theory_band_table_toggled(self, _item)

    def _on_theory_bands_text_edited(self):
        from arpes.ui.widgets.params_theory_state import on_theory_bands_text_edited
        return on_theory_bands_text_edited(self)

    def _schedule_theory_overlay_changed(self):
        from arpes.ui.widgets.params_theory_state import schedule_theory_overlay_changed
        return schedule_theory_overlay_changed(self)

    def _emit_theory_overlay_changed_now(self):
        from arpes.ui.widgets.params_theory_state import emit_theory_overlay_changed_now
        return emit_theory_overlay_changed_now(self)

    def load_fit_params(self, fp: FitParams):
        for sp, val in [
            (self.sp_evs, fp.ev_start), (self.sp_eve, fp.ev_end),
            (self.sp_kmin, fp.k_min), (self.sp_kmax, fp.k_max),
            (self.sp_sff, fp.smooth_fit), (self.sp_sfd, fp.smooth_detect),
            (self.sp_xg, fp.xg_range), (self.sp_cx, fp.center_init),
            (self.sp_ma, fp.min_amplitude), (self.sp_mj, fp.max_jump),
            (self.sp_mdc_ewin, getattr(fp, "mdc_energy_window", 0.0)),
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
        # Migration : alias historique asymmetric → independent
        wm = "independent" if str(fp.width_mode) == "asymmetric" else str(fp.width_mode)
        idx_wm = self.cmb_wm.findData(wm)
        if idx_wm >= 0:
            self.cmb_wm.blockSignals(True)
            self.cmb_wm.setCurrentIndex(idx_wm)
            self.cmb_wm.blockSignals(False)
        self.cmb_sd.setCurrentText(fp.scan_direction)
        if hasattr(self, "cmb_lineshape"):
            target = str(getattr(fp, "shape", "lorentzian") or "lorentzian")
            idx = self.cmb_lineshape.findData(target)
            if idx >= 0:
                self.cmb_lineshape.blockSignals(True)
                self.cmb_lineshape.setCurrentIndex(idx)
                self.cmb_lineshape.blockSignals(False)

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

    def _on_pair_param_changed(self, _=None):
        self._save_pair()
        self.fit_only_changed.emit()

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
                wm_p = str(preset["width_mode"])
                if wm_p == "asymmetric":
                    wm_p = "independent"
                idx_p = self.cmb_wm.findData(wm_p)
                if idx_p >= 0:
                    self.cmb_wm.blockSignals(True)
                    self.cmb_wm.setCurrentIndex(idx_p)
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

    def update_fit_quality(self, fit_result: dict | None, chi2_threshold: float,
                           *, current_hash: str | None = None) -> None:
        if not hasattr(self, "lbl_fit_quality"):
            return
        if not fit_result:
            self.lbl_fit_quality.setText("")
            self.lbl_fit_quality.setStyleSheet(
                "color:#888;font-family:monospace;font-size:10px;"
            )
            return
        # numpy-safe : éviter `... or []` (bool ambigu si ndarray non vide)
        _chi2 = fit_result.get("chi2_red")
        arr = np.asarray([] if _chi2 is None else _chi2, dtype=float)
        arr = arr[np.isfinite(arr)]
        _ef = fit_result.get("e_fitted")
        n_total = 0 if _ef is None else len(_ef)
        if arr.size == 0:
            self.lbl_fit_quality.setText(
                f"{n_total} fitted slices | χ² unavailable"
            )
            self.lbl_fit_quality.setStyleSheet(
                "color:#888;font-family:monospace;font-size:10px;"
            )
            return
        med = float(np.median(arr))
        bad = int(np.sum(arr > float(chi2_threshold)))
        ratio = bad / max(arr.size, 1)
        color = "#8fc" if ratio < 0.3 else "#fc8"
        text = f"χ²_red med: {med:.2f}  |  {arr.size} slices  |  {bad} suspect"
        stored_hash = str(fit_result.get("params_hash") or "")
        if current_hash and stored_hash and current_hash != stored_hash:
            text = f"⚠ STALE — params changed since fit | {text}"
            color = "#f87171"  # rouge clair
        self.lbl_fit_quality.setText(text)
        self.lbl_fit_quality.setStyleSheet(
            f"color:{color};font-family:monospace;font-size:12px;font-weight:bold;"
        )
        self.lbl_fit_quality.setToolTip(
            "STALE = MDC/EF/correction parameters changed since this fit.\n"
            "Re-run Full fit to realign."
            if "STALE" in text else "Current MDC fit quality."
        )
