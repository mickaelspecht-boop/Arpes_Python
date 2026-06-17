"""State helpers for FitParamsPanel theory overlay controls."""
from __future__ import annotations


def theory_overlay_config(panel) -> dict:
    return {
        "enabled": bool(panel.chk_theory.isChecked()),
        "material_id": panel.txt_theory_mpid.text().strip(),
        "segment": panel.cmb_theory_segment.currentText().strip(),
        "path_convention": panel.cmb_theory_convention.currentData() or "mp_bulk",
        "mu_shift": float(panel.sp_theory_mu.value()),
        "z_scale": float(panel.sp_theory_z.value()),
        "energy_shift": -float(panel.sp_theory_mu.value()),
        "k_shift": float(panel.sp_theory_dk.value()),
        "k_scale": float(panel.sp_theory_kscale.value()),
        "alpha": float(panel.sp_theory_alpha.value()),
        "max_bands": int(panel.sp_theory_max.value()),
        "mirror_gamma": bool(panel.chk_theory_mirror.isChecked()),
        "band_indices": panel.txt_theory_bands.text().strip(),
        "ef_window": (
            float(panel.sp_theory_efwin.value())
            if panel.chk_theory_ef_only.isChecked() else 0.0
        ),
        "color_by_band": bool(panel.chk_theory_color.isChecked()),
        "with_projections": bool(panel.chk_theory_projections.isChecked()),
        "crystal_a": float(panel.sp_crystal_a.value()),
    }


def set_theory_overlay_state(panel, overlay: dict) -> None:
    data = overlay.get("data") or {}
    config = overlay.get("config") or {}
    segments = list(overlay.get("segments") or [])
    panel.chk_theory.blockSignals(True)
    panel.chk_theory.setChecked(bool(overlay.get("enabled", False)))
    panel.chk_theory.blockSignals(False)
    if data.get("material_id"):
        panel.txt_theory_mpid.setText(str(data.get("material_id")))
    panel.cmb_theory_segment.blockSignals(True)
    panel.cmb_theory_segment.clear()
    panel.cmb_theory_segment.addItem("")
    panel.cmb_theory_segment.addItems(segments)
    if config.get("segment"):
        panel.cmb_theory_segment.setCurrentText(str(config.get("segment")))
    panel.cmb_theory_segment.blockSignals(False)
    panel.cmb_theory_convention.blockSignals(True)
    wanted_convention = str(config.get("path_convention") or "mp_bulk")
    idx = panel.cmb_theory_convention.findData(wanted_convention)
    panel.cmb_theory_convention.setCurrentIndex(max(0, idx))
    panel.cmb_theory_convention.blockSignals(False)
    for sp, key, default in (
        (panel.sp_theory_mu, "mu_shift", -float(config.get("energy_shift", 0.0) or 0.0)),
        (panel.sp_theory_z, "z_scale", 1.0),
        (panel.sp_theory_dk, "k_shift", 0.0),
        (panel.sp_theory_kscale, "k_scale", 1.0),
        (panel.sp_theory_alpha, "alpha", 0.65),
        (panel.sp_theory_max, "max_bands", 10),
    ):
        sp.blockSignals(True)
        sp.setValue(config.get(key, default))
        sp.blockSignals(False)
    panel.chk_theory_mirror.blockSignals(True)
    panel.chk_theory_mirror.setChecked(bool(config.get("mirror_gamma", False)))
    panel.chk_theory_mirror.blockSignals(False)
    panel.txt_theory_bands.blockSignals(True)
    panel.txt_theory_bands.setText(str(config.get("band_indices", "") or ""))
    panel.txt_theory_bands.blockSignals(False)
    win = float(config.get("ef_window", 0.0) or 0.0)
    panel.sp_theory_efwin.blockSignals(True)
    panel.sp_theory_efwin.setValue(win if win > 0 else 0.0)
    panel.sp_theory_efwin.blockSignals(False)
    panel.chk_theory_ef_only.blockSignals(True)
    panel.chk_theory_ef_only.setChecked(win > 0.0)
    panel.chk_theory_ef_only.blockSignals(False)
    panel.chk_theory_color.blockSignals(True)
    panel.chk_theory_color.setChecked(bool(config.get("color_by_band", True)))
    panel.chk_theory_color.blockSignals(False)
    panel.chk_theory_projections.blockSignals(True)
    panel.chk_theory_projections.setChecked(bool(config.get("with_projections", False)))
    panel.chk_theory_projections.blockSignals(False)
    panel._populate_theory_band_table(
        data.get("band_meta") or [],
        data.get("band_character") or [],
        str(config.get("band_indices", "") or ""),
    )
    panel.lbl_theory_status.setText(_theory_status_text(overlay, data, config))
    _sync_anchor_picker(panel, data)


def _sync_anchor_picker(panel, data: dict) -> None:
    """Refresh the manual HS-point label list and disarm pick mode.

    Called on every overlay state change (import / clear / file switch) so the
    label combo always matches the current DFT and a new file never inherits an
    armed picker (the unchecked button re-emits toggled → pick disarmed).
    """
    combo = getattr(panel, "cmb_theory_anchor_label", None)
    if combo is not None:
        names: list[str] = []
        for item in data.get("labels") or []:
            name = str(item.get("label") or "").strip()
            if name and name not in names:
                names.append(name)
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
        combo.blockSignals(False)
    btn = getattr(panel, "btn_theory_anchor_pick", None)
    if btn is not None and btn.isChecked():
        btn.setChecked(False)  # emits toggled → set_pick_active(False)


def _theory_status_text(overlay: dict, data: dict, config: dict) -> str:
    warning = overlay.get("warning") or ""
    mpid = data.get("material_id") or ""
    if not mpid:
        return "Visual guide only."
    source = str(data.get("source") or "")
    prefix = "MP DFT" if source == "materials_project" else "Local DFT"
    efermi = data.get("efermi")
    try:
        ef_txt = f" | DFT E_F={float(efermi):.3f} eV (already subtracted)"
    except (TypeError, ValueError):
        ef_txt = ""
    txt = f"{prefix} {mpid}.{ef_txt} Visual guide; manual alignment required."
    cs = str(data.get("crystal_system") or "")
    if source == "materials_project":
        cs_txt = f" {cs}" if cs else ""
        txt += (
            f"\nPath = 3D BULK BZ{cs_txt} (Setyawan: Γ,X,P,N,Z...). "
            "The FS overlay uses the 2D SURFACE BZ (Γ,X,M,Y,S): "
            "different names are normal (3D bulk != 2D surface)."
        )
    if warning:
        txt += f" Warning: {warning}"
    comparison = overlay.get("comparison") or []
    if comparison:
        best = comparison[0]
        txt += (
            f"\nComparaison: bande {best.get('band_index')} "
            f"{best.get('branch')} paire {int(best.get('pair_index', 0)) + 1}, "
            f"RMS={float(best.get('rms_e', 0.0)) * 1000:.0f} meV "
            f"({int(best.get('n_points', 0))} pts)."
        )
    return txt


def populate_theory_band_table(_panel, _band_meta, _band_character, _band_indices) -> None:
    """Legacy hook kept for old callers; visual picker replaced table."""
    return


def on_theory_band_table_toggled(_panel, _item) -> None:
    """Legacy hook kept for sessions/tests from the old table UI."""
    return


def on_theory_bands_text_edited(panel) -> None:
    """Legacy text field edited manually."""
    panel._schedule_theory_overlay_changed()


def schedule_theory_overlay_changed(panel) -> None:
    """Coalesce live DFT UI edits into one overlay redraw."""
    panel._theory_overlay_timer.start()


def emit_theory_overlay_changed_now(panel) -> None:
    panel._theory_overlay_timer.stop()
    panel.theory_overlay_changed.emit()
