"""FS controller restore helpers."""
from __future__ import annotations


def restore_fs_crystal_settings_from_entry(ctrl, entry) -> None:
    """Sync BZ-crystal widgets from FileEntry on file load."""
    if not hasattr(ctrl, "_fs_controls") or entry is None:
        return
    c = ctrl._fs_controls
    widgets = [c.sp_v0, c.cmb_kz_plane, c.sp_phi_c, c.chk_bz_xtal, c.chk_hs_xtal, c.ed_mp_id]
    if hasattr(c, "sp_fs_rotation"):
        widgets.append(c.sp_fs_rotation)
    for w in widgets:
        w.blockSignals(True)
    try:
        if hasattr(c, "sp_fs_rotation"):
            c.sp_fs_rotation.setValue(float(getattr(entry, "fs_rotation_deg", 0.0) or 0.0))
        c.sp_v0.setValue(float(getattr(entry, "fs_v0", 12.0) or 12.0))
        plane = str(getattr(entry, "fs_kz_plane", "Auto") or "Auto")
        idx = c.cmb_kz_plane.findText(plane)
        if idx >= 0:
            c.cmb_kz_plane.setCurrentIndex(idx)
        c.sp_phi_c.setValue(float(getattr(entry, "fs_phi_c_deg", 0.0) or 0.0))
        c.chk_bz_xtal.setChecked(bool(getattr(entry, "fs_bz_crystal_visible", False)))
        c.chk_hs_xtal.setChecked(bool(getattr(entry, "fs_hs_crystal_visible", False)))
        lat = getattr(entry, "fs_lattice", None) or {}
        c.ed_mp_id.setText(str(lat.get("mp_id", "") or ""))
    finally:
        for w in widgets:
            w.blockSignals(False)
    if hasattr(ctrl, "_params") and hasattr(ctrl._params, "chk_distortion_fs_propagate"):
        ctrl._sync_distortion_fs_toggles(bool(getattr(entry, "propagate_distortion_to_fs", False)))
    ctrl._fs_distortion_cache_invalidate()
