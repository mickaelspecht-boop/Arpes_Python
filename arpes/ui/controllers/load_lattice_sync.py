"""Lattice parameter sync helpers for LoadController."""
from __future__ import annotations

from arpes.core.sample import sample_for_entry


def _set_spin(obj, name: str, value: float, *, block: bool = True) -> None:
    sp = getattr(obj, name, None) if obj is not None else None
    if sp is None or float(value or 0.0) <= 0.0:
        return
    try:
        if abs(float(sp.value()) - float(value)) <= 1e-9:
            return
    except Exception:
        pass
    if block and hasattr(sp, "blockSignals"):
        old = sp.blockSignals(True)
        try:
            sp.setValue(float(value))
        finally:
            sp.blockSignals(old)
    else:
        sp.setValue(float(value))


def lattice_a_for_load(ctrl, entry, entry_key: str | None = None) -> float | None:
    """Resolve lattice a before loading axes; persist UI fallback if needed."""
    sample = sample_for_entry(ctrl._session, entry, entry_key=entry_key)
    if sample.has_lattice_a:
        return float(sample.a_angstrom)
    try:
        ui_a = float(ctrl._params.sp_crystal_a.value())
    except Exception:
        ui_a = 0.0
    if ui_a <= 0:
        try:
            fs_controls = getattr(ctrl._parent, "_fs_controls", None)
            ui_a = float(getattr(fs_controls, "sp_a").value()) if fs_controls is not None else 0.0
        except Exception:
            ui_a = 0.0
    if ui_a <= 0:
        return None
    entry.meta.crystal_a_angstrom = float(ui_a)
    _set_spin(ctrl._params, "sp_crystal_a", ui_a)
    try:
        ctrl._session.save()
    except Exception:
        pass
    return float(ui_a)


def sync_lattice_widgets_for_entry(ctrl, entry, entry_key: str | None) -> None:
    """Mirror resolved sample lattice into general, FS and KZ spinboxes."""
    sample = sample_for_entry(ctrl._session, entry, entry_key=entry_key)
    a = float(sample.a_angstrom or 0.0)
    b = float(sample.b_angstrom or a or 0.0)
    c = float(sample.c_angstrom or 0.0)
    _set_spin(ctrl._params, "sp_crystal_a", a)
    fs_controls = getattr(ctrl._parent, "_fs_controls", None)
    _set_spin(fs_controls, "sp_a", a)
    _set_spin(fs_controls, "sp_b", b)
    kz_controls = getattr(ctrl._parent, "_kz_controls", None)
    _set_spin(kz_controls, "sp_a", a)
    _set_spin(kz_controls, "sp_c", c)
