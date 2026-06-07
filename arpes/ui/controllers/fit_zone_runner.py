"""Multi-zone fit runner helpers for FitRunnerController."""
from __future__ import annotations

import traceback

from PyQt6.QtWidgets import QApplication

from arpes.physics.fit import MdcFitter


def refresh_zones_strip(ctrl) -> None:
    p = ctrl._parent
    if not hasattr(p._params, "zones_strip"):
        return
    if p._current_path:
        key = ctrl._session.key_for_path(p._current_path)
        entry = ctrl._session.get_or_create(key)
        p._params.zones_strip.set_zones(entry.fit_zones, entry.active_zone_id)
    else:
        p._params.zones_strip.set_zones([], None)


def on_zone_activated(ctrl, zone_id: str) -> None:
    """Selecting a zone in the strip loads its params into the spinboxes."""
    p = ctrl._parent
    if not p._current_path:
        return
    key = ctrl._session.key_for_path(p._current_path)
    entry = ctrl._session.get_or_create(key)
    for z in entry.fit_zones:
        if z.get("id") != zone_id:
            continue
        try:
            from arpes.core.session import FitParams
            fp = FitParams(**{
                k: v for k, v in z.get("fit_params", {}).items()
                if k in FitParams.__dataclass_fields__
            })
            ctrl._params.load_fit_params(fp)
        except Exception:
            pass
        from arpes.core.fit_result_store import set_fit_result
        fr = z.get("fit_result")
        p._fit_res = fr
        set_fit_result(entry, fr, zone_id=zone_id)
        ctrl._update_mdc_tab_label(fr)
        ctrl._redraw_all_fit_views()
        return


def fit_run_all_zones(ctrl) -> None:
    """Run an independent MDC fit for every active zone, sequentially."""
    p = ctrl._parent
    if p.ap is None:
        ctrl._status("Warning: arpes_plots not loaded")
        return
    if not p._current_path:
        return
    key = ctrl._session.key_for_path(p._current_path)
    entry = ctrl._session.get_or_create(key)
    active = [z for z in entry.fit_zones if z.get("active", True)]
    if not active:
        ctrl._status("No active zone. Click '+' to create one.")
        return
    controller = MdcFitter(p.ap)
    n_ok = 0
    n_fail = 0
    zctrl = getattr(p, "_fit_zones_ctrl", None)
    # Snapshot global gamma before the loop so asymmetric warning compares
    # each zone against the same reference, not its own center_init.
    try:
        gamma_global = float(ctrl._params.sp_cx.value())
    except Exception:
        gamma_global = 0.0
    for i, zone in enumerate(active):
        ctrl._status(f"Fit zone {i + 1}/{len(active)} ({zone.get('label')}) ...")
        QApplication.processEvents()
        try:
            from arpes.core.session import FitParams
            fp = FitParams(**{
                k: v for k, v in zone.get("fit_params", {}).items()
                if k in FitParams.__dataclass_fields__
            })
            ctrl._params.load_fit_params(fp)
            data, kpar, ev = ctrl._get_work_data()
            if data is None:
                n_fail += 1
                continue
            fr = controller.run_full_fit(
                data, kpar, ev, fp,
                resolution_source=getattr(
                    ctrl._params, "_resolution_source_detail", "",
                ),
            )
            fr["params_hash"] = ctrl._current_fit_params_hash(entry, fp=fp)
            fr["zone_id"] = zone["id"]
            fr["zone_label"] = zone.get("label")
            from arpes.physics.distortion import is_distortion_active
            fr["distorted"] = bool(
                entry.bm_distortion
                and is_distortion_active(entry.bm_distortion)
            )
            fr["grid_active"] = bool((entry.grid_correction or {}).get("enabled"))
            if zctrl is not None:
                try:
                    warn = zctrl.asymmetric_warning(zone, gamma_global)
                    if warn:
                        fr["asymmetric_warning"] = warn
                        ctrl._status(warn)
                except Exception:
                    pass
            if zctrl is not None:
                zctrl.store_result(zone["id"], fr)
            n_ok += 1
        except Exception as exc:
            traceback.print_exc()
            n_fail += 1
            ctrl._status(f"Zone {zone.get('label')} failed: {exc}")
    if zctrl is not None:
        from arpes.core.fit_result_store import set_fit_result
        active_z = zctrl.active_zone(entry)
        if active_z and active_z.get("fit_result"):
            p._fit_res = active_z["fit_result"]
            set_fit_result(entry, p._fit_res, zone_id=active_z.get("id"))
    ctrl._session.save()
    ctrl._refresh_zones_strip()
    ctrl._redraw_all_fit_views()
    ctrl._status(
        f"Multi-zone run finished: {n_ok} OK"
        + (f", {n_fail} failed" if n_fail else "")
    )
