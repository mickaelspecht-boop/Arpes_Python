"""Multi-zone fit runner helpers for FitRunnerController.

Also owns the zones-strip signal wiring (``wire_zones_strip``) so panels.py
stays under the LOC cap, and the debounced ROI/param auto-bind that pushes panel
edits back into the active zone.
"""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from arpes.physics.fit import MdcFitter


def wire_zones_strip(window) -> None:
    """Connect the zones table widget and set up debounced auto-bind.

    Signal contract is identical to the old combo widget. ``rename`` is now
    connected (previously dead). The auto-bind timer snapshots the current panel
    params into the active zone after a short idle; programmatic loads use
    ``blockSignals`` so it only fires on genuine user edits.
    """
    params = getattr(window, "_params", None)
    zs = getattr(params, "zones_strip", None)
    if zs is None:
        return
    zs.add_zone_requested.connect(
        lambda: (window.fit_zone_action("add", {}), _post_mutation(window))
    )
    zs.remove_zone_requested.connect(
        lambda zid: (window.fit_zone_action("remove", {"zone_id": zid}),
                     _post_mutation(window, reload_active=True))
    )
    zs.active_zone_changed.connect(
        lambda zid: (window.fit_zone_action("set_active", {"zone_id": zid}),
                     window._fit_runner_ctrl._on_zone_activated(zid))
    )
    zs.toggle_zone_active.connect(
        lambda zid, on: (window.fit_zone_action(
            "toggle_active", {"zone_id": zid, "value": on}),
            _post_mutation(window))
    )
    zs.rename_zone_requested.connect(
        lambda zid, label: (window.fit_zone_action(
            "rename", {"zone_id": zid, "label": label}),
            _post_mutation(window))
    )
    zs.run_all_zones_requested.connect(window._fit_run_all_zones)
    zs.clear_zone_results.connect(
        lambda: (window.fit_zone_action("clear_results", {}),
                 _post_mutation(window, reload_active=True))
    )

    timer = QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(400)

    def _autobind() -> None:
        res = window.fit_zone_action("update_active_from_params", {})
        if isinstance(res, dict) and res.get("ok"):
            window._refresh_zones_strip()

    timer.timeout.connect(_autobind)
    window._zone_autobind_timer = timer
    for sig_name in ("params_changed", "fit_only_changed"):
        sig = getattr(params, sig_name, None)
        if sig is not None:
            sig.connect(lambda *_, _t=timer: _t.start())


def _post_mutation(window, *, reload_active: bool = False) -> None:
    """Keep table AND plot overlays coherent after any zone CRUD.

    The single source of the "zones don't refresh / don't delete on the map"
    bugs was that the wiring refreshed only the table. Every mutation must also
    redraw the fit views so rectangles/kF appear, move and disappear in step.

    ``reload_active`` reloads the (possibly new) active zone's params + result
    into the panel — needed after remove/clear, where the active selection or
    the result changed. When no zone remains, the panel result is reset.
    """
    window._refresh_zones_strip()
    runner = getattr(window, "_fit_runner_ctrl", None)
    if reload_active:
        path = getattr(window, "_current_path", None)
        entry = None
        if path:
            entry = window._session.get_or_create(window._session.key_for_path(path))
        if entry is not None and entry.active_zone_id and runner is not None:
            # on_zone_activated reloads the panel and redraws every fit view.
            runner._on_zone_activated(entry.active_zone_id)
            return
        # No active zone left: clear the legacy result mirror + panel labels.
        if runner is not None:
            _show_zone_result_in_panel(runner, None)
    # _redraw_all_fit_views lives on the fit-runner controller (not proxied on
    # the window). Removing the last zone took this no-active-zone path and
    # called it on the window → AttributeError.
    if runner is not None:
        runner._redraw_all_fit_views()


def _show_zone_result_in_panel(ctrl, fr) -> None:
    """Reflect a zone's fit_result in the panel: result label, tab marker,
    fit-quality. Keeps the visible summary in sync with the selected zone."""
    p = ctrl._parent
    p._fit_res = fr
    ctrl._update_mdc_tab_label(fr)
    params = ctrl._params
    if not fr:
        try:
            params.lbl_res.setText("")
        except Exception:
            pass
        return
    try:
        crystal_a = float(params.sp_crystal_a.value())
    except Exception:
        crystal_a = 0.0
    try:
        params.lbl_res.setText(MdcFitter.summarize(fr, crystal_a=crystal_a).label_text)
    except Exception:
        pass
    if hasattr(params, "update_fit_quality"):
        try:
            threshold = float(params.sp_chi2_threshold.value())
        except Exception:
            threshold = 5.0
        try:
            params.update_fit_quality(
                fr, threshold, current_hash=ctrl._current_fit_params_hash())
        except Exception:
            pass


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
    """Selecting a zone loads its params + result into the panel."""
    p = ctrl._parent
    # Cancel any pending auto-bind so loading this zone does not write back.
    t = getattr(p, "_zone_autobind_timer", None)
    if t is not None:
        t.stop()
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
        set_fit_result(entry, fr, zone_id=zone_id)
        _show_zone_result_in_panel(ctrl, fr)
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
            set_fit_result(entry, active_z["fit_result"], zone_id=active_z.get("id"))
    ctrl._session.save()
    ctrl._refresh_zones_strip()
    # Re-sync the panel to the ACTIVE zone (params/result/labels) so it does not
    # show the last zone fitted in the loop. on_zone_activated redraws too.
    if entry.active_zone_id:
        on_zone_activated(ctrl, entry.active_zone_id)
    else:
        ctrl._redraw_all_fit_views()
    ctrl._status(
        f"Multi-zone run finished: {n_ok} OK"
        + (f", {n_fail} failed" if n_fail else "")
    )
