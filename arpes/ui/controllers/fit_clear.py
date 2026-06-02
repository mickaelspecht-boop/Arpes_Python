"""Clear-fit helper for FitRunnerController."""
from __future__ import annotations


def clear_kf(ctrl) -> None:
    p = ctrl._parent
    p._fit_res = None
    if p._current_path:
        from arpes.core.fit_result_store import clear_fit_result
        key = ctrl._session.key_for_path(p._current_path)
        entry = ctrl._session.get_or_create(key)
        clear_fit_result(entry)
        entry.annotations = {}
        ctrl._session.save()
        if hasattr(p, "_browser"):
            p._browser.refresh_item(key)
    p._fit_selected = []
    if hasattr(ctrl._params, "update_fit_quality"):
        ctrl._params.update_fit_quality(None, 5.0)
    ctrl._update_mdc_tab_label(None)
    ctrl._redraw_all_fit_views()
    ctrl._params.lbl_res.setText("kF effacé")
    results = getattr(p, "_results", None)
    if results is not None and hasattr(results, "refresh"):
        results.refresh()
