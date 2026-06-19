"""Link fitted kF points between the Results plot and the BM map."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PICK_RADIUS_PX = 10.0


def append_branch_refs(panel, filename: str, branch: str, pair_idx: int, k_values, e_values) -> None:
    refs = getattr(panel, "_result_point_refs", None)
    if refs is None:
        panel._result_point_refs = refs = []
    k = np.asarray(k_values, dtype=float)
    e = np.asarray(e_values, dtype=float)
    n = min(k.size, e.size)
    for idx in range(n):
        if np.isfinite(k[idx]) and np.isfinite(e[idx]):
            refs.append({
                "file": str(filename),
                "branch": str(branch),
                "pair": int(pair_idx),
                "index": int(idx),
                "k": float(k[idx]),
                "e": float(e[idx]),
            })


def on_results_click(panel, event) -> None:
    if event.inaxes is not panel._canvas.ax or event.xdata is None or event.ydata is None:
        return
    ref = _nearest_result_ref(panel, event)
    if ref is None:
        return
    select_result_ref(panel, ref, navigate_to_bm=True)


def select_result_ref(panel, ref: dict[str, Any], *, navigate_to_bm: bool) -> None:
    panel._linked_selection = dict(ref)
    highlight_results_selection(panel)
    if navigate_to_bm:
        _navigate_to_bm(panel, ref)


def sync_from_bm_selection(panel, filename: str, selection: tuple[str, int, int] | None) -> None:
    if selection is None:
        panel._linked_selection = None
        highlight_results_selection(panel)
        return
    branch, pair_idx, point_idx = selection
    for ref in getattr(panel, "_result_point_refs", []) or []:
        if (
            ref.get("file") == filename
            and ref.get("branch") == branch
            and int(ref.get("pair", -1)) == int(pair_idx)
            and int(ref.get("index", -1)) == int(point_idx)
        ):
            panel._linked_selection = dict(ref)
            highlight_results_selection(panel)
            return
    panel._linked_selection = {
        "file": filename, "branch": branch, "pair": int(pair_idx),
        "index": int(point_idx),
    }


def highlight_results_selection(panel) -> None:
    old = getattr(panel, "_linked_result_artist", None)
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    panel._linked_result_artist = None
    ref = getattr(panel, "_linked_selection", None)
    if not ref or "k" not in ref or "e" not in ref:
        panel._canvas.redraw()
        return
    ax = panel._canvas.ax
    panel._linked_result_artist = ax.scatter(
        [float(ref["k"])], [float(ref["e"])],
        s=95, facecolors="none", edgecolors="#fbbf24",
        linewidths=1.8, zorder=20,
    )
    panel._canvas.redraw()


def _nearest_result_ref(panel, event) -> dict[str, Any] | None:
    refs = getattr(panel, "_result_point_refs", []) or []
    if not refs:
        return None
    pts = np.asarray([[r["k"], r["e"]] for r in refs], dtype=float)
    disp = event.inaxes.transData.transform(pts)
    click = np.asarray(event.inaxes.transData.transform((float(event.xdata), float(event.ydata))))
    d2 = np.sum((disp - click) ** 2, axis=1)
    idx = int(np.argmin(d2))
    if float(d2[idx]) > _PICK_RADIUS_PX ** 2:
        return None
    return dict(refs[idx])


def _navigate_to_bm(panel, ref: dict[str, Any]) -> None:
    host = getattr(panel, "_host", None)
    if host is None:
        return
    filename = str(ref["file"])
    path = _path_for_session_key(panel, filename)
    if path is not None:
        current = getattr(host, "_current_path", None)
        if current is None or _norm_path(current) != _norm_path(path):
            browser = getattr(host, "_browser", None)
            if browser is not None:
                try:
                    browser.select_file(str(path))
                except Exception:
                    pass
            host._load_file(str(path))
    host._fit_selected = [(str(ref["branch"]), int(ref["pair"]), int(ref["index"]))]
    host._sel_k = float(ref["k"])
    host._sel_ev = float(ref["e"])
    try:
        from arpes.ui.tab_index import IDX_BM
        host._tabs.setCurrentIndex(IDX_BM)
    except Exception:
        pass
    try:
        host._draw_current_view(include_curves=False)
    except Exception:
        pass
    try:
        host._status(
            f"Linked kF point: {filename} {ref['branch']} P{int(ref['pair']) + 1} "
            f"#{int(ref['index']) + 1} at k={float(ref['k']):+.4f}, E={float(ref['e']):+.4f}."
        )
    except Exception:
        pass


def _path_for_session_key(panel, filename: str) -> Path | None:
    p = Path(filename)
    if p.is_absolute():
        return p
    folder = getattr(panel._session, "folder", None)
    return Path(folder) / filename if folder else p


def _norm_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)
