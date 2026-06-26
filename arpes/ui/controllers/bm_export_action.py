"""Band-map figure export action.

Free function taking the plot controller as first argument, kept out of
``plot_controller.py`` to respect the per-file LOC cap. Builds a print-style
band map in the current display mode (raw, second derivative, or curvature)
with only the EF line, and saves it.
"""
from __future__ import annotations

from pathlib import Path


def export_bm_figure(ctrl) -> None:
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    if ctrl._data_disp is None or ctrl._raw_data is None:
        QMessageBox.information(ctrl._parent, "Export figure",
                                "Load a band map first.")
        return
    d = ctrl._raw_data
    disp = ctrl._data_disp
    mode = ctrl._cmb_view.currentText()
    kpar = d["kpar"]
    ev = d["ev_arr"]
    cmap, ckw = ctrl._map_color_kwargs(disp, mode, roi_scale=False)
    fname = Path(d["path"]).name
    path, selected = QFileDialog.getSaveFileName(
        ctrl._parent, "Export band map", "bandmap.png",
        "PNG image, 300 dpi (*.png);;SVG vector (*.svg)",
    )
    if not path:
        return
    if "SVG" in selected and not path.lower().endswith(".svg"):
        path += ".svg"
    elif "PNG" in selected and not path.lower().endswith(".png"):
        path += ".png"
    try:
        from arpes.ui.widgets.bm_export import build_bm_export_figure
        fig = build_bm_export_figure(
            disp, kpar, ev, cmap=cmap, color_kwargs=ckw,
            gamma=ctrl._sp_gamma.value(), title=f"{fname}  [{mode}]",
        )
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight")
    except Exception:
        QMessageBox.critical(
            ctrl._parent, "Export failed",
            f"The figure could not be saved to:\n{path}",
        )
