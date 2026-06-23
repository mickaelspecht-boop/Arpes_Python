"""MDC waterfall view for the Results tab (prototype).

Stacks the per-slice fitted MDCs of ONE file (data + model), offset vertically
by binding energy and coloured by energy, with the fitted kF± marked and joined
into the dispersion line on top. Readability is tied to the fit ``step``: only
the fitted slices are drawn, and the display is further decimated above
``MAX_CURVES`` so a step-less (every-row) fit still renders cleanly.

Needs a Full fit (per-slice ``fit_curves`` / ``residuals``). Ensemble fits have
no model curves → a banner explains it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

MAX_CURVES = 40


def _banner(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#ccc",
            fontsize=10, transform=ax.transAxes, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])


def populate_waterfall_files(panel) -> None:
    """Fill the waterfall file selector with the visible fitted files."""
    cmb = getattr(panel, "_cmb_wf_file", None)
    if cmb is None:
        return
    visible = set(panel._visible_files())
    names = [n for n, e in panel._session.files.items()
             if e.fit_result and (e.fit_result.get("e_fitted") is not None)
             and (not visible or n in visible)]
    cur = cmb.currentText()
    cmb.blockSignals(True)
    cmb.clear()
    cmb.addItems(names)
    if cur in names:
        cmb.setCurrentText(cur)
    cmb.blockSignals(False)


def draw_mdc_waterfall(panel) -> None:
    canvas = panel._canvas_wf
    ax = canvas.ax
    ax.cla()
    ax.set_facecolor("#1a1a1a")
    canvas.fig.set_facecolor("#2b2b2b")
    ax.tick_params(colors="w")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")

    name = panel._cmb_wf_file.currentText() if hasattr(panel, "_cmb_wf_file") else ""
    entry = panel._session.files.get(name)
    fr = entry.fit_result if entry else None
    if not fr:
        _banner(ax, "No fitted file selected.")
        canvas.canvas.draw_idle(); return

    _ef = fr.get("e_fitted")
    _fk = fr.get("fit_kpar")
    e_fitted = np.asarray(_ef if _ef is not None else [], dtype=float)
    fit_kpar = np.asarray(_fk if _fk is not None else [], dtype=float)
    curves = fr.get("fit_curves")
    resid = fr.get("residuals")
    if fit_kpar.size == 0 or curves is None or len(curves) == 0 or e_fitted.size == 0:
        _banner(ax, "MDC waterfall needs a Full fit (per-slice model curves).\n"
                    "Run 'Full fit' instead of 'Ensemble'.")
        canvas.canvas.draw_idle(); return

    n = len(e_fitted)
    decim = max(1, int(np.ceil(n / MAX_CURVES)))
    idxs = list(range(0, n, decim))

    # Vertical offset between stacked MDCs, from the typical peak amplitude.
    amps = []
    for i in idxs:
        cv = np.asarray(curves[i], dtype=float)
        if cv.size:
            amps.append(float(np.nanmax(cv) - np.nanmin(cv)))
    amp = float(np.nanmedian(amps)) if amps else 1.0
    off_step = 0.6 * amp if amp > 0 else 1.0

    cmap = plt.get_cmap("viridis")
    emin, emax = float(np.nanmin(e_fitted)), float(np.nanmax(e_fitted))
    espan = (emax - emin) or 1.0
    show_model = (not hasattr(panel, "_chk_wf_model")) or panel._chk_wf_model.isChecked()

    kfm, kfp = fr.get("kF_minus"), fr.get("kF_plus")
    disp_m: list[tuple[float, float]] = []
    disp_p: list[tuple[float, float]] = []
    for j, i in enumerate(idxs):
        cv = np.asarray(curves[i], dtype=float)
        if cv.size != fit_kpar.size:
            continue
        rs = (np.asarray(resid[i], dtype=float)
              if resid is not None and i < len(resid) and np.size(resid[i]) == cv.size
              else np.zeros_like(cv))
        data = cv + rs
        y0 = j * off_step
        col = cmap((e_fitted[i] - emin) / espan)
        ax.plot(fit_kpar, data + y0, color=col, lw=0.6, alpha=0.65)
        if show_model:
            ax.plot(fit_kpar, cv + y0, color=col, lw=1.4, alpha=0.95)
        for branch, store in ((kfm, disp_m), (kfp, disp_p)):
            if not branch:
                continue
            kf = np.asarray(branch[0], dtype=float)
            if i < kf.size and np.isfinite(kf[i]) and fit_kpar.min() <= kf[i] <= fit_kpar.max():
                yi = float(np.interp(kf[i], fit_kpar, data)) + y0
                ax.plot(kf[i], yi, marker="o", ms=3, color="w",
                        mec=col, mew=0.8, zorder=5)
                store.append((float(kf[i]), yi))

    for store, cl in ((disp_m, "#ff6b6b"), (disp_p, "#6bb6ff")):
        if len(store) >= 2:
            xs, ys = zip(*store)
            ax.plot(xs, ys, color=cl, lw=1.1, alpha=0.85, zorder=4)

    ax.set_xlabel(r"$k_\parallel$ (π/a)", color="w", fontsize=10)
    ax.set_ylabel("MDC intensity  (stacked, low E → top = E_F)", color="w", fontsize=9)
    shown = len(idxs)
    extra = f", 1/{decim}" if decim > 1 else ""
    ax.set_title(f"{name} — MDC waterfall ({shown}/{n} fitted slices{extra})",
                 color="w", fontsize=9)
    try:
        canvas.fig.tight_layout()
    except Exception:
        pass
    canvas.canvas.draw_idle()
