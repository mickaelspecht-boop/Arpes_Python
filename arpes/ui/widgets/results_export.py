"""Scientific figure export for the Results tab (free functions, panel-first).

Split out of ``results.py`` (700-LOC cap). Builds a single white-background
figure with the kF dispersion and the Γ(E) lifetime side by side — axes, grid,
legend of the fitted files, dispersion realignment honoured, and the Γ(E)
reliability mask applied (merged/saturated slices greyed). Writes a provenance
sidecar next to the figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QFileDialog

from arpes.analysis.results import gamma_reliability_mask
from arpes.core.sample import sample_for_entry
from arpes.io.export import export_provenance


def export_fig(panel) -> None:
    path, _ = QFileDialog.getSaveFileName(
        panel, "Export figure", str(panel._session.folder or Path.home()),
        "PDF (*.pdf);;PNG (*.png)")
    if not path:
        return
    fig = build_scientific_export_figure(panel)
    fig.savefig(path, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    write_figure_metadata_sidecar(panel, path)


def build_scientific_export_figure(panel):
    from arpes.ui.widgets import results_bands
    visible = panel._visible_files()
    fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in (ax_d, ax_g):
        ax.set_facecolor("white")
        ax.grid(True, color="#d0d0d0", lw=0.6, alpha=0.85)
        ax.tick_params(colors="black")
        for sp in ax.spines.values():
            sp.set_color("black")
    from arpes.ui.widgets.results_gamma import _gamma_trend
    palette = list(plt.cm.tab10.colors)  # 10 maximally-distinct qualitative colours
    model = "linear" if getattr(panel, "_cmb_gamma_model", None) is not None \
        and panel._cmb_gamma_model.currentIndex() == 1 else "quadratic"
    g_lo, g_hi = panel._gamma_e_range()
    eq_lines: list = []
    plotted_d = 0
    plotted_g = 0
    fidx = 0
    for name, entry in panel._session.files.items():
        if entry.fit_result is None or name not in visible:
            continue
        fr = entry.fit_result
        ev = np.asarray(fr.get("e_fitted", []), dtype=float)
        color = palette[fidx % len(palette)]  # one distinct colour per file
        fidx += 1
        label_base = f"{name}"
        for i in range(int(fr.get("n_pairs") or entry.fit_params.n_pairs or 1)):
            if not results_bands.band_visible(panel, name, i):
                continue
            style = results_bands.band_style(color, i)
            km = np.asarray((fr.get("kF_minus") or [])[i], dtype=float) if i < len(fr.get("kF_minus") or []) else np.array([])
            kp = np.asarray((fr.get("kF_plus") or [])[i], dtype=float) if i < len(fr.get("kF_plus") or []) else np.array([])
            km_p, ev_p = panel._aligned_dispersion_values(name, entry, km, ev)
            kp_p, _ = panel._aligned_dispersion_values(name, entry, kp, ev)
            lbl = results_bands.band_label(name, entry, i)
            ax_d.plot(
                km_p, ev_p, marker=style["marker_minus"],
                linestyle=style["linestyle"], ms=3.2, lw=0.9,
                color=style["color"], alpha=0.90, label=lbl,
            )
            ax_d.plot(
                kp_p, ev_p, marker=style["marker_plus"],
                linestyle=style["linestyle"], ms=3.2, lw=0.9,
                color=style["color"], alpha=0.90, label="_",
            )
            plotted_d += 1
        g_arrays = fr.get("gamma_corrige") or fr.get("gamma") or []
        sg_arrays = fr.get("sigma_gamma") or []
        if not sg_arrays:
            sg_arrays = (fr.get("ensemble") or {}).get("gamma_std") or []
        for i, g_raw in enumerate(g_arrays):
            if not results_bands.band_visible(panel, name, i):
                continue
            style = results_bands.band_style(color, i)
            pairs = list(getattr(entry.fit_params, "pairs", None) or [])
            gmax = pairs[i].get("gamma_max") if i < len(pairs) else entry.fit_params.gamma_max
            g = np.asarray(g_raw, dtype=float)
            n = min(ev.size, g.size)
            if n < 3:
                continue
            e_n = ev[:n]
            finite = np.isfinite(e_n) & np.isfinite(g[:n])
            if int(finite.sum()) < 3:
                continue
            # Export shows ONLY the chosen lifetime fit window [g_lo, g_hi].
            inrange = (e_n >= min(g_lo, g_hi)) & (e_n <= max(g_lo, g_hi))
            reliable = gamma_reliability_mask(fr, pair_index=i, gamma_max=gmax)[:n] & finite & inrange
            unreliable = finite & inrange & ~reliable
            if reliable.any():
                ax_g.plot(
                    e_n[reliable], g[:n][reliable],
                    marker=style["marker_plus"], linestyle=style["linestyle"],
                    ms=3.2, lw=0.9, color=style["color"], alpha=0.90,
                    label=results_bands.band_label(name, entry, i)
                    if plotted_g < 8 else "_",
                )
            if unreliable.any():
                ax_g.plot(e_n[unreliable], g[:n][unreliable], "x", ms=4, color="#999",
                          alpha=0.6, label="unreliable" if plotted_g == 0 else "_")
            sg_arr = None
            if i < len(sg_arrays):
                sg = np.asarray(sg_arrays[i], dtype=float)[:n]
                sg_arr = sg
                bv = reliable & np.isfinite(sg) & (sg > 0)
                if bv.any():
                    ax_g.errorbar(e_n[bv], g[:n][bv], yerr=sg[bv], fmt="none",
                                  ecolor=style["color"], elinewidth=0.7,
                                  capsize=2, alpha=0.7)
            # Chosen trend on the reliable + in-range slices, with its equation.
            sg_rel = sg_arr[reliable] if sg_arr is not None else None
            trend = _gamma_trend(e_n[reliable], g[:n][reliable], sg_rel,
                                 model=model, e_range=(g_lo, g_hi))
            if trend is not None:
                ax_g.plot(trend[0], trend[1], "--", color=style["color"], lw=1.2, alpha=0.9)
                intercept, slope = trend[2], trend[3]
                band_lbl = results_bands.band_label(name, entry, i)
                if model == "linear":
                    eq = f"{band_lbl}: Γ = {intercept:.3f} + {slope:.2f}·E"
                else:
                    eq = f"{band_lbl}: Γ₀ = {intercept:.3f}, a = {slope:.1f}"
                eq_lines.append((eq, style["color"], name, i, intercept, slope))
            plotted_g += 1
    if eq_lines:
        header = ("a + b·E" if model == "linear" else "Γ₀ + a·E²")
        ax_g.text(0.02, 0.98, f"Fit {header}  E∈[{g_lo:.3f}, {g_hi:.3f}] eV",
                  transform=ax_g.transAxes, ha="left", va="top",
                  fontsize=7, fontweight="bold", color="black")
        for j, (eq, col, *_rest) in enumerate(eq_lines[:8]):
            ax_g.text(0.02, 0.93 - 0.05 * j, eq, transform=ax_g.transAxes,
                      ha="left", va="top", fontsize=7, color=col)
    panel._export_gamma_equations = [
        {"file": nm, "pair": pi + 1, "model": model,
         "intercept": ic, "slope": sl, "e_range": [g_lo, g_hi]}
        for (_eq, _c, nm, pi, ic, sl) in eq_lines
    ]
    ax_d.axhline(0, color="black", lw=0.8, ls="--", alpha=0.55)
    ax_d.axvline(0, color="black", lw=0.8, ls="--", alpha=0.55)
    ax_d.set_xlabel(r"$k_\parallel$ (π/a)")
    ax_d.set_ylabel(r"$E - E_F$ (eV)")
    ax_d.set_title("Dispersion kF(E)" + (" — centrée sur Γ" if panel._chk_align_gamma.isChecked() else ""))
    ax_g.set_xlabel(r"$E - E_F$ (eV)")
    ax_g.set_ylabel(r"$\Gamma_k$ (HWHM, π/a)")
    ax_g.set_title(r"Lifetime $\Gamma_k(E)$")
    if plotted_g:
        ax_g.set_xlim(min(g_lo, g_hi), max(g_lo, g_hi))  # only the chosen fit window
    for ax in (ax_d, ax_g):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=7, frameon=True, facecolor="white", edgecolor="#888", loc="best")
    if plotted_d == 0:
        ax_d.text(0.5, 0.5, "Aucune dispersion fitte visible", ha="center", va="center",
                  transform=ax_d.transAxes)
    if plotted_g == 0:
        ax_g.text(0.5, 0.5, "Aucune Γ(E) disponible", ha="center", va="center",
                  transform=ax_g.transAxes)
    return fig


def write_figure_metadata_sidecar(panel, fig_path: str) -> None:
    meta_path = Path(fig_path).with_suffix(".meta.json")
    visible = sorted(panel._visible_files())
    files_meta = []
    for name in visible:
        entry = panel._session.files.get(name)
        if entry is None:
            continue
        m = entry.meta
        files_meta.append({
            "file": name,
            "hv": float(getattr(m, "hv", 0.0) or 0.0),
            "T_K": float(getattr(m, "temperature", 0.0) or 0.0),
            "direction": str(getattr(m, "direction", "") or ""),
            "polarization": str(getattr(m, "polarization", "") or ""),
            "formula": str(getattr(m, "formula", "") or ""),
            "mp_id": str(getattr(m, "mp_id", "") or ""),
            "crystal_a_angstrom": float(getattr(m, "crystal_a_angstrom", 0.0) or 0.0),
            "sample_config": sample_for_entry(panel._session, entry, name).to_dict(),
            "ef_offset": float(getattr(entry, "ef_offset", 0.0) or 0.0),
            "fitted": bool(entry.fit_result),
        })
    payload = {
        "figure": Path(fig_path).name,
        "provenance": export_provenance(
            panel._session, content="figure", file_names=visible,
        ),
        "export_style": "scientific_white_dispersion_gamma",
        "dispersion_alignment": {
            "auto_gamma_center": bool(panel._chk_align_gamma.isChecked()),
            "manual_offsets": {
                name: {"dk_pi_a": dk, "dE_eV": de}
                for name, (dk, de) in sorted(panel._dispersion_offsets.items())
            },
        },
        "gamma_fit_equations": list(getattr(panel, "_export_gamma_equations", []) or []),
        "session_folder": str(panel._session.folder) if panel._session.folder else "",
        "n_files_visible": len(files_meta),
        "files": files_meta,
        "session_notes": str(getattr(panel._session, "session_notes", "") or "")[:500],
    }
    try:
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        pass
