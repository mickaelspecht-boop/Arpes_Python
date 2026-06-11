"""Per-tab matplotlib + summary rendering extracted from band_analysis_panel.

Each ``show_*_result`` takes the panel as ``p`` so it can reach the
``p.tb_canvas / p.tb_summary / p.tb_notes`` widgets (analogous for kink/gap).
"""
from __future__ import annotations

import numpy as np


def show_tb_result(p, tb: dict, *, k=None, E=None, E_fit=None) -> None:
    params = tb.get("params", {})
    per = tb.get("perr", {})
    parts = [f"<b>Model:</b> {tb.get('model', '')}"]
    for name, v in params.items():
        err = per.get(name, 0.0)
        parts.append(f"<b>{name}</b>={v:.4f}±{err:.4f} eV")
    if tb.get("m_eff_over_me") is not None:
        parts.append(f"<b>m*/m</b>={tb['m_eff_over_me']:.3f}")
    if tb.get("bandwidth_eV") is not None:
        parts.append(f"<b>W</b>={tb['bandwidth_eV']:.3f} eV")
    parts.append(f"χ²_red={tb.get('chi2_red', 0.0):.2e} (N={tb.get('n_points', 0)})")
    p.tb_summary.setText(" — ".join(parts))
    ax = p.tb_canvas.ax
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    if k is not None and E is not None:
        ax.plot(k, E, "o", ms=3, color="#fbbf24", label="MDC peaks")
    if k is not None and E_fit is not None:
        order = np.argsort(k)
        ax.plot(k[order], E_fit[order], "-", lw=1.5,
                color="#60a5fa", label="TB fit")
    ax.set_xlabel("k (Å⁻¹)", color="#ddd")
    ax.set_ylabel("E − E_F (eV)", color="#ddd")
    ax.tick_params(colors="#ddd")
    ax.legend(facecolor="#2b2b2b", edgecolor="#444", labelcolor="#ddd",
              fontsize=8)
    p.tb_canvas.redraw()
    notes = tb.get("notes") or []
    p.tb_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")


def show_kink_result(p, kink: dict) -> None:
    lam = kink.get("lambda")
    lam_err = kink.get("lambda_err")
    vb = kink.get("v_bare")
    parts = []
    if lam is not None:
        parts.append(f"<b>λ</b>={lam:.3f}" + (f"±{lam_err:.3f}" if lam_err else ""))
        # Honest qualitative context (electron-boson coupling literature).
        if lam < 0:
            parts.append("<span style='color:#e05c5c'>unphysical (λ&lt;0) — "
                         "check bare-band window</span>")
        elif lam < 0.3:
            parts.append("<span style='color:#9cf'>weak coupling</span>")
        elif lam <= 1.5:
            parts.append("<span style='color:#7ec97e'>typical range "
                         "(0.3–1.5)</span>")
        else:
            parts.append("<span style='color:#e6b35a'>strong — verify the "
                         "bare model</span>")
    if vb is not None:
        parts.append(f"v_bare={vb:.3f} eV·Å")
    p.kink_summary.setText(" — ".join(parts) or "λ not extractable.")
    E = np.asarray(kink.get("E_exp") or [])
    re = np.asarray(kink.get("re_sigma") or [])
    im = kink.get("im_sigma")
    ax_re, ax_im = p.kink_canvas.axes
    for ax in (ax_re, ax_im):
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#ddd")
    ax_re.plot(E, re, "-o", ms=3, color="#fbbf24")
    ax_re.set_ylabel("Re Σ (eV)", color="#ddd")
    ax_re.axhline(0, color="#666", lw=0.5)
    if im is not None:
        ax_im.plot(E, np.asarray(im), "-o", ms=3, color="#60a5fa")
        ax_im.set_ylabel("Im Σ (eV)", color="#ddd")
    else:
        ax_im.text(0.5, 0.5, "Γ_MDC missing → Im Σ N/A",
                   ha="center", va="center", color="#aaa",
                   transform=ax_im.transAxes)
    ax_im.set_xlabel("E − E_F (eV)", color="#ddd")
    p.kink_canvas.redraw()
    notes = kink.get("notes") or []
    p.kink_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")


def show_gap_result(p, gap: dict) -> None:
    Ds = gap.get("deltas_meV") or []
    errs = gap.get("delta_err_meV") or []
    Gs = gap.get("gammas_meV") or []
    parts = []
    res_meV = float(gap.get("resolution_meV", 0.0) or 0.0)
    for i, D in enumerate(Ds):
        e = errs[i] if i < len(errs) else 0.0
        txt = f"Δ<sub>{i+1}</sub>={D:.2f}±{e:.2f} meV"
        # A gap below ~2× the instrumental resolution is not resolved — say it.
        if res_meV > 0:
            if D < 2.0 * res_meV:
                txt += (" <span style='color:#e05c5c'>(not resolved: "
                        f"&lt;2×res {res_meV:.1f} meV)</span>")
            else:
                txt += " <span style='color:#7ec97e'>✓ &gt;2×res</span>"
        parts.append(txt)
    for i, G in enumerate(Gs):
        parts.append(f"Γ<sub>{i+1}</sub>={G:.2f} meV")
    parts.append(f"k_F={gap.get('k_F_inv_A', 0.0):.3f} Å⁻¹")
    parts.append(f"χ²_red={gap.get('chi2_red', 0.0):.2e}")
    p.gap_summary.setText(" — ".join(parts))
    omega = np.asarray(gap.get("omega_meV") or [])
    I_sym = np.asarray(gap.get("I_sym") or [])
    I_fit = np.asarray(gap.get("I_fit") or [])
    ax = p.gap_canvas.ax
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#ddd")
    ax.plot(omega, I_sym, "o", ms=3, color="#fbbf24", label="symmetrized")
    ax.plot(omega, I_fit, "-", lw=1.5, color="#60a5fa", label="Dynes fit")
    for D in Ds:
        ax.axvline(D, color="#a78bfa", ls="--", lw=0.6)
        ax.axvline(-D, color="#a78bfa", ls="--", lw=0.6)
    ax.set_xlabel("ω = E − E_F (meV)", color="#ddd")
    ax.set_ylabel("I_sym", color="#ddd")
    ax.legend(facecolor="#2b2b2b", edgecolor="#444", labelcolor="#ddd",
              fontsize=8)
    p.gap_canvas.redraw()
    notes = gap.get("notes") or []
    p.gap_notes.setHtml("<br>".join(f"⚠ {n}" for n in notes) if notes else "")


def restore_all(p, ba: dict) -> None:
    if "tb" in ba:
        show_tb_result(p, ba["tb"])
    else:
        p.tb_summary.setText("No TB fit.")
        p.tb_canvas.ax.clear()
        p.tb_canvas.redraw()
        p.tb_notes.clear()
    if "kink" in ba:
        show_kink_result(p, ba["kink"])
    else:
        p.kink_summary.setText("No kink analysis.")
        for ax in p.kink_canvas.axes:
            ax.clear()
        p.kink_canvas.redraw()
        p.kink_notes.clear()
    if "gap" in ba:
        show_gap_result(p, ba["gap"])
    else:
        p.gap_summary.setText("No gap fit.")
        p.gap_canvas.ax.clear()
        p.gap_canvas.redraw()
        p.gap_notes.clear()
