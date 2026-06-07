"""Summary tab HTML render extracted from band_analysis_panel.

Pure formatting helpers (no Qt state) — return ready-to-setHtml strings.
"""
from __future__ import annotations


def cross_validation_block(tb: dict, kink: dict) -> str | None:
    m_over_me = tb.get("m_eff_over_me") if tb else None
    lam = kink.get("lambda") if kink else None
    if m_over_me is None or lam is None:
        return None
    predicted = 1.0 + lam
    ratio = m_over_me / predicted if predicted > 0 else float("nan")
    flag = ""
    if abs(ratio - 1.0) > 0.3:
        flag = " ⚠ discrepancy &gt;30%: review bare band (kink) or TB model."
    return (
        f"m*/m = {m_over_me:.3f} vs predicted (1+λ) = {predicted:.3f}"
        f" — ratio = {ratio:.2f}.{flag}"
    )


def render_summary_html(
    ba: dict, *, has_fit: bool, n_points: int, n_pairs: int,
) -> str:
    lines: list[str] = []
    lines.append("<table cellpadding='3' style='font-size:11px;'>")
    lines.append("<tr><th align='left'>Source</th><th align='left'>Metric</th>"
                 "<th align='left'>Value</th><th align='left'>Note</th></tr>")
    if has_fit:
        lines.append(
            f"<tr><td>MDC</td><td>points</td><td>{n_points}</td>"
            f"<td>{n_pairs} pair(s)</td></tr>"
        )
    else:
        lines.append("<tr><td colspan='4'><i>No MDC fit. Run the "
                     "MDC fit to enable analyses.</i></td></tr>")
        lines.append("</table>")
        return "\n".join(lines)
    tb = ba.get("tb") or {}
    kink = ba.get("kink") or {}
    gap = ba.get("gap") or {}
    if tb:
        params = tb.get("params", {})
        perr = tb.get("perr", {})
        for name, v in params.items():
            err = perr.get(name, 0.0)
            lines.append(
                f"<tr><td>TB</td><td>{name}</td>"
                f"<td>{v:+.4f} ± {err:.4f} eV</td><td></td></tr>"
            )
        if tb.get("m_eff_over_me") is not None:
            lines.append(
                f"<tr><td>TB</td><td>m*/m</td>"
                f"<td>{tb['m_eff_over_me']:.3f}</td><td></td></tr>"
            )
        if tb.get("bandwidth_eV") is not None:
            lines.append(
                f"<tr><td>TB</td><td>W (bandwidth)</td>"
                f"<td>{tb['bandwidth_eV']:.3f} eV</td><td></td></tr>"
            )
        lines.append(
            f"<tr><td>TB</td><td>χ²_red</td>"
            f"<td>{tb.get('chi2_red', 0.0):.2e}</td>"
            f"<td>N={tb.get('n_points', 0)}</td></tr>"
        )
    if kink:
        lam = kink.get("lambda")
        err = kink.get("lambda_err")
        vb = kink.get("v_bare")
        if lam is not None:
            note = ""
            if lam < 0:
                note = "⚠ λ&lt;0 unphysical"
            elif lam > 2.5:
                note = "⚠ λ very high"
            lines.append(
                f"<tr><td>Kink</td><td>λ</td>"
                f"<td>{lam:.3f}"
                + (f" ± {err:.3f}" if err else "")
                + f"</td><td>{note}</td></tr>"
            )
        if vb is not None:
            lines.append(
                f"<tr><td>Kink</td><td>v_bare</td>"
                f"<td>{vb:.3f} eV·Å</td><td></td></tr>"
            )
    if gap:
        for i, D in enumerate(gap.get("deltas_meV") or []):
            errs = gap.get("delta_err_meV") or []
            e = errs[i] if i < len(errs) else 0.0
            lines.append(
                f"<tr><td>Gap</td><td>Δ<sub>{i+1}</sub></td>"
                f"<td>{D:.2f} ± {e:.2f} meV</td><td></td></tr>"
            )
        for i, G in enumerate(gap.get("gammas_meV") or []):
            lines.append(
                f"<tr><td>Gap</td><td>Γ<sub>{i+1}</sub></td>"
                f"<td>{G:.2f} meV</td><td></td></tr>"
            )
        lines.append(
            f"<tr><td>Gap</td><td>k_F</td>"
            f"<td>{gap.get('k_F_inv_A', 0.0):.3f} Å⁻¹</td><td></td></tr>"
        )
        lines.append(
            f"<tr><td>Gap</td><td>χ²_red</td>"
            f"<td>{gap.get('chi2_red', 0.0):.2e}</td><td></td></tr>"
        )
    lines.append("</table>")
    cross = cross_validation_block(tb, kink)
    if cross:
        lines.append("<br><b>Consistency:</b><br>")
        lines.append(cross)
    all_notes: list[str] = []
    for label, payload in (("TB", tb), ("Kink", kink), ("Gap", gap)):
        for n in payload.get("notes") or []:
            all_notes.append(f"<b>[{label}]</b> {n}")
    if all_notes:
        lines.append("<br><br><b>Warnings:</b><br>")
        lines.append("<br>".join(f"⚠ {n}" for n in all_notes))
    return "\n".join(lines)
