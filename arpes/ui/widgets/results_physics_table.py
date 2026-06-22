"""Physical-results table population for the Results tab (free function).

Split out of ``results.py`` (700-LOC cap). Fills the "Physical results ± σ"
table (kF, vF, m*, Γ₀) for one file, honouring the bootstrap toggle, the chosen
Γ(E) window, the band-centre kF correction, and flagging a refused (non-linear)
dispersion instead of a silent blank.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTableWidgetItem

from arpes.analysis.results import compute_results
from arpes.core.sample import require_lattice_a, sample_for_entry


def populate_physics_rows(panel, filename: str, fr: dict, n_pairs: int, meta=None) -> None:
    entry = panel._session.files.get(filename)
    try:
        a_val = require_lattice_a(
            sample_for_entry(panel._session, entry, filename), context=filename)
    except ValueError as exc:
        row = panel._table_phys.rowCount()
        panel._table_phys.insertRow(row)
        for col, val in enumerate([filename, "missing a", str(exc), "—", "—", "—"]):
            panel._table_phys.setItem(row, col, QTableWidgetItem(val))
        return
    g_lo, g_hi = panel._gamma_e_range()
    bundle = compute_results(
        fr, e_window_kF=0.10, e_window_gamma=0.30,
        crystal_a_angstrom=a_val,
        gamma_max=getattr(getattr(entry, "fit_params", None), "gamma_max", None),
        gamma_e_lo=g_lo, gamma_e_hi=g_hi,
    )
    if panel._chk_bootstrap.isChecked():
        from arpes.analysis.bootstrap import bootstrap_branch_result
        from arpes.analysis.results import _pair_center
        branches = [
            bootstrap_branch_result(
                fr, branch=br.branch, pair_index=br.pair_index,
                e_window=0.10, crystal_a_angstrom=a_val, n_iter=500,
                center=_pair_center(fr, br.pair_index),
            )
            for br in bundle.branches
        ]
    else:
        branches = bundle.branches
    gamma_by_pair = {g.pair_index: g for g in bundle.gamma_fl}
    for br in branches:
        row = panel._table_phys.rowCount()
        panel._table_phys.insertRow(row)
        label = f"P{br.pair_index + 1} {br.branch.replace('kF_', '')}"
        kf = panel._fmt(br.kF_at_EF, br.kF_at_EF_sigma, dec=4)
        vf = panel._fmt(br.vF_eV_pi_a, br.vF_sigma, dec=2)
        mstar = panel._fmt(br.m_star_over_me, br.m_star_sigma, dec=2)
        g_fl = gamma_by_pair.get(br.pair_index)
        g0 = panel._fmt(g_fl.gamma_zero, g_fl.gamma_zero_sigma, dec=4) if g_fl else "—"
        # Guard: a refused (non-linear / too-few-points) dispersion → show the
        # reason in vF/m* instead of a silent "—".
        reason = "" if getattr(br, "linear_ok", True) else str(getattr(br, "refused_reason", "") or "")
        if reason:
            short = "non-linéaire" if ("nonlinear" in reason or "curvature" in reason) else "fit refusé"
            vf = mstar = f"— {short}"
        for col, val in enumerate([filename, label, kf, vf, mstar, g0]):
            item = QTableWidgetItem(val)
            if reason and col in (3, 4):
                item.setForeground(QColor("#e6b35a"))
                item.setToolTip(
                    f"vF/m* non extraits : {reason}.\n"
                    "La bande n'est pas linéaire dans la fenêtre E (elle se "
                    "retourne / kink). Raccourcis la fenêtre E à la partie "
                    "linéaire sous le retournement, puis relance.")
            panel._table_phys.setItem(row, col, item)
