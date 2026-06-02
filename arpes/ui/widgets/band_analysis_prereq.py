"""Prerequisite and autofill helpers for BandAnalysisPanel."""
from __future__ import annotations


def update_prerequisites(panel, *, has_fit: bool, n_pairs: int, n_points: int = 0) -> None:
    """Refresh badges + enable/disable Run, hide pair spinbox if 1 paire."""
    panel._has_fit = bool(has_fit)
    panel._n_pairs = max(1, int(n_pairs))
    if has_fit:
        badge_txt = f"● MDC ✓ {n_points} pts, {panel._n_pairs} paire(s)"
        badge_css = (
            "color:#86efac; background:#14532d; padding:2px 6px;"
            " border-radius:3px; font-size:10px;"
        )
        run_enabled = True
    else:
        badge_txt = "⚠ MDC non fitté — onglet désactivé"
        badge_css = (
            "color:#fca5a5; background:#7f1d1d; padding:2px 6px;"
            " border-radius:3px; font-size:10px;"
        )
        run_enabled = False
    for badge in (panel.tb_badge, panel.kink_badge, panel.gap_badge):
        badge.setText(badge_txt)
        badge.setStyleSheet(badge_css)
    for btn in (panel.tb_run_btn, panel.kink_run_btn, panel.gap_run_btn):
        btn.setEnabled(run_enabled)
    for spin in (panel.tb_pair, panel.kink_pair, panel.gap_pair):
        spin.setMaximum(max(0, panel._n_pairs - 1))
    show_pair = panel._n_pairs > 1
    for lbl, spin in (
        panel._tb_pair_form_row,
        panel._kink_pair_form_row,
        panel._gap_pair_form_row,
    ):
        lbl.setVisible(show_pair)
        spin.setVisible(show_pair)


def apply_autofill(panel, target: str, defaults: dict) -> None:
    """Apply auto-filled defaults to a specific tab's spinboxes."""
    if target == "tb":
        if "a" in defaults:
            panel.tb_a.setValue(float(defaults["a"]))
        if "branch" in defaults:
            idx = panel.tb_branch.findText(str(defaults["branch"]))
            if idx >= 0:
                panel.tb_branch.setCurrentIndex(idx)
    elif target == "kink":
        if "E_F" in defaults:
            panel.kink_EF.setValue(float(defaults["E_F"]))
        if "window_lo" in defaults:
            panel.kink_win_lo.setValue(float(defaults["window_lo"]))
        if "window_hi" in defaults:
            panel.kink_win_hi.setValue(float(defaults["window_hi"]))
        if "branch" in defaults:
            idx = panel.kink_branch.findText(str(defaults["branch"]))
            if idx >= 0:
                panel.kink_branch.setCurrentIndex(idx)
    elif target == "gap":
        if "E_F" in defaults:
            panel.gap_EF.setValue(float(defaults["E_F"]))
        if "omega_max_meV" in defaults:
            panel.gap_omega_max.setValue(float(defaults["omega_max_meV"]))
        if "branch" in defaults:
            idx = panel.gap_branch.findText(str(defaults["branch"]))
            if idx >= 0:
                panel.gap_branch.setCurrentIndex(idx)
