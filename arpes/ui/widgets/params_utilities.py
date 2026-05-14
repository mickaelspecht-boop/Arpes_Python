"""Section 'Utilitaires' regroupée en accordéon (QToolBox).

Contient trois sous-sections collapsables (une seule visible à la fois) :
- Filtre grille (effet trame détecteur)
- DFT / Théorie (overlay bandes calculées)
- Distorsion BM (correction trapèze θ + parabole E)

Chaque sous-section délègue à son builder existant pour le contenu, mais
remet le titre de QGroupBox interne à vide pour éviter la double-bordure.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QToolBox, QVBoxLayout, QWidget


def build_utilities_section(panel, lay) -> None:
    from arpes.ui.widgets.params_distortion import build_bm_distortion_section
    from arpes.ui.widgets.params_ef import build_utils_section
    from arpes.ui.widgets.params_theory import build_theory_section

    panel._utilities_toolbox = QToolBox()
    panel._utilities_toolbox.setStyleSheet(
        "QToolBox::tab { background:#3a3a4a; color:#cde; padding:5px;"
        " border-radius:3px; font-weight:bold; }"
        "QToolBox::tab:selected { background:#4a4a6a; color:#fff; }"
    )

    # ── page 1 : filtre grille ──────────────────────────────────────────────
    page_grid = QWidget()
    page_grid_lay = QVBoxLayout(page_grid)
    page_grid_lay.setContentsMargins(2, 2, 2, 2)
    build_utils_section(panel, page_grid_lay)
    if hasattr(panel, "_utils_widget"):
        panel._utils_widget.setTitle("")
        panel._utils_widget.setFlat(True)
    panel._utilities_toolbox.addItem(page_grid, "Filtre grille (FFT)")

    # ── page 2 : DFT/Théorie ────────────────────────────────────────────────
    page_theory = QWidget()
    page_theory_lay = QVBoxLayout(page_theory)
    page_theory_lay.setContentsMargins(2, 2, 2, 2)
    build_theory_section(panel, page_theory_lay)
    if hasattr(panel, "_theory_widget"):
        panel._theory_widget.setTitle("")
        panel._theory_widget.setFlat(True)
    panel._utilities_toolbox.addItem(page_theory, "DFT / Théorie")

    # ── page 3 : distorsion BM ──────────────────────────────────────────────
    page_dist = QWidget()
    page_dist_lay = QVBoxLayout(page_dist)
    page_dist_lay.setContentsMargins(2, 2, 2, 2)
    build_bm_distortion_section(panel, page_dist_lay)
    if hasattr(panel, "_distortion_widget"):
        panel._distortion_widget.setTitle("")
        panel._distortion_widget.setFlat(True)
    panel._utilities_toolbox.addItem(page_dist, "Distorsion BM")

    panel._utilities_toolbox.setCurrentIndex(0)
    lay.addWidget(panel._utilities_toolbox)
