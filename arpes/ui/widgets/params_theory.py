"""Section DFT / Theory du FitParamsPanel.

Builder externe pour le groupbox theory overlay (MP-ID, segment, alignement
manuel, boutons import/comparer/vider, recherche par formule).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from arpes.ui.widgets._qt_helpers import compact_button, dspin, ispin
from arpes.ui.app_settings import get_mp_api_key, set_mp_api_key


def build_theory_section(panel, lay) -> None:
    panel._theory_widget = QGroupBox("DFT / Theory")
    fl_th = QFormLayout(panel._theory_widget)
    panel.chk_theory = QCheckBox("Show DFT overlay")
    panel.chk_theory.setToolTip(
        "Displays imported DFT bands as a visual guide.\n"
        "No fit or correction uses these bands automatically."
    )
    panel.txt_theory_mpid = QLineEdit()
    panel.txt_theory_mpid.setPlaceholderText("mp-149")
    panel.txt_theory_mpid.setToolTip(
        "Materials Project ID. Press Enter to import directly.\n"
        "Requires mp-api + a Materials Project API key (field below or MP_API_KEY env)."
    )
    panel.txt_theory_mpid.returnPressed.connect(panel.theory_import_requested)
    # User-supplied Materials Project API key (masked). Persisted per-user via
    # QSettings (never in the repo or session). Empty -> falls back to MP_API_KEY
    # env var, so a developer's local key keeps working without typing it here.
    panel.txt_mp_api_key = QLineEdit()
    panel.txt_mp_api_key.setEchoMode(QLineEdit.EchoMode.Password)
    panel.txt_mp_api_key.setPlaceholderText("your Materials Project API key")
    panel.txt_mp_api_key.setText(get_mp_api_key())
    panel.txt_mp_api_key.setToolTip(
        "Get a free key at materialsproject.org (Dashboard → API).\n"
        "Stored locally for this user only; never uploaded or shared.\n"
        "Leave empty to use the MP_API_KEY environment variable instead."
    )
    panel.txt_mp_api_key.editingFinished.connect(
        lambda: set_mp_api_key(panel.txt_mp_api_key.text())
    )
    panel.btn_theory_search = compact_button(QPushButton("Search MP"), max_width=130)
    panel.btn_theory_search.setToolTip(
        "Searches Materials Project by chemical formula (network).\n"
        "Opens a dialog with candidates and their MPID."
    )
    panel.btn_theory_search.clicked.connect(panel.theory_search_requested)
    mpid_row = QWidget()
    mpid_lay = QHBoxLayout(mpid_row)
    mpid_lay.setContentsMargins(0, 0, 0, 0)
    mpid_lay.addWidget(panel.txt_theory_mpid, 1)
    mpid_lay.addWidget(panel.btn_theory_search)
    panel.cmb_theory_segment = QComboBox()
    panel.cmb_theory_segment.setEditable(True)
    panel.cmb_theory_segment.setToolTip("DFT segment suggested from logbook direction; editable.")
    panel.cmb_theory_convention = QComboBox()
    panel.cmb_theory_convention.addItem("MP bulk 3D", "mp_bulk")
    panel.cmb_theory_convention.addItem("ARPES pnictides 2D", "arpes_pnictides")
    panel.cmb_theory_convention.setToolTip(
        "Label-display convention in the picker.\n"
        "MP bulk = raw 3D Materials Project path.\n"
        "ARPES pnictides = adds common 2D Γ/X/M/S aliases as annotations."
    )
    panel.sp_theory_mu = dspin(0.0, -5.0, 5.0, 0.05, dec=3)
    panel.sp_theory_mu.setKeyboardTracking(False)
    panel.sp_theory_mu.setToolTip(
        "DFT chemical shift μ (eV), before renormalization.\n"
        "Overlay transform: E = Z x (E_DFT - μ).\n"
        "μ moves DFT Fermi crossings; Z adjusts dispersion."
    )
    # Alias legacy: d'anciens controllers/tests utilisent encore sp_theory_de.
    panel.sp_theory_de = panel.sp_theory_mu
    panel.sp_theory_z = dspin(1.0, 0.05, 5.0, 0.05, dec=3)
    panel.sp_theory_z.setKeyboardTracking(False)
    panel.sp_theory_z.setToolTip(
        "Global dispersion renormalization Z.\n"
        "Z=1: unchanged DFT. Z<1: flatter bands.\n"
        "Applied globally to all selected bands."
    )
    panel.sp_theory_dk = dspin(0.0, -5.0, 5.0, 0.02, dec=3)
    panel.sp_theory_dk.setKeyboardTracking(False)
    panel.sp_theory_dk.setToolTip("Manual k shift applied to DFT bands (displayed π/a).")
    panel.sp_theory_kscale = dspin(1.0, 0.1, 5.0, 0.05, dec=3)
    panel.sp_theory_kscale.setKeyboardTracking(False)
    panel.sp_theory_kscale.setToolTip("Manual k scale factor to bring DFT and ARPES axes closer.")
    panel.sp_theory_alpha = dspin(0.65, 0.05, 1.0, 0.05, dec=2)
    panel.sp_theory_max = ispin(10, 1, 80)
    panel.sp_crystal_a = dspin(0.0, 0.0, 50.0, 0.001, dec=4)
    panel.sp_crystal_a.setToolTip(
        "Crystal lattice parameter a (Å). Used to convert kF (π/a) to\n"
        "Å⁻¹ and compute m*/m_e in the results table. 0 = reduced units."
    )
    panel.sp_crystal_a.valueChanged.connect(panel.crystal_a_changed)
    panel.txt_theory_bands = QLineEdit()
    panel.txt_theory_bands.setPlaceholderText("e.g. 1,3,5-8 (empty = top-N)")
    panel.txt_theory_bands.setToolTip(
        "0-based DFT band indices to display. Ranges accepted: `1,3,5-8`.\n"
        "Empty -> automatic top-N selection by Y-window overlap."
    )
    panel.txt_theory_bands.editingFinished.connect(panel._on_theory_bands_text_edited)
    panel.btn_theory_pick_bands = QPushButton("Choose bands...")
    panel.btn_theory_pick_bands.setToolTip(
        "Opens the DFT diagram and lets you click the bands to display."
    )
    panel.btn_theory_pick_bands.clicked.connect(panel.theory_band_picker_requested)
    # --- filtre fenêtre E_F (B+A) ---
    panel.sp_theory_efwin = dspin(0.0, 0.0, 10.0, 0.05, dec=3)
    panel.sp_theory_efwin.setKeyboardTracking(False)
    panel.sp_theory_efwin.setToolTip(
        "±E window around E_F (eV). 0 = disabled.\n"
        "Filters the overlay and list: only bands crossing\n"
        "this window around E_F (E=0) are kept."
    )
    panel.chk_theory_ef_only = QCheckBox("Only bands crossing E_F")
    panel.chk_theory_ef_only.setToolTip(
        "Prechecks only bands crossing the E_F window\n"
        "above and restricts the overlay to them."
    )
    panel.chk_theory_color = QCheckBox("Color by band + legend")
    panel.chk_theory_color.setChecked(True)
    panel.chk_theory_color.setToolTip(
        "Stable color per band index + legend.\n"
        "Unchecked -> all bands in white (legacy rendering)."
    )
    panel.chk_theory_projections = QCheckBox("Fetch orbital projections (network)")
    panel.chk_theory_projections.setToolTip(
        "On MP import: tries to fetch orbital projections\n"
        "to show character (e.g. Ti-d) per band. Slower.\n"
        "No effect if Materials Project does not provide projections."
    )
    for w in (panel.sp_theory_efwin, panel.chk_theory_ef_only,
              panel.chk_theory_color):
        sig = w.stateChanged if isinstance(w, QCheckBox) else w.valueChanged
        sig.connect(panel._schedule_theory_overlay_changed)
    panel.chk_theory_mirror = QCheckBox("Mirror Γ (k → -k)")
    panel.chk_theory_mirror.setToolTip(
        "Duplicates DFT bands mirrored around Γ (k → -k).\n"
        "Useful for symmetric scans [-X, Γ, +X] where the Setyawan-Curtarolo path\n"
        "covers only the right half. Valid for centrosymmetric crystals\n"
        "(e.g. BaNi₂As₂ I4/mmm)."
    )
    btn_theory_refresh = QPushButton("Refresh MP")
    btn_theory_refresh.setToolTip(
        "Reimports the MP-ID while ignoring the disk cache.\n"
        "Needed to fetch the true MP band path\n"
        "(branches) if the import was cached before this version."
    )
    btn_theory_refresh.clicked.connect(panel.theory_refresh_requested)
    btn_theory_local_import = QPushButton("Import local")
    btn_theory_local_import.setToolTip(
        "Imports local DFT bands from vasprun.xml, QE .dat/.txt table,\n"
        "or minimal YAML/JSON schema."
    )
    btn_theory_local_import.clicked.connect(panel.theory_local_import_requested)
    btn_theory_clear = QPushButton("Clear")
    btn_theory_clear.setToolTip("Removes the current DFT overlay.")
    btn_theory_clear.clicked.connect(panel.theory_clear_requested)
    btn_theory_compare = QPushButton("Compare to fit")
    btn_theory_compare.setToolTip(
        "Scores DFT bands against fitted kF points.\n"
        "Visual diagnostic only; does not modify the fit."
    )
    btn_theory_compare.clicked.connect(panel.theory_compare_requested)
    btn_self_energy = QPushButton("Re Σ(E)")
    btn_self_energy.setToolTip(
        "Computes Re Σ(E) = E_exp - E_DFT(k_exp) using the best DFT band\n"
        "against the current MDC fit, then opens a diagnostic plot."
    )
    btn_self_energy.clicked.connect(panel.self_energy_requested)
    btn_theory_mu_fit = QPushButton("Auto-fit μ")
    btn_theory_mu_fit.setToolTip(
        "Computes the μ that best aligns the selected DFT band to the\n"
        "fitted kF points (least squares, robust median). Writes Shift μ.\n"
        "Requires an MDC fit and imported DFT."
    )
    btn_theory_mu_fit.clicked.connect(panel.theory_mu_fit_requested)
    btn_theory_align = QPushButton("Align π/a")
    btn_theory_align.setToolTip(
        "Computes k scale and Δk to map the chosen segment to [0, 1] (π/a).\n"
        "First segment label -> 0, second -> 1."
    )
    btn_theory_align.clicked.connect(panel.theory_align_requested)
    # --- Manual high-symmetry-point calibration ---
    panel.cmb_theory_anchor_label = QComboBox()
    panel.cmb_theory_anchor_label.setToolTip(
        "High-symmetry point to place on the BM map.\n"
        "Pick it here, enable 'Pick on BM', then click where it sits on the band."
    )
    panel.btn_theory_anchor_pick = QPushButton("Pick on BM")
    panel.btn_theory_anchor_pick.setCheckable(True)
    panel.btn_theory_anchor_pick.setToolTip(
        "Click on the BM map to place the selected high-symmetry point at that k.\n"
        "Place ≥2 points (e.g. Γ and X), then 'Fit & align' to set scale + Δk."
    )
    panel.btn_theory_anchor_pick.toggled.connect(panel.theory_anchor_pick_toggled)
    panel.btn_theory_anchor_apply = QPushButton("Fit & align")
    panel.btn_theory_anchor_apply.setToolTip(
        "Fit k_scale + Δk from the placed points (linear fit ≥2 points;\n"
        "1 point anchors Δk only) and align the DFT onto them."
    )
    panel.btn_theory_anchor_apply.clicked.connect(panel.theory_anchor_apply_requested)
    panel.btn_theory_anchor_clear = QPushButton("Clear pts")
    panel.btn_theory_anchor_clear.setToolTip("Remove all placed high-symmetry points.")
    panel.btn_theory_anchor_clear.clicked.connect(panel.theory_anchor_clear_requested)
    btn_theory_efalign = QPushButton("Force μ=0")
    btn_theory_efalign.setToolTip(
        "Resets the chemical shift to μ = 0.\n"
        "Overlay: E = Z × E_DFT.\n"
        "This is a raw DFT reference, not a computed ARPES/DFT alignment."
    )
    btn_theory_efalign.clicked.connect(panel.theory_efalign_requested)
    def _btn_grid(buttons):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(3)
        for i, b in enumerate(buttons):
            b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            b.setMaximumWidth(170)
            g.addWidget(b, i // 3, i % 3)
        for c in range(3):
            g.setColumnStretch(c, 1)
        return w

    def _section(title):
        lbl = QLabel(title)
        lbl.setStyleSheet(
            "color:#9fc;font-size:10px;font-weight:bold;"
            "margin-top:6px;border-top:1px solid #444;padding-top:3px;"
        )
        return lbl

    # boutons regroupés par rôle (source vs diagnostic) au lieu d'une grille
    # fourre-tout : l'utilisateur retrouve l'action dans sa phase de travail.
    source_btns = _btn_grid([
        btn_theory_refresh, btn_theory_local_import, btn_theory_clear,
    ])
    align_btns = _btn_grid([btn_theory_mu_fit, btn_theory_align, btn_theory_efalign])
    diag_btns = _btn_grid([btn_theory_compare, btn_self_energy])

    panel.lbl_theory_status = QLabel("Visual guide only.")
    panel.lbl_theory_status.setWordWrap(True)
    panel.lbl_theory_status.setStyleSheet("color:#aaa;font-size:10px;")
    for w in (
        panel.chk_theory, panel.cmb_theory_segment, panel.cmb_theory_convention,
        panel.sp_theory_mu, panel.sp_theory_z,
        panel.sp_theory_dk, panel.sp_theory_kscale, panel.sp_theory_alpha,
        panel.sp_theory_max, panel.chk_theory_mirror,
    ):
        signal = w.stateChanged if isinstance(w, QCheckBox) else (
            w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
        )
        signal.connect(panel._schedule_theory_overlay_changed)

    # 1 · Source DFT
    fl_th.addRow(_section("1 · Source DFT"))
    fl_th.addRow(panel.chk_theory)
    fl_th.addRow("MP-ID:", mpid_row)
    fl_th.addRow("MP API key:", panel.txt_mp_api_key)
    fl_th.addRow("Convention:", panel.cmb_theory_convention)
    fl_th.addRow(source_btns)
    # 2 · Displayed Bands
    fl_th.addRow(_section("2 · Displayed Bands"))
    fl_th.addRow(panel.btn_theory_pick_bands)
    fl_th.addRow("Band idx:", panel.txt_theory_bands)
    fl_th.addRow("E_F window (eV):", panel.sp_theory_efwin)
    fl_th.addRow(panel.chk_theory_ef_only)
    fl_th.addRow("Max bands:", panel.sp_theory_max)
    fl_th.addRow(panel.chk_theory_color)
    fl_th.addRow(panel.chk_theory_projections)
    # 3 · DFT ↔ ARPES Alignment
    fl_th.addRow(_section("3 · DFT ↔ ARPES Alignment"))
    fl_th.addRow("Segment:", panel.cmb_theory_segment)
    fl_th.addRow("Shift μ (eV):", panel.sp_theory_mu)
    fl_th.addRow("Renorm Z:", panel.sp_theory_z)
    fl_th.addRow("Δk:", panel.sp_theory_dk)
    fl_th.addRow("Scale k:", panel.sp_theory_kscale)
    fl_th.addRow("Crystal a (Å):", panel.sp_crystal_a)
    fl_th.addRow(panel.chk_theory_mirror)
    fl_th.addRow(align_btns)
    # Manual calibration from user-placed high-symmetry points.
    anchor_pick_row = QWidget()
    apr_lay = QHBoxLayout(anchor_pick_row)
    apr_lay.setContentsMargins(0, 0, 0, 0)
    apr_lay.setSpacing(4)
    apr_lay.addWidget(panel.cmb_theory_anchor_label, 1)
    apr_lay.addWidget(panel.btn_theory_anchor_pick)
    fl_th.addRow("HS point:", anchor_pick_row)
    fl_th.addRow(_btn_grid([panel.btn_theory_anchor_apply, panel.btn_theory_anchor_clear]))
    # 4 · Rendering & Diagnostics
    fl_th.addRow(_section("4 · Rendering & Diagnostics"))
    fl_th.addRow("Opacity:", panel.sp_theory_alpha)
    fl_th.addRow(diag_btns)
    fl_th.addRow(panel.lbl_theory_status)
    lay.addWidget(panel._theory_widget)
