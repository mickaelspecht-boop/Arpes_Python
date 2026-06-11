"""Per-subfolder sample setup dialog, shown when a data folder is opened.

One row per top-level sample subfolder (FS scan-dataset folders are already
excluded by core/sample_layout). The user fills the work function φ and the
in-plane lattice parameter a; blank (0) means "unknown — the app will ask at
fit time", so Skip is always safe.

Council conditions honoured:
- QDoubleSpinBox everywhere (locale-proof: a French "4,2" cannot silently
  become 0);
- pre-filled values show their provenance (logbook/session) as a tooltip and
  are not overwritten on Apply unless the user actually edited them;
- the mode radio lets the user override the sample-vs-scan auto-detection.
"""
from __future__ import annotations

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_PHI_RANGE = (0.0, 30.0)   # eV; 0 = unknown
_A_RANGE = (0.0, 50.0)     # Å; 0 = unknown
_PHI_TYPICAL = (3.0, 7.0)  # outside -> orange highlight (warning, not error)


def _spin(lo: float, hi: float, decimals: int) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setLocale(QLocale(QLocale.Language.C))
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(0.1)
    sp.setSpecialValueText("—")  # 0 shown as "unknown"
    sp.setKeyboardTracking(False)
    return sp


class SampleSetupDialog(QDialog):
    """Work function + lattice parameter per sample subfolder."""

    def __init__(self, parent=None, *, folder_name: str = "",
                 subfolders: list[tuple[str, int]] | None = None,
                 n_root_files: int = 0,
                 detected_mode: str = "multi",
                 existing: dict[str, dict] | None = None,
                 session_default: dict | None = None):
        """`subfolders` = [(name, n_files)]; `existing` = session.sample_configs;
        `session_default` = session.current_sample (used to pre-fill)."""
        super().__init__(parent)
        self.setWindowTitle(f"Sample setup — {folder_name or 'folder'}")
        self._subfolders = list(subfolders or [])
        self._existing = dict(existing or {})
        self._default = dict(session_default or {})

        lay = QVBoxLayout(self)
        n_sub = len(self._subfolders)
        if n_sub:
            head = (f"Detected {n_sub} sample subfolder(s)"
                    + (f" + {n_root_files} file(s) at the folder root"
                       if n_root_files else "") + ".")
        else:
            head = ("No sample subfolders detected — the subfolders (if any) "
                    "look like scan datasets, so the whole folder is treated "
                    "as one sample.")
        intro = QLabel(
            head + "\nFill the work function φ and lattice parameter a for "
            "each sample.\nLeave a field on “—” (unknown): the app will ask "
            "when the value is actually needed."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # Mode override (auto-detection is a heuristic, never a hard rule).
        mode_row = QHBoxLayout()
        self.rb_single = QRadioButton("One sample for the whole folder")
        self.rb_multi = QRadioButton("One sample per subfolder")
        suffix = " (auto-detected)"
        if detected_mode == "single" or not self._subfolders:
            self.rb_single.setChecked(True)
            self.rb_single.setText(self.rb_single.text() + suffix)
        else:
            self.rb_multi.setChecked(True)
            self.rb_multi.setText(self.rb_multi.text() + suffix)
        self.rb_single.setToolTip(
            "Use when the subfolders are scans of the SAME sample\n"
            "(e.g. FS maps stored as folders of slices)."
        )
        self.rb_multi.setToolTip("Use when each subfolder is a different sample.")
        mode_row.addWidget(self.rb_single)
        mode_row.addWidget(self.rb_multi)
        mode_row.addStretch(1)
        lay.addLayout(mode_row)
        self.rb_single.toggled.connect(self._update_mode)

        # --- multi mode: one row per subfolder -----------------------------
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Sample", "φ (eV)", "a (Å)", "Files"])
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(160)
        self.table.setMaximumHeight(420)  # 40 samples: scroll, don't overflow screen
        self._row_spins: list[tuple[str, QDoubleSpinBox, QDoubleSpinBox]] = []
        rows = self._subfolders or []
        self.table.setRowCount(len(rows))
        for i, (name, n_files) in enumerate(rows):
            item = QTableWidgetItem(name + "/")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            sp_phi = _spin(*_PHI_RANGE, 2)
            sp_a = _spin(*_A_RANGE, 4)
            self._prefill(name, sp_phi, sp_a)
            sp_phi.valueChanged.connect(lambda _v, sp=sp_phi: self._tint_phi(sp))
            self._tint_phi(sp_phi)
            self.table.setCellWidget(i, 1, sp_phi)
            self.table.setCellWidget(i, 2, sp_a)
            cnt = QTableWidgetItem(str(n_files))
            cnt.setFlags(cnt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 3, cnt)
            self._row_spins.append((name, sp_phi, sp_a))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)

        self.btn_same = QPushButton("Same φ/a for all")
        self.btn_same.setToolTip("Copy the first row's values to every sample.")
        self.btn_same.clicked.connect(self._apply_same_for_all)
        lay.addWidget(self.btn_same, alignment=Qt.AlignmentFlag.AlignLeft)

        # --- single mode: one form ------------------------------------------
        self._single_box = QWidget()
        srow = QHBoxLayout(self._single_box)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(QLabel("φ (eV):"))
        self.sp_phi_single = _spin(*_PHI_RANGE, 2)
        srow.addWidget(self.sp_phi_single)
        srow.addWidget(QLabel("a (Å):"))
        self.sp_a_single = _spin(*_A_RANGE, 4)
        srow.addWidget(self.sp_a_single)
        srow.addStretch(1)
        self._prefill("", self.sp_phi_single, self.sp_a_single)
        self.sp_phi_single.valueChanged.connect(
            lambda _v: self._tint_phi(self.sp_phi_single))
        self._tint_phi(self.sp_phi_single)
        lay.addWidget(self._single_box)

        buttons = QDialogButtonBox()
        btn_skip = buttons.addButton(
            "Skip — configure later", QDialogButtonBox.ButtonRole.RejectRole)
        btn_skip.setToolTip(
            "Nothing is saved. The app keeps working and asks for φ/a at fit time.")
        btn_apply = buttons.addButton(
            "Apply && Continue", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_apply.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._update_mode()

    # ------------------------------------------------------------------ utils
    def _prefill(self, key: str, sp_phi: QDoubleSpinBox, sp_a: QDoubleSpinBox) -> None:
        """Pre-fill from existing per-folder config, else the session sample."""
        cfg = self._existing.get(key) or {}
        src = "saved sample setup" if cfg else "session/logbook default"
        base = cfg or self._default
        phi = float(base.get("work_function_eV", 0.0) or 0.0)
        a = float(base.get("a_angstrom", 0.0) or 0.0)
        if phi > 0:
            sp_phi.setValue(phi)
            sp_phi.setToolTip(f"Pre-filled from {src}")
        if a > 0:
            sp_a.setValue(a)
            sp_a.setToolTip(f"Pre-filled from {src}")

    @staticmethod
    def _tint_phi(sp: QDoubleSpinBox) -> None:
        v = float(sp.value())
        unusual = v > 0 and not (_PHI_TYPICAL[0] <= v <= _PHI_TYPICAL[1])
        sp.setStyleSheet("background:#7a5c1e;" if unusual else "")
        if unusual:
            sp.setToolTip("⚠ Unusual work function — typical range 3–7 eV")

    def _update_mode(self) -> None:
        single = self.rb_single.isChecked()
        self.table.setVisible(not single)
        self.btn_same.setVisible(not single)
        self._single_box.setVisible(single)
        # Carry values across modes so nothing typed is lost.
        if single and self._row_spins:
            name, sp_phi, sp_a = self._row_spins[0]
            if self.sp_phi_single.value() == 0 and sp_phi.value() > 0:
                self.sp_phi_single.setValue(sp_phi.value())
            if self.sp_a_single.value() == 0 and sp_a.value() > 0:
                self.sp_a_single.setValue(sp_a.value())
        elif not single and self._row_spins:
            for _name, sp_phi, sp_a in self._row_spins:
                if sp_phi.value() == 0 and self.sp_phi_single.value() > 0:
                    sp_phi.setValue(self.sp_phi_single.value())
                if sp_a.value() == 0 and self.sp_a_single.value() > 0:
                    sp_a.setValue(self.sp_a_single.value())

    def _apply_same_for_all(self) -> None:
        if not self._row_spins:
            return
        _n, first_phi, first_a = self._row_spins[0]
        for _name, sp_phi, sp_a in self._row_spins[1:]:
            sp_phi.setValue(first_phi.value())
            sp_a.setValue(first_a.value())

    # ----------------------------------------------------------------- output
    def result_configs(self) -> dict[str, dict]:
        """{sample_key: partial SampleConfig dict} — only non-zero values.

        Single mode returns {"": {...}} (whole folder = session-wide sample).
        """
        def cfg(phi: float, a: float) -> dict:
            out: dict = {}
            if phi > 0:
                out["work_function_eV"] = float(phi)
            if a > 0:
                out["a_angstrom"] = float(a)
            return out

        if self.rb_single.isChecked():
            c = cfg(self.sp_phi_single.value(), self.sp_a_single.value())
            return {"": c} if c else {}
        out: dict[str, dict] = {}
        for name, sp_phi, sp_a in self._row_spins:
            c = cfg(sp_phi.value(), sp_a.value())
            if c:
                out[name] = c
        return out
