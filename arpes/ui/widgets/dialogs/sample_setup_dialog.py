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

from pathlib import Path

from PyQt6.QtCore import QLocale, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_LOGBOOK_NONE = "None"
_LOGBOOK_OTHER = "Other file…"


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _sheet_names_safe(path: str) -> tuple[list[str], str]:
    """(sheets, "") on success, ([], reason) on unreadable file."""
    if not path:
        return [], ""
    try:
        from arpes.io.logbook_io import get_xlsx_sheet_names
        return get_xlsx_sheet_names(path), ""
    except Exception as exc:
        return [], str(exc)

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
                 session_default: dict | None = None,
                 folder_path: str = "",
                 existing_logbooks: dict[str, dict] | None = None):
        """`subfolders` = [(name, n_files)]; `existing` = session.sample_configs;
        `session_default` = session.current_sample; `folder_path` enables xlsx
        auto-detection; `existing_logbooks` = session.scoped_logbooks (prefill)."""
        super().__init__(parent)
        self.setWindowTitle(f"Sample & logbook setup — {folder_name or 'folder'}")
        self._subfolders = list(subfolders or [])
        self._existing = dict(existing or {})
        self._default = dict(session_default or {})
        self._folder_path = str(folder_path or "")
        self._existing_logbooks = dict(existing_logbooks or {})
        self.browse_only_requested = False
        self._row_logbooks: list[tuple[str, QComboBox]] = []
        self._row_files: dict[str, tuple[str, str]] = {}  # rel -> (path, sheet) mode B

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

        # --- global Excel logbook (optional) --------------------------------
        self._grp_logbook = QGroupBox("Logbook (optional)")
        gl = QHBoxLayout(self._grp_logbook)
        gl.addWidget(QLabel("Excel file:"))
        self.ed_logbook = QLineEdit()
        self.ed_logbook.setReadOnly(True)
        self.ed_logbook.setPlaceholderText("no logbook — pick a .xlsx to enable per-sample sheets")
        gl.addWidget(self.ed_logbook, stretch=1)
        btn_lb = QPushButton("Browse…")
        btn_lb.setToolTip(
            "One Excel file with one sheet per sample: sheets are matched to\n"
            "samples by name automatically. Per-sample files: use “Other\n"
            "file…” in a row's Logbook column instead."
        )
        btn_lb.clicked.connect(self._pick_global_logbook)
        gl.addWidget(btn_lb)
        lay.addWidget(self._grp_logbook)
        self._global_sheets: list[str] = []
        self._autodetect_global_logbook()

        # --- multi mode: one row per subfolder -----------------------------
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Sample", "φ (eV)", "a (Å)", "Files", "Logbook"])
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
            cmb = QComboBox()
            cmb.activated.connect(lambda _i, rel=name, c=None: self._on_logbook_combo(rel))
            self.table.setCellWidget(i, 4, cmb)
            self._row_logbooks.append((name, cmb))
        self._refresh_logbook_combos()
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
        srow.addWidget(QLabel("Sheet:"))
        self.cmb_sheet_single = QComboBox()
        srow.addWidget(self.cmb_sheet_single)
        lay.addWidget(self._single_box)

        buttons = QDialogButtonBox()
        btn_browse = buttons.addButton(
            "Browse only", QDialogButtonBox.ButtonRole.ResetRole)
        btn_browse.setToolTip(
            "Just look at the data: close and stop asking for sample/logbook\n"
            "setup in this session. Everything stays configurable later via\n"
            "the “Samples…” button. Fits will still ask for φ/a when needed.")
        btn_browse.clicked.connect(self._on_browse_only)
        btn_skip = buttons.addButton(
            "Skip — configure later", QDialogButtonBox.ButtonRole.RejectRole)
        btn_skip.setToolTip(
            "Nothing is saved. The app keeps working, asks again on next\n"
            "folder open, and asks for φ/a at fit time.")
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
        if hasattr(self, "cmb_sheet_single"):
            self._refresh_single_sheet_combo()
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

    # ------------------------------------------------------------- logbook
    def _autodetect_global_logbook(self) -> None:
        """Pre-select: previously saved global path, else lone xlsx at root."""
        saved = {str(c.get("path", "")) for c in self._existing_logbooks.values()}
        saved = {p for p in saved if p}
        if len(saved) == 1:
            self.ed_logbook.setText(next(iter(saved)))
        elif self._folder_path:
            xlsx = sorted(Path(self._folder_path).glob("*.xlsx"))
            xlsx = [x for x in xlsx if not x.name.startswith("~$")]
            if len(xlsx) == 1:
                self.ed_logbook.setText(str(xlsx[0]))
                self.ed_logbook.setToolTip("Auto-detected in the folder")
        self._reload_global_sheets()

    def _pick_global_logbook(self) -> None:
        start = self._folder_path or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel logbook", start, "Excel (*.xlsx *.xls *.xlsm)")
        if not path:
            return
        self.ed_logbook.setText(path)
        self._reload_global_sheets()
        self._refresh_logbook_combos()

    def _reload_global_sheets(self) -> None:
        self._global_sheets, err = _sheet_names_safe(self.ed_logbook.text())
        if err:
            self.ed_logbook.setStyleSheet("color:#e05c5c;")
            self.ed_logbook.setToolTip(f"⚠ Cannot read sheets: {err}")
        else:
            self.ed_logbook.setStyleSheet("")

    def _saved_logbook_for(self, rel: str) -> tuple[str, str]:
        cfg = self._existing_logbooks.get(rel) or {}
        return str(cfg.get("path", "") or ""), str(cfg.get("sheet", "") or "")

    def _refresh_logbook_combos(self) -> None:
        """(Re)build every row combo: None | sheets of the global xlsx |
        Other file…; pre-select saved sheet or name-matched sheet."""
        for rel, cmb in self._row_logbooks:
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem(_LOGBOOK_NONE)
            for sheet in self._global_sheets:
                cmb.addItem(f"Sheet: {sheet}", sheet)
            cmb.addItem(_LOGBOOK_OTHER)
            saved_path, saved_sheet = self._saved_logbook_for(rel)
            if rel in self._row_files:
                path, sheet = self._row_files[rel]
                cmb.insertItem(cmb.count() - 1, f"{Path(path).name} [{sheet}]",
                               ("file", path, sheet))
                cmb.setCurrentIndex(cmb.count() - 2)
                cmb.setToolTip(path)
            elif saved_path and saved_path == self.ed_logbook.text() and saved_sheet:
                idx = cmb.findData(saved_sheet)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
                else:
                    cmb.setCurrentIndex(0)
                    cmb.setToolTip(
                        f"⚠ Previously used sheet '{saved_sheet}' not found "
                        "in the current file — reselect.")
            elif saved_path and saved_sheet:
                self._row_files[rel] = (saved_path, saved_sheet)
                cmb.insertItem(cmb.count() - 1,
                               f"{Path(saved_path).name} [{saved_sheet}]",
                               ("file", saved_path, saved_sheet))
                cmb.setCurrentIndex(cmb.count() - 2)
                cmb.setToolTip(saved_path)
            else:
                target = _norm(rel)
                hit = next((sh for sh in self._global_sheets
                            if _norm(sh) and (_norm(sh) in target or target in _norm(sh))), None)
                if hit is not None:
                    cmb.setCurrentIndex(cmb.findData(hit))
            cmb.blockSignals(False)
        self._refresh_single_sheet_combo()

    def _refresh_single_sheet_combo(self) -> None:
        if not hasattr(self, "cmb_sheet_single"):
            return  # constructor order: table is built before the single box
        cmb = self.cmb_sheet_single
        cmb.clear()
        cmb.addItem(_LOGBOOK_NONE)
        for sheet in self._global_sheets:
            cmb.addItem(f"Sheet: {sheet}", sheet)
        saved_path, saved_sheet = self._saved_logbook_for("")
        idx = cmb.findData(saved_sheet) if saved_sheet else -1
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        elif len(self._global_sheets) == 1:
            cmb.setCurrentIndex(1)

    def _on_logbook_combo(self, rel: str) -> None:
        """Handle “Other file…”: pick a per-sample file + auto sheet."""
        cmb = next(c for r, c in self._row_logbooks if r == rel)
        if cmb.currentText() != _LOGBOOK_OTHER:
            if not isinstance(cmb.currentData(), tuple):
                self._row_files.pop(rel, None)  # back to None / global sheet
            return
        start = self._folder_path or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, f"Logbook for {rel}", start, "Excel (*.xlsx *.xls *.xlsm)")
        if not path:
            cmb.setCurrentIndex(0)
            return
        sheets, err = _sheet_names_safe(path)
        if err:
            cmb.setCurrentIndex(0)
            cmb.setToolTip(f"⚠ Cannot read sheets: {err}")
            return
        target = _norm(rel)
        sheet = next((sh for sh in sheets
                      if _norm(sh) and (_norm(sh) in target or target in _norm(sh))),
                     sheets[0] if sheets else "")
        self._row_files[rel] = (path, sheet)
        self._refresh_logbook_combos()

    def _on_browse_only(self) -> None:
        self.browse_only_requested = True
        self.reject()

    def result_logbooks(self) -> list[dict]:
        """[{rel, path, sheet}] for every row with a logbook selected.

        Single mode: one entry with rel "" (session-wide logbook).
        """
        out: list[dict] = []
        if self.rb_single.isChecked():
            sheet = self.cmb_sheet_single.currentData()
            path = self.ed_logbook.text()
            if sheet and path:
                out.append({"rel": "", "path": path, "sheet": str(sheet)})
            return out
        for rel, cmb in self._row_logbooks:
            data = cmb.currentData()
            if isinstance(data, tuple) and data and data[0] == "file":
                out.append({"rel": rel, "path": data[1], "sheet": data[2]})
            elif isinstance(data, str) and data and self.ed_logbook.text():
                out.append({"rel": rel, "path": self.ed_logbook.text(),
                            "sheet": data})
        return out

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
