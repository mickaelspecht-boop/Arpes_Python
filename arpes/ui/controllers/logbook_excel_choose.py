"""Interactive Excel sheet / header-row / column-mapping choosers.

Split out of ``logbook_controller.py`` to keep that file under the 700-LOC cap.
Free functions take the controller as first argument (``ctrl``) and use
``ctrl._parent`` (dialog parent) and ``ctrl._session`` (preferred-sheet hint),
mirroring the codified split pattern. Pure UI orchestration, no business logic.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from arpes.io.logbook import _cell_text, _norm_text


def choose_excel_sheet(ctrl, sheet_names: list[str]) -> str:
    if not sheet_names:
        return ""
    if len(sheet_names) == 1:
        return sheet_names[0]
    dlg = QDialog(ctrl._parent)
    dlg.setWindowTitle("Logbook sheet")
    lay = QVBoxLayout(dlg)
    label = QLabel("Choose the sheet that matches the compound / dataset.")
    label.setWordWrap(True)
    lay.addWidget(label)
    cmb = QComboBox()
    cmb.addItems(sheet_names)
    preferred = ""
    if ctrl._session.folder is not None:
        preferred = ctrl._session.folder.name
    preferred_norm = _norm_text(preferred)
    if ctrl._session.logbook_sheet in sheet_names:
        cmb.setCurrentText(ctrl._session.logbook_sheet)
    elif preferred in sheet_names:
        cmb.setCurrentText(preferred)
    else:
        for sheet in sheet_names:
            sheet_norm = _norm_text(sheet)
            if sheet_norm and sheet_norm in preferred_norm:
                cmb.setCurrentText(sheet)
                break
    lay.addWidget(cmb)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return ""
    return cmb.currentText()


def choose_excel_table(ctrl, raw, candidates: list[int]):
    from arpes.io.logbook_io import excel_table_from_header, _looks_like_title
    if not candidates:
        return None

    def score_row(row_idx: int) -> float:
        try:
            df, m = excel_table_from_header(raw, row_idx)
        except Exception:
            return -10.0
        s = int(bool(m.get("file"))) * 3 + int(bool(m.get("hv"))) * 3
        s += int(bool(m.get("temperature"))) + int(bool(m.get("polarization")))
        s += int(bool(m.get("direction"))) + int(bool(m.get("azi")))
        s += int(bool(m.get("polar"))) + int(bool(m.get("tilt")))
        if _looks_like_title(m.get("file", "")):
            s -= 5
        if _looks_like_title(m.get("hv", "")):
            s -= 5
        s += min(len(df), 30) / 1000
        return s

    scored = sorted(candidates, key=score_row, reverse=True)
    dlg = QDialog(ctrl._parent)
    dlg.setWindowTitle("Logbook header row")
    lay = QVBoxLayout(dlg)
    label = QLabel(
        "Choose the row that contains the real column names "
        "(sorted by relevance - the best guess is at the top)."
    )
    label.setWordWrap(True)
    lay.addWidget(label)
    cmb = QComboBox()
    for row_idx in scored:
        values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
        preview = " | ".join(v for v in values if v)
        score = score_row(row_idx)
        tag = "✓" if score >= 6 else ("?" if score >= 3 else "✗")
        cmb.addItem(f"{tag} Row {row_idx + 1}: {preview[:140]}", row_idx)
    lay.addWidget(cmb)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return excel_table_from_header(raw, int(cmb.currentData()))


def choose_logbook_mapping(ctrl, columns: list[str], mapping: dict[str, str]) -> dict[str, str]:
    dlg = QDialog(ctrl._parent)
    dlg.setWindowTitle("Logbook columns")
    lay = QFormLayout(dlg)
    combos: dict[str, QComboBox] = {}
    labels = {
        "file": "File / scan:",
        "hv": "hν:",
        "temperature": "Temperature:",
        "polarization": "Polarization:",
        "direction": "Direction / chemin:",
        "azi": "Azimut:",
        "polar": "Polar / theta manip:",
        "tilt": "Tilt / phi manip:",
        "crystal_a_angstrom": "a lattice (Å):",
        "crystal_b_angstrom": "b lattice (Å):",
        "crystal_c_angstrom": "c lattice (Å):",
        "work_function_eV": "φ work function (eV):",
    }
    choices = [""] + columns
    for key, label in labels.items():
        cmb = QComboBox()
        cmb.addItems(choices)
        current = mapping.get(key, "")
        if current in choices:
            cmb.setCurrentText(current)
        lay.addRow(label, cmb)
        combos[key] = cmb
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addRow(buttons)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return mapping
    return {key: cmb.currentText() for key, cmb in combos.items()}
