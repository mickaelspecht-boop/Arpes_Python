"""Dialog for choosing the high-symmetry label convention of the BZ overlay.

Display-only renames (the geometry never changes): a preset combo fills the
table, and each label stays editable so any article convention can be
reproduced. Duplicate labels are rejected — the logbook direction matching
("Γ-Σ"…) resolves labels by name and must stay unambiguous.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from arpes.physics.bz import (
    BZ_LABEL_CONVENTION_PRESETS,
    BZ_LABEL_CONVENTION_TITLES,
    bz_high_symmetry_points,
)


class BZLabelsDialog(QDialog):
    """Preset + per-point renames for the theoretical BZ labels."""

    def __init__(self, parent=None, *, shape: str = "square",
                 half_x: float = 1.0, half_y: float = 1.0,
                 angle_deg: float = 90.0,
                 current_overrides: dict | None = None,
                 current_preset: str = ""):
        super().__init__(parent)
        self.setWindowTitle("BZ label conventions")
        self._canonical = self._canonical_labels(shape, half_x, half_y, angle_deg)

        lay = QVBoxLayout(self)
        info = QLabel(
            "Rename the high-symmetry points to match an article's convention.\n"
            "Display-only: positions are unchanged. Logbook directions (Γ-Σ…)\n"
            "match the renamed labels."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self.cmb_preset = QComboBox()
        self.cmb_preset.addItem("Custom", "")
        for key in BZ_LABEL_CONVENTION_PRESETS:
            self.cmb_preset.addItem(BZ_LABEL_CONVENTION_TITLES.get(key, key), key)
        form = QFormLayout()
        form.addRow("Preset:", self.cmb_preset)

        overrides = dict(current_overrides or {})
        self._edits: dict[str, QLineEdit] = {}
        for lab in self._canonical:
            ed = QLineEdit(overrides.get(lab, lab))
            ed.setMaxLength(4)
            ed.textChanged.connect(self._validate)
            self._edits[lab] = ed
            form.addRow(f"{lab} →", ed)
        lay.addLayout(form)

        self._lbl_error = QLabel("")
        self._lbl_error.setStyleSheet("color:#e05c5c;")
        lay.addWidget(self._lbl_error)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        if current_preset and self.cmb_preset.findData(current_preset) >= 0:
            self.cmb_preset.setCurrentIndex(self.cmb_preset.findData(current_preset))
        self.cmb_preset.currentIndexChanged.connect(self._apply_preset)
        self._validate()

    @staticmethod
    def _canonical_labels(shape, half_x, half_y, angle_deg) -> list[str]:
        """Distinct canonical labels of the zone, Γ excluded (never renamed)."""
        seen: list[str] = []
        for _x, _y, name, _c in bz_high_symmetry_points(shape, half_x, half_y, angle_deg):
            if name and name != "Γ" and name not in seen:
                seen.append(name)
        return seen

    def _apply_preset(self) -> None:
        key = self.cmb_preset.currentData()
        preset = BZ_LABEL_CONVENTION_PRESETS.get(key or "", None)
        if preset is None:
            return  # "Custom": leave fields as they are
        for lab, ed in self._edits.items():
            ed.blockSignals(True)
            ed.setText(preset.get(lab, lab))
            ed.blockSignals(False)
        self._validate()

    def _validate(self) -> None:
        values = [ed.text().strip() for ed in self._edits.values()]
        ok = True
        msg = ""
        if any(not v for v in values):
            ok, msg = False, "Empty label."
        elif len(set(values)) != len(values):
            ok, msg = False, "Duplicate label — each point needs a unique name."
        self._lbl_error.setText(msg)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def overrides(self) -> dict[str, str]:
        """Renames differing from the canonical labels (identity dropped)."""
        out: dict[str, str] = {}
        for lab, ed in self._edits.items():
            val = ed.text().strip()
            if val and val != lab:
                out[lab] = val
        return out

    def preset_key(self) -> str:
        return str(self.cmb_preset.currentData() or "")
