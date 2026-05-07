"""Dialog de comparaison de deux sessions JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from arpes.analysis.session_diff import SessionDiffRow, compare_session_payloads

SESSION_FILTER = "ARPES Session (*.arpes-session.json *.json)"


class SessionDiffDialog(QDialog):
    """Compare deux fichiers de session sauvegardes."""

    HEADERS = [
        "Fichier", "Paire/Branche", "Statut", "Delta kF", "Delta vF",
        "Delta m*", "A kF±sigma", "B kF±sigma", "A vF±sigma", "B vF±sigma",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparer sessions")
        self.resize(980, 560)
        self._path_a: Path | None = None
        self._path_b: Path | None = None

        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self._lbl_a = QLabel("Session A : aucune")
        self._lbl_b = QLabel("Session B : aucune")
        btn_a = QPushButton("Session A...")
        btn_b = QPushButton("Session B...")
        btn_load = QPushButton("Charger")
        btn_a.clicked.connect(lambda: self._choose_path("A"))
        btn_b.clicked.connect(lambda: self._choose_path("B"))
        btn_load.clicked.connect(self.load_diff)
        row.addWidget(btn_a)
        row.addWidget(self._lbl_a, stretch=1)
        row.addWidget(btn_b)
        row.addWidget(self._lbl_b, stretch=1)
        row.addWidget(btn_load)
        root.addLayout(row)

        self._warning = QLabel("")
        self._warning.setStyleSheet("color:#b45309;")
        root.addWidget(self._warning)

        self._table = QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSortingEnabled(True)
        root.addWidget(self._table, stretch=1)

    def _choose_path(self, which: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choisir session {which}", str(Path.home()), SESSION_FILTER,
        )
        if not path:
            return
        p = Path(path)
        if which == "A":
            self._path_a = p
            self._lbl_a.setText(p.name)
            self._lbl_a.setToolTip(str(p))
        else:
            self._path_b = p
            self._lbl_b.setText(p.name)
            self._lbl_b.setToolTip(str(p))

    def load_diff(self) -> None:
        if self._path_a is None or self._path_b is None:
            QMessageBox.warning(self, "Comparer sessions", "Choisir les sessions A et B.")
            return
        try:
            payload_a = json.loads(self._path_a.read_text())
            payload_b = json.loads(self._path_b.read_text())
            rows = compare_session_payloads(payload_a, payload_b)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.critical(self, "Comparer sessions", f"Lecture impossible : {exc}")
            return
        self._populate(rows)
        self._warning.setText(_crystal_warning(payload_a, payload_b))

    def _populate(self, rows: list[SessionDiffRow]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row.filename,
                _branch_label(row),
                row.status,
                _fmt(row.delta_kF, dec=5),
                _fmt(row.delta_vF, dec=4),
                _fmt(row.delta_m_star, dec=4),
                _fmt_pm(row.a_kF, row.a_kF_sigma, dec=5),
                _fmt_pm(row.b_kF, row.b_kF_sigma, dec=5),
                _fmt_pm(row.a_vF, row.a_vF_sigma, dec=4),
                _fmt_pm(row.b_vF, row.b_vF_sigma, dec=4),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)


def _branch_label(row: SessionDiffRow) -> str:
    if row.pair_index < 0 or not row.branch:
        return ""
    return f"P{row.pair_index + 1} / {row.branch}"


def _fmt(value: float, *, dec: int) -> str:
    return f"{value:.{dec}f}" if np.isfinite(value) else "-"


def _fmt_pm(value: float, sigma: float, *, dec: int) -> str:
    if not (np.isfinite(value) and np.isfinite(sigma)):
        return "-"
    return f"{value:.{dec}f} +/- {sigma:.{dec}f}"


def _crystal_warning(payload_a: dict, payload_b: dict) -> str:
    files_a = payload_a.get("files") or {}
    files_b = payload_b.get("files") or {}
    changed = []
    for name in sorted(set(files_a) & set(files_b)):
        a_val = ((files_a[name].get("meta") or {}).get("crystal_a_angstrom") or 0.0)
        b_val = ((files_b[name].get("meta") or {}).get("crystal_a_angstrom") or 0.0)
        try:
            a_float = float(a_val)
            b_float = float(b_val)
        except (TypeError, ValueError):
            continue
        if abs(a_float - b_float) > 1e-9:
            changed.append(name)
    if not changed:
        return ""
    return f"Attention : crystal_a different pour {len(changed)} fichier(s), chaque session utilise sa valeur."
