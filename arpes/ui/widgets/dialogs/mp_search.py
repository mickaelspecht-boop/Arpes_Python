"""Dialog Materials Project — recherche MPID par formule chimique.

Lance la recherche réseau dans un QThread pour ne pas bloquer l'UI 2-5s.
Le dialog n'importe aucun controller : il émet `mpid_selected(str)` que
le widget parent connecte vers `txt_theory_mpid`.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from arpes.theory.materials_project import (
    MaterialsProjectUnavailable,
    search_by_formula,
)


class _SearchWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, formula: str, max_results: int = 25):
        super().__init__()
        self._formula = formula
        self._max_results = max_results

    def run(self) -> None:
        try:
            results = search_by_formula(self._formula, max_results=self._max_results)
            self.finished.emit(results)
        except MaterialsProjectUnavailable as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Recherche échouée: {exc}")


class MPSearchDialog(QDialog):
    mpid_selected = pyqtSignal(str)

    COLUMNS = ("MPID", "Formule", "Système", "Groupe d'espace", "ΔE hull (eV/atom)", "Stable")

    def __init__(self, parent=None, initial_formula: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Materials Project — recherche par formule")
        self.resize(720, 480)
        self._thread: QThread | None = None
        self._worker: _SearchWorker | None = None
        self._build(initial_formula)

    def _build(self, initial_formula: str) -> None:
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Formule:"))
        self.txt_formula = QLineEdit(initial_formula)
        self.txt_formula.setPlaceholderText("BaNi2As2")
        self.txt_formula.returnPressed.connect(self._start_search)
        row.addWidget(self.txt_formula, 1)
        self.btn_search = QPushButton("Chercher")
        self.btn_search.clicked.connect(self._start_search)
        row.addWidget(self.btn_search)
        lay.addLayout(row)

        self.lbl_status = QLabel("Tape une formule chimique puis Entrée.")
        self.lbl_status.setStyleSheet("color:#9fc;font-size:10px;")
        lay.addWidget(self.lbl_status)

        self.tbl = QTableWidget(0, len(self.COLUMNS))
        self.tbl.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.itemDoubleClicked.connect(self._on_row_double_click)
        lay.addWidget(self.tbl, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Sélectionner")
        bb.accepted.connect(self._accept_selection)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _start_search(self) -> None:
        formula = self.txt_formula.text().strip()
        if not formula:
            self.lbl_status.setText("Attention : formule vide.")
            return
        self._stop_thread()
        self.btn_search.setEnabled(False)
        self.lbl_status.setText(f"Recherche {formula} sur Materials Project ...")
        self.tbl.setRowCount(0)

        self._thread = QThread(self)
        self._worker = _SearchWorker(formula)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_results)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_results(self, results: list) -> None:
        self.btn_search.setEnabled(True)
        if not results:
            self.lbl_status.setText("Aucun résultat. Vérifie la formule (sensible au cas, Ba2NiAs2 ≠ BaNi2As2).")
            return
        self.lbl_status.setText(f"{len(results)} candidat(s). Double-clic ou Sélectionner.")
        self.tbl.setRowCount(len(results))
        for row, r in enumerate(results):
            cells = [
                r["material_id"],
                r["formula_pretty"],
                r["crystal_system"],
                r["spacegroup_symbol"],
                f"{r['energy_above_hull']:.4f}",
                "Oui" if r["is_stable"] else "Non",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, r["material_id"])
                self.tbl.setItem(row, col, item)
        self.tbl.selectRow(0)

    def _on_failed(self, message: str) -> None:
        self.btn_search.setEnabled(True)
        self.lbl_status.setText(f"Erreur : {message}")

    def _on_row_double_click(self, _item: QTableWidgetItem) -> None:
        self._accept_selection()

    def _accept_selection(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            self.lbl_status.setText("Sélectionne une ligne d'abord.")
            return
        cell = self.tbl.item(row, 0)
        mpid = str(cell.data(Qt.ItemDataRole.UserRole) or cell.text() or "").strip()
        if not mpid:
            return
        self.mpid_selected.emit(mpid)
        self.accept()

    def _stop_thread(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        self._worker = None
        self._thread = None

    def closeEvent(self, event) -> None:
        self._stop_thread()
        super().closeEvent(event)
