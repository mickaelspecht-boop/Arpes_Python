"""Materials Project dialog: search MPID by chemical formula.

Runs the network search in a QThread so the UI is not blocked for 2-5 s.
The dialog imports no controller: it emits `mpid_selected(str)`, which
the parent widget connects to `txt_theory_mpid`.
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
from arpes.ui.app_settings import resolve_mp_api_key


_ACTIVE_SEARCHES: set[tuple[QThread, QObject]] = set()


class _SearchWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, formula: str, max_results: int = 25):
        super().__init__()
        self._formula = formula
        self._max_results = max_results

    def run(self) -> None:
        try:
            results = search_by_formula(self._formula, max_results=self._max_results,
                                        api_key=resolve_mp_api_key())
            self.finished.emit(results)
        except MaterialsProjectUnavailable as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Search failed: {exc}")


class MPSearchDialog(QDialog):
    mpid_selected = pyqtSignal(str)

    COLUMNS = ("MPID", "Formula", "System", "Space group", "ΔE hull (eV/atom)", "Stable")

    def __init__(self, parent=None, initial_formula: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Materials Project — formula search")
        self.resize(720, 480)
        self._thread: QThread | None = None
        self._worker: _SearchWorker | None = None
        self._search_token = 0
        self._build(initial_formula)

    def _build(self, initial_formula: str) -> None:
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Formula:"))
        self.txt_formula = QLineEdit(initial_formula)
        self.txt_formula.setPlaceholderText("BaNi2As2")
        self.txt_formula.returnPressed.connect(self._start_search)
        row.addWidget(self.txt_formula, 1)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self._start_search)
        row.addWidget(self.btn_search)
        lay.addLayout(row)

        self.lbl_status = QLabel("Type a chemical formula, then press Enter.")
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
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Select")
        bb.accepted.connect(self._accept_selection)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _start_search(self) -> None:
        formula = self.txt_formula.text().strip()
        if not formula:
            self.lbl_status.setText("Warning: empty formula.")
            return
        self._cancel_thread()
        self._search_token += 1
        token = self._search_token
        self.btn_search.setEnabled(False)
        self.lbl_status.setText(f"Searching {formula} on Materials Project ...")
        self.tbl.setRowCount(0)

        self._thread = QThread()
        self._worker = _SearchWorker(formula)
        self._worker.moveToThread(self._thread)
        _ACTIVE_SEARCHES.add((self._thread, self._worker))
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(lambda results, t=token: self._on_results(t, results))
        self._worker.failed.connect(lambda message, t=token: self._on_failed(t, message))
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(
            lambda th=self._thread, wk=self._worker, t=token: self._cleanup_thread(th, wk, t)
        )
        self._thread.start()

    def _on_results(self, token: int, results: list) -> None:
        if token != self._search_token:
            return
        self.btn_search.setEnabled(True)
        if not results:
            self.lbl_status.setText("No results. Check the formula (case-sensitive, Ba2NiAs2 ≠ BaNi2As2).")
            return
        self.lbl_status.setText(f"{len(results)} candidate(s). Double-click or Select.")
        self.tbl.setRowCount(len(results))
        for row, r in enumerate(results):
            cells = [
                r["material_id"],
                r["formula_pretty"],
                r["crystal_system"],
                r["spacegroup_symbol"],
                f"{r['energy_above_hull']:.4f}",
                "Yes" if r["is_stable"] else "No",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, r["material_id"])
                self.tbl.setItem(row, col, item)
        self.tbl.selectRow(0)

    def _on_failed(self, token: int, message: str) -> None:
        if token != self._search_token:
            return
        self.btn_search.setEnabled(True)
        self.lbl_status.setText(f"Error: {message}")

    def _on_row_double_click(self, _item: QTableWidgetItem) -> None:
        self._accept_selection()

    def _accept_selection(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            self.lbl_status.setText("Select a row first.")
            return
        cell = self.tbl.item(row, 0)
        mpid = str(cell.data(Qt.ItemDataRole.UserRole) or cell.text() or "").strip()
        if not mpid:
            return
        self.mpid_selected.emit(mpid)
        self.accept()

    def _cancel_thread(self) -> None:
        self._search_token += 1
        thread = self._thread
        worker = self._worker
        self._thread = None
        self._worker = None
        if thread is None:
            return
        if worker is not None:
            for signal in (worker.finished, worker.failed):
                try:
                    signal.disconnect()
                except Exception:
                    pass
            try:
                worker.finished.connect(thread.quit)
                worker.failed.connect(thread.quit)
            except Exception:
                pass
        try:
            thread.quit()
        except Exception:
            pass
        try:
            self.btn_search.setEnabled(True)
        except Exception:
            pass

    def _cleanup_thread(self, thread: QThread, worker: QObject, token: int) -> None:
        _ACTIVE_SEARCHES.discard((thread, worker))
        if token == self._search_token:
            self._worker = None
            self._thread = None

    def reject(self) -> None:
        self._cancel_thread()
        super().reject()

    def closeEvent(self, event) -> None:
        self._cancel_thread()
        super().closeEvent(event)
