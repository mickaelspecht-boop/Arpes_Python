"""Live, dockable processing-log panel.

Shows the chronological provenance journal of the current signal and refreshes
itself in real time. Fully decoupled from the instrumentation: it polls the
current entry's ``processing_history`` (cheap length check) instead of being
notified, so any controller that calls ``processing_history.log_event`` shows up
here automatically, with no UI wiring at the mutation site.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from arpes.core import processing_history as ph
from arpes.core.experience_log import build_full_report

_CAT_LABELS = {
    ph.CAT_LOAD: "Load",
    ph.CAT_ENERGY: "Energy",
    ph.CAT_GAMMA: "Gamma",
    ph.CAT_NORM: "Norm",
    ph.CAT_GRID: "Grid",
    ph.CAT_DISTORT: "Distort",
    ph.CAT_FS: "FS",
    ph.CAT_KZ: "kz",
    ph.CAT_FIT: "Fit",
    ph.CAT_ZONE: "Zone",
    ph.CAT_KF: "kF",
    ph.CAT_BAND: "Band",
    ph.CAT_THEORY: "Theory",
    ph.CAT_EDIT: "Edit",
}


class ProcessingLogDock(QDockWidget):
    """Always-available live view of the current signal's processing journal."""

    def __init__(self, window):
        super().__init__("Processing log", window)
        self._window = window
        self.setObjectName("ProcessingLogDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._key: str | None = None
        self._count: int = -1
        self._build()
        # Poll for changes so the panel stays live without coupling every
        # mutation site to the UI (length check is O(1)).
        self._timer = QTimer(self)
        self._timer.setInterval(600)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        body = QWidget(self)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self._title = QLabel("No signal")
        self._title.setStyleSheet("font-weight:600; color:#e5e7eb;")
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("Filter:"))
        self._filter = QComboBox()
        self._filter.addItem("All steps", "")
        for cat in ph.CATEGORIES:
            self._filter.addItem(_CAT_LABELS.get(cat, cat), cat)
        self._filter.currentIndexChanged.connect(lambda *_: self._render())
        filt.addWidget(self._filter, 1)
        lay.addLayout(filt)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setStyleSheet(
            "QListWidget { background:#161616; color:#dcdcdc; font-family:monospace;"
            " font-size:11px; border:1px solid #2a2a2a; }"
            " QListWidget::item { padding:1px 2px; }"
            " QListWidget::item:alternate { background:#1c1c1c; }"
        )
        lay.addWidget(self._list, 1)

        btns = QHBoxLayout()
        self._btn_report = QPushButton("Full report…")
        self._btn_report.setToolTip("Open the full per-signal report (timeline + state) dialog.")
        self._btn_report.clicked.connect(self._open_report)
        self._btn_export = QPushButton("Export .md")
        self._btn_export.setToolTip("Save this signal's timeline + state snapshot as Markdown.")
        self._btn_export.clicked.connect(self._export)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Erase the recorded journal for this signal (cannot be undone).")
        self._btn_clear.clicked.connect(self._clear)
        for b in (self._btn_report, self._btn_export, self._btn_clear):
            btns.addWidget(b)
        lay.addLayout(btns)

        self.setWidget(body)

    # ---------------------------------------------------------------- helpers
    def _current(self):
        w = self._window
        path = getattr(w, "_current_path", None)
        if not path:
            return None, None
        try:
            key = w._session.key_for_path(path)
            return key, w._session.files.get(key)
        except Exception:
            return None, None

    def _poll(self) -> None:
        if not self.isVisible():
            return
        key, entry = self._current()
        count = ph.event_count(entry) if entry else 0
        if key != self._key or count != self._count:
            self._render()

    def refresh(self) -> None:
        """Force an immediate re-render (called on file switch)."""
        self._render()

    def _render(self) -> None:
        key, entry = self._current()
        self._key = key
        self._count = ph.event_count(entry) if entry else 0
        self._title.setText(f"{key or 'No signal'}  ·  {self._count} event(s)")
        self._list.clear()
        if entry is None:
            return
        cat_filter = self._filter.currentData()
        for ev in (getattr(entry, "processing_history", []) or []):
            if cat_filter and ev.get("category") != cat_filter:
                continue
            self._list.addItem(QListWidgetItem(self._fmt_event(ev)))
        self._list.scrollToBottom()

    @staticmethod
    def _fmt_event(ev: dict) -> str:
        ts = str(ev.get("ts", "")).replace("T", " ").replace("Z", "")
        clock = ts.split(" ")[-1] if " " in ts else ts
        cat = _CAT_LABELS.get(ev.get("category", ""), str(ev.get("category", "")).upper())
        action = str(ev.get("action", ""))
        detail = str(
            ev.get("summary")
            or ", ".join(f"{k}={v}" for k, v in (ev.get("params") or {}).items())
        )
        head = f"{clock}  {cat:<7} {action}"
        return f"{head}  —  {detail}" if detail else head

    # ------------------------------------------------------------------ slots
    def _open_report(self) -> None:
        try:
            self._window._experience_log_ctrl.open_dialog()
        except Exception:
            pass

    def _export(self) -> None:
        key, entry = self._current()
        if entry is None:
            QMessageBox.information(self, "Processing log", "No signal selected.")
            return
        suggested = Path(str(key).replace("/", "_")).with_suffix(".processing-log.md")
        folder = getattr(getattr(self._window, "_session", None), "folder", None)
        if folder:
            suggested = Path(folder) / suggested
        path, _ = QFileDialog.getSaveFileName(
            self, "Export processing log", str(suggested),
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(build_full_report(entry, name=key), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Processing log", f"Write failed: {exc}")

    def _clear(self) -> None:
        key, entry = self._current()
        if entry is None:
            return
        if QMessageBox.question(
            self, "Processing log",
            f"Erase the recorded journal for {key}?",
        ) != QMessageBox.StandardButton.Yes:
            return
        ph.clear_history(entry)
        try:
            self._window._session.save()
        except Exception:
            pass
        self._render()
