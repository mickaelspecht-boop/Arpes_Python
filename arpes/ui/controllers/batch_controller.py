"""Batch fit: applies _fit_full to all unfitted files in the folder.

Synchronous with QApplication.processEvents between files (no threading).
QProgressDialog with Cancel button. Already-fitted files are preserved
unless `force=True` (reserved for future programmatic calls).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog


class BatchController:
    def __init__(self, parent):
        self._parent = parent

    @property
    def _session(self):
        return self._parent._session

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    def _batch_fit_folder(self, force: bool = False) -> None:
        session = self._session
        if session.folder is None:
            QMessageBox.information(self._parent, "Batch fit",
                                    "No folder is open.")
            return
        targets: list[tuple[str, Path]] = []
        for name, entry in session.files.items():
            if (not force) and entry.fit_result:
                continue
            full = session.folder / name if session.folder else Path(name)
            if full.exists():
                targets.append((name, full))
        if not targets:
            QMessageBox.information(
                self._parent, "Batch fit",
                "No unfitted files found in the session.\n"
                "Load files via the browser first.",
            )
            return

        confirm = QMessageBox.question(
            self._parent, "Batch fit",
            f"Run a full MDC fit on {len(targets)} file(s)?\n\n"
            "The current MDC Fit panel parameters will be used.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        progress = QProgressDialog(
            "Batch fit in progress...", "Cancel", 0, len(targets), self._parent,
        )
        progress.setWindowTitle("Batch fit folder")
        # ApplicationModal: the loop pumps QApplication.processEvents() between
        # files, so a WindowModal dialog still lets clicks reach sibling docks
        # (e.g. the browser "Samples…" button), which opened a nested modal
        # dialog mid-loop and froze the UI. ApplicationModal blocks all windows.
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        ok = 0
        skipped = 0
        self._parent._batch_busy = True
        try:
            ok, skipped = self._run_targets(targets, progress)
        finally:
            self._parent._batch_busy = False
            progress.close()

        results = getattr(self._parent, "_results", None)
        if results is not None:
            try:
                results.refresh()
            except Exception:
                pass

        msg = f"Batch done: {ok} fitted, {skipped} failed/skipped out of {len(targets)} targets."
        self._status(msg)
        QMessageBox.information(self._parent, "Batch fit", msg)

    def _run_targets(self, targets, progress) -> tuple[int, int]:
        session = self._session
        ok = 0
        skipped = 0
        for i, (name, full) in enumerate(targets):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Fit {i + 1}/{len(targets)} : {name}")
            QApplication.processEvents()
            try:
                self._parent._load_ctrl.load(str(full))
                self._parent._fit_full()
                entry = session.files.get(name)
                if entry is not None and entry.fit_result:
                    ok += 1
                else:
                    skipped += 1
            except Exception as exc:
                skipped += 1
                self._status(f"Batch: failed {name} ({exc})")
            progress.setValue(i + 1)
            QApplication.processEvents()
        return ok, skipped
