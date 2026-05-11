"""Batch fit : applique _fit_full sur tous les fichiers non-fittés du dossier.

Synchrone avec QApplication.processEvents entre fichiers (pas de threading).
QProgressDialog avec bouton Annuler. Fichiers déjà fittés sont préservés
sauf si `force=True` (réservé à un futur appel programmatique).
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
                                    "Aucun dossier ouvert.")
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
                "Aucun fichier non-fitté trouvé dans la session.\n"
                "Charge d'abord les fichiers via le navigateur.",
            )
            return

        confirm = QMessageBox.question(
            self._parent, "Batch fit",
            f"Lancer un fit MDC complet sur {len(targets)} fichier(s) ?\n\n"
            "Les paramètres actuels du panneau Fit MDC sont utilisés.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        progress = QProgressDialog(
            "Batch fit en cours...", "Annuler", 0, len(targets), self._parent,
        )
        progress.setWindowTitle("Batch fit dossier")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

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
                self._status(f"Batch : échec {name} ({exc})")
            progress.setValue(i + 1)
            QApplication.processEvents()
        progress.close()

        results = getattr(self._parent, "_results", None)
        if results is not None:
            try:
                results.refresh()
            except Exception:
                pass

        msg = f"Batch terminé : {ok} fittés, {skipped} échec/ignoré sur {len(targets)} cibles."
        self._status(msg)
        QMessageBox.information(self._parent, "Batch fit", msg)
