"""Controller UI pour ingestion logbook (CSV/TSV/Excel).

Sort `arpes_explorer.ArpesExplorer._read_logbook` et ses dialogs (sheet/table/
mapping) de la God class. La logique de parsing pure reste dans
`arpes.io.logbook` / `arpes.io.logbook_io` ; ce contrôleur gère uniquement la
couche Qt (fichier de dialog, sélection feuille/header, persistance session).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from arpes.io.logbook import LogbookManager, _cell_text, _norm_text
from arpes.io.logbook_io import read_logbook as read_logbook_file


class LogbookIngestController:
    """Charge un logbook utilisateur et applique ses valeurs aux contrôles UI.

    `parent` doit exposer `_session`, `_params`, `_browser`, `_current_path`,
    `_status` et `_apply_logbook_to_controls` n'est plus nécessaire (déplacé
    ici sous `apply_to_controls`).
    """

    def __init__(self, parent):
        self._parent = parent

    # ------------------------------------------------------------------ helpers
    @property
    def _session(self):
        return self._parent._session

    @property
    def _params(self):
        return self._parent._params

    @property
    def _browser(self):
        return self._parent._browser

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    # ---------------------------------------------------------------- dialogs
    def open_dialog(self) -> None:
        """Ouvre un QFileDialog pour choisir un logbook puis le charge."""
        start = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Logbook ARPES", start,
            "Logbook (*.xlsx *.xls *.csv *.tsv);;Tous les fichiers (*)")
        if not path:
            return
        try:
            records, mapping, sheet_name = self.read(Path(path))
            self._session.logbook_path = path
            self._session.logbook_sheet = sheet_name
            self._session.logbook_mapping = mapping
            self._session.logbook_records = records
            self._session.save()
            used = ", ".join(f"{k}={v or '—'}" for k, v in mapping.items())
            sheet_txt = f" [{sheet_name}]" if sheet_name else ""
            self._status(f"Logbook chargé : {Path(path).name}{sheet_txt} | {len(records)} lignes | {used}")
            QMessageBox.information(
                self._parent, "Logbook chargé",
                f"{Path(path).name}{sheet_txt}\n{len(records)} lignes lues.\n\nColonnes détectées :\n{used}"
            )
            if self._parent._current_path:
                self.apply_to_controls(self._parent._current_path)
            self._browser.refresh()
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook", str(exc))
            self._status(f"⚠ Logbook : {exc}")

    def read(self, path: Path) -> tuple[list[dict], dict[str, str], str]:
        """Lit le logbook via `read_logbook_file` en injectant les sélecteurs UI."""
        result = read_logbook_file(
            path,
            sheet_selector=self._choose_excel_sheet,
            table_selector=self._choose_excel_table,
            mapping_selector=self._choose_logbook_mapping,
        )
        return result.records, result.mapping, result.sheet_name

    def _choose_excel_sheet(self, sheet_names: list[str]) -> str:
        if not sheet_names:
            return ""
        if len(sheet_names) == 1:
            return sheet_names[0]
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Feuille du logbook")
        lay = QVBoxLayout(dlg)
        label = QLabel("Choisis la feuille qui correspond au compound / dataset.")
        label.setWordWrap(True)
        lay.addWidget(label)
        cmb = QComboBox()
        cmb.addItems(sheet_names)
        preferred = ""
        if self._session.folder is not None:
            preferred = self._session.folder.name
        preferred_norm = _norm_text(preferred)
        if self._session.logbook_sheet in sheet_names:
            cmb.setCurrentText(self._session.logbook_sheet)
        elif preferred in sheet_names:
            cmb.setCurrentText(preferred)
        else:
            for sheet in sheet_names:
                sheet_norm = _norm_text(sheet)
                if sheet_norm and sheet_norm in preferred_norm:
                    cmb.setCurrentText(sheet)
                    break
        lay.addWidget(cmb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return ""
        return cmb.currentText()

    def _choose_excel_table(self, raw, candidates: list[int]):
        from arpes.io.logbook_io import excel_table_from_header
        if not candidates:
            return None
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Ligne d'en-tête du logbook")
        lay = QVBoxLayout(dlg)
        label = QLabel("Choisis la ligne qui contient les vrais noms de colonnes.")
        label.setWordWrap(True)
        lay.addWidget(label)
        cmb = QComboBox()
        for row_idx in candidates:
            values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
            preview = " | ".join(v for v in values if v)
            cmb.addItem(f"Ligne {row_idx + 1}: {preview[:140]}", row_idx)
        lay.addWidget(cmb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return excel_table_from_header(raw, int(cmb.currentData()))

    def _choose_logbook_mapping(self, columns: list[str], mapping: dict[str, str]) -> dict[str, str]:
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Colonnes du logbook")
        lay = QFormLayout(dlg)
        combos: dict[str, QComboBox] = {}
        labels = {
            "file": "Fichier / scan:",
            "hv": "hν:",
            "temperature": "Température:",
            "polarization": "Polarisation:",
            "direction": "Direction / chemin:",
            "azi": "Azimut:",
            "polar": "Polar / theta manip:",
            "tilt": "Tilt / phi manip:",
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
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return mapping
        return {key: cmb.currentText() for key, cmb in combos.items()}

    # ---------------------------------------------------------------- session
    def find_record_for_path(self, path) -> dict | None:
        manager = LogbookManager(
            self._session.logbook_records,
            self._session.logbook_mapping,
            self._session.folder,
        )
        return manager.find_record_for_path(path)

    def apply_to_controls(self, path) -> bool:
        manager = LogbookManager(
            self._session.logbook_records,
            self._session.logbook_mapping,
            self._session.folder,
        )
        entry = self._session.get_or_create(self._session.key_for_path(path))
        values = manager.apply_to_entry(entry, path)
        if values.hv is not None:
            self._params.sp_hv.blockSignals(True)
            self._params.sp_hv.setValue(values.hv)
            self._params.sp_hv.blockSignals(False)
        if values.has_any():
            self._session.save()
        return values.has_any()
