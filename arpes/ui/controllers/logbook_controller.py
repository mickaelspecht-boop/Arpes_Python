"""Controller UI pour ingestion logbook (CSV/TSV/Excel).

Sort `arpes_explorer.ArpesExplorer._read_logbook` et ses dialogs (sheet/table/
mapping) de la God class. La logique de parsing pure reste dans
`arpes.io.logbook` / `arpes.io.logbook_io` ; ce contrôleur gère uniquement la
couche Qt (fichier de dialog, sélection feuille/header, persistance session).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
        self._last_applied_values = None

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
    # ----------------------------------------------------- subfolder helpers
    def _session_subfolders(self) -> list[str]:
        """Sous-dossiers (relatifs) de la session contenant au moins un fichier.

        Récursif, ignore les répertoires cachés et `.arpes_cache`.
        """
        root = self._session.folder
        if root is None or not root.exists():
            return []
        out: list[str] = []
        for d in sorted(p for p in root.rglob("*") if p.is_dir()):
            name = d.name
            if name.startswith(".") or name == ".arpes_cache":
                continue
            if any(part.startswith(".") or part == ".arpes_cache" for part in d.relative_to(root).parts):
                continue
            try:
                has_file = any(f.is_file() for f in d.iterdir())
            except OSError:
                has_file = False
            if has_file:
                out.append(str(d.relative_to(root)))
        return out

    def _pick_subfolders_dialog(self, rels: list[str], logbook_name: str) -> list[str]:
        """Liste à cases des sous-dossiers ; pré-coche ceux qui matchent le nom."""
        from arpes.io.logbook import _norm_text
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Sous-dossiers visés par ce logbook")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            f"Logbook : <b>{logbook_name}</b><br>"
            "Coche les sous-dossiers auxquels l'appliquer (les fichiers hors de "
            "ces dossiers ne seront pas touchés)."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lst = QListWidget()
        name_norm = _norm_text(Path(logbook_name).stem)
        for rel in rels:
            it = QListWidgetItem(rel)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            rel_norm = _norm_text(Path(rel).name)
            pre = bool(rel_norm) and (rel_norm in name_norm or name_norm in rel_norm)
            it.setCheckState(Qt.CheckState.Checked if pre else Qt.CheckState.Unchecked)
            lst.addItem(it)
        lay.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return []
        return [lst.item(i).text() for i in range(lst.count())
                if lst.item(i).checkState() == Qt.CheckState.Checked]

    # ---------------------------------------------------------------- dialogs
    def add_scoped_logbook(self) -> None:
        """Charge un logbook et l'attache à un ou plusieurs sous-dossiers.

        Flux : on choisit d'abord le FICHIER logbook, puis on coche les
        sous-dossiers visés. Chaque record est tagué `_subfolder_rel` et ne
        matche que les fichiers de ce sous-dossier — permet de coller deux
        logbooks distincts (ex: CA041 et CA046) dans la même session.
        """
        if self._session.folder is None:
            QMessageBox.warning(self._parent, "Logbook scopé",
                                "Ouvre d'abord un dossier de session.")
            return
        start = str(self._session.folder)
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Fichier logbook à attacher", start,
            "Tous les fichiers (*);;Logbook (*.xlsx *.xls *.xlsm *.csv *.tsv *.txt)")
        if not path:
            return
        rels = self._session_subfolders()
        if not rels:
            QMessageBox.warning(self._parent, "Logbook scopé",
                                "Aucun sous-dossier détecté dans la session.\n"
                                "Utilise 'Charger logbook (global)' à la place.")
            return
        chosen = self._pick_subfolders_dialog(rels, Path(path).name)
        if not chosen:
            return
        try:
            records, mapping, sheet_name = self.read(Path(path))
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook scopé", str(exc))
            self._status(f"Attention: Logbook scopé : {exc}")
            return
        # vire les anciens records de ces scopes, puis ré-ajoute une copie taguée
        keep = [r for r in self._session.logbook_records
                if not (isinstance(r, dict) and r.get("_subfolder_rel") in chosen)]
        added = []
        for rel in chosen:
            for r in records:
                if isinstance(r, dict):
                    rc = dict(r)
                    rc["_subfolder_rel"] = rel
                    added.append(rc)
            self._session.scoped_logbooks[rel] = {
                "path": path, "sheet": sheet_name, "n": len(records),
                "mapping": dict(mapping),
            }
        self._session.logbook_records = keep + added
        self._session.save()
        scope_txt = ", ".join(chosen)
        self._status(f"Logbook scopé '{Path(path).name}' → {scope_txt} | {len(records)} lignes ×{len(chosen)}")
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"logbook scopé → {len(chosen)} sous-dossier(s)")
        QMessageBox.information(
            self._parent, "Logbook scopé chargé",
            f"Fichier : {Path(path).name}\nSous-dossiers : {scope_txt}\n{len(records)} lignes chacun."
        )
        if self._parent._current_path:
            self.apply_to_controls(self._parent._current_path)
        self._browser.refresh()

    # ---------------------------------------------------- auto-attach scopes
    def auto_attach_scoped_logbooks_xlsx(self) -> None:
        """Scanne toutes les sheets d'un xlsx, auto-attache par Folder Name.

        Robuste aux variations de template :
        - Cherche cellule "Folder Name" / "Folder" / "Dossier" / variations
          dans les 15 premières lignes de chaque sheet (case-insensitive).
        - Matche contre sous-dossiers session : exact, case-insensitive,
          normalisé (alphanumérique seul), basename, substring (≥3 char).
        - Ignore sheets sans "Folder Name" ou sans colonne file+hv détectées.
        - Liste résultats avant validation user (dialog confirmation).
        """
        from arpes.io.logbook_io import scan_xlsx_for_scoped_logbooks
        try:
            import pandas as pd
        except ImportError:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                "pandas requis pour scanner les xlsx.")
            return
        if self._session.folder is None:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                "Ouvre d'abord un dossier de session.")
            return
        start = str(self._session.folder)
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Fichier xlsx à scanner", start,
            "Excel (*.xlsx *.xls *.xlsm)")
        if not path:
            return
        rels = self._session_subfolders()
        if not rels:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                "Aucun sous-dossier détecté dans la session.")
            return
        try:
            results = scan_xlsx_for_scoped_logbooks(pd, path, rels)
        except Exception as exc:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                f"Scan échoué : {exc}")
            self._status(f"✗ Auto-scope : {exc}")
            return
        if not results:
            QMessageBox.information(
                self._parent, "Auto-scope logbook",
                "Aucune sheet exploitable trouvée :\n"
                "- Pas de cellule « Folder Name » dans les sheets, ou\n"
                "- Le nom déclaré ne correspond à aucun sous-dossier session, ou\n"
                "- Sheets sans colonnes file+hv détectables.\n\n"
                f"Sous-dossiers session : {', '.join(rels[:10])}"
            )
            return

        # Dialog confirmation avec preview matches
        chosen = self._confirm_auto_scoped_dialog(results)
        if not chosen:
            return

        # Read records de chaque sheet via read_logbook standard (pour conservation
        # du flux inherit_logbook_context). Override sheet_name = sheet ciblée.
        for entry in chosen:
            sheet = entry["sheet"]
            rel = entry["subfolder_rel"]
            try:
                records, mapping, _sn = self.read(
                    Path(path),
                    sheet_override=sheet,
                )
            except Exception as exc:
                self._status(f"✗ Auto-scope : {sheet} → {exc}")
                continue
            # vire anciens records de ce scope
            self._session.logbook_records = [
                r for r in self._session.logbook_records
                if not (isinstance(r, dict) and r.get("_subfolder_rel") == rel)
            ]
            for r in records:
                if isinstance(r, dict):
                    rc = dict(r)
                    rc["_subfolder_rel"] = rel
                    self._session.logbook_records.append(rc)
            self._session.scoped_logbooks[rel] = {
                "path": path, "sheet": sheet, "n": len(records),
                "mapping": dict(mapping),
            }
        self._session.save()
        attached_txt = ", ".join(f"{e['sheet']}→{e['subfolder_rel']}" for e in chosen)
        self._status(f"✓ Auto-scope : {len(chosen)} sheets attachées ({attached_txt})")
        if self._parent._current_path:
            self.apply_to_controls(self._parent._current_path)
        self._browser.refresh()

    def _confirm_auto_scoped_dialog(self, results: list[dict]) -> list[dict]:
        """Dialog : preview matches sheet→sous-dossier, user coche ceux à attacher."""
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Auto-attacher scopés — confirmation")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            f"<b>{len(results)} match(es) détectés</b><br>"
            "Décoche ceux à ignorer. Les anciens scopés sur ces sous-dossiers "
            "seront <b>remplacés</b>."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lst = QListWidget()
        for r in results:
            txt = (f"{r['sheet']:25s} → {r['subfolder_rel']}  "
                   f"(folder déclaré: {r['folder_declared']}, {r['n_rows']} lignes)")
            it = QListWidgetItem(txt)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, r)
            lst.addItem(it)
        lay.addWidget(lst)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return []
        return [
            lst.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(lst.count())
            if lst.item(i).checkState() == Qt.CheckState.Checked
        ]

    # --------------------------------------------------------- detach / état
    def attached_logbooks(self) -> list[tuple[str, str, str]]:
        """Retourne [(scope_label, filename, key)] pour menu/affichage.

        key = "" pour le global, sinon le rel du sous-dossier.
        """
        out: list[tuple[str, str, str]] = []
        if self._session.logbook_path:
            out.append(("(global)", Path(self._session.logbook_path).name, ""))
        for rel, meta in sorted(self._session.scoped_logbooks.items()):
            out.append((rel, Path(meta.get("path", "")).name or "?", rel))
        return out

    def detach_logbook(self, key: str) -> None:
        """Détache le logbook global (key='') ou scopé (key=rel)."""
        if key == "":
            self._session.logbook_path = ""
            self._session.logbook_sheet = ""
            self._session.logbook_records = [
                r for r in self._session.logbook_records
                if isinstance(r, dict) and r.get("_subfolder_rel")
            ]
            label = "global"
        else:
            self._session.scoped_logbooks.pop(key, None)
            self._session.logbook_records = [
                r for r in self._session.logbook_records
                if not (isinstance(r, dict) and r.get("_subfolder_rel") == key)
            ]
            label = f"scopé '{key}'"
        self._session.save()
        self._status(f"Logbook {label} détaché.")
        self._browser.refresh()

    def open_dialog(self) -> None:
        """Ouvre un QFileDialog pour choisir un logbook puis le charge."""
        start = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Logbook ARPES", start,
            "Tous les fichiers (*);;Logbook (*.xlsx *.xls *.xlsm *.csv *.tsv *.txt)")
        if not path:
            return
        try:
            records, mapping, sheet_name = self.read(Path(path))
            self._session.logbook_path = path
            self._session.logbook_sheet = sheet_name
            self._session.logbook_mapping = mapping
            scoped = [r for r in self._session.logbook_records
                      if isinstance(r, dict) and r.get("_subfolder_rel")]
            self._session.logbook_records = scoped + list(records)
            self._session.save()
            used = ", ".join(f"{k}={v or '—'}" for k, v in mapping.items())
            sheet_txt = f" [{sheet_name}]" if sheet_name else ""
            self._status(f"Logbook chargé : {Path(path).name}{sheet_txt} | {len(records)} lignes | {used}")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"logbook chargé ({len(records)} lignes)")
            QMessageBox.information(
                self._parent, "Logbook chargé",
                f"{Path(path).name}{sheet_txt}\n{len(records)} lignes lues.\n\nColonnes détectées :\n{used}"
            )
            if self._parent._current_path:
                self.apply_to_controls(self._parent._current_path)
            self._browser.refresh()
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook", str(exc))
            self._status(f"Attention: Logbook : {exc}")

    def read(
        self,
        path: Path,
        *,
        sheet_override: str | None = None,
    ) -> tuple[list[dict], dict[str, str], str]:
        """Lit le logbook via `read_logbook_file` en injectant les sélecteurs UI.

        ``sheet_override`` : force la sheet utilisée (court-circuite le dialog
        de sélection). Utilisé par l'auto-attach scopé.
        """
        sheet_selector = self._choose_excel_sheet
        if sheet_override is not None:
            sheet_selector = lambda _names: sheet_override
        result = read_logbook_file(
            path,
            sheet_selector=sheet_selector,
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
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Ligne d'en-tête du logbook")
        lay = QVBoxLayout(dlg)
        label = QLabel(
            "Choisis la ligne qui contient les vrais noms de colonnes "
            "(triées par pertinence — la meilleure devinée est en haut)."
        )
        label.setWordWrap(True)
        lay.addWidget(label)
        cmb = QComboBox()
        for row_idx in scored:
            values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
            preview = " | ".join(v for v in values if v)
            score = score_row(row_idx)
            tag = "✓" if score >= 6 else ("?" if score >= 3 else "✗")
            cmb.addItem(f"{tag} Ligne {row_idx + 1}: {preview[:140]}", row_idx)
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
    def _manager(self) -> LogbookManager:
        scoped = {
            rel: meta.get("mapping", {})
            for rel, meta in (self._session.scoped_logbooks or {}).items()
            if isinstance(meta, dict) and meta.get("mapping")
        }
        return LogbookManager(
            self._session.logbook_records,
            self._session.logbook_mapping,
            self._session.folder,
            scoped_mappings=scoped,
        )

    def find_record_for_path(self, path) -> dict | None:
        return self._manager().find_record_for_path(path)

    def apply_to_controls(self, path) -> bool:
        manager = self._manager()
        entry = self._session.get_or_create(self._session.key_for_path(path))
        values = manager.apply_to_entry(entry, path)
        self._last_applied_values = values
        if values.hv is not None:
            self._params.set_hv_value_with_source(values.hv, "logbook")
        if values.has_any():
            self._session.save()
        return values.has_any()
