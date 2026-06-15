"""UI controller for logbook ingestion (CSV/TSV/Excel).

Moves `arpes_explorer.ArpesExplorer._read_logbook` and its dialogs
(sheet/table/mapping) out of the God class. Pure parsing logic stays in
`arpes.io.logbook` / `arpes.io.logbook_io`; this controller handles only the Qt
layer (file dialog, sheet/header selection, session persistence).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QLocale, Qt
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
    QDoubleSpinBox,
    QVBoxLayout,
)

from arpes.core.sample import SampleConfig
from arpes.io.logbook import LogbookManager, _cell_float, _cell_text, _norm_text
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
        """Relative session subfolders containing at least one file.

        Recursive, ignores hidden directories and `.arpes_cache`.
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
        """Checklist of subfolders; pre-check those matching the name."""
        from arpes.io.logbook import _norm_text
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Subfolders targeted by this logbook")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            f"Logbook: <b>{logbook_name}</b><br>"
            "Check the subfolders to apply it to (files outside these "
            "folders will not be changed)."
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
            QMessageBox.warning(self._parent, "Scoped logbook",
                                "Open a session folder first.")
            return
        start = str(self._session.folder)
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Logbook file to attach", start,
            "All files (*);;Logbook (*.xlsx *.xls *.xlsm *.csv *.tsv *.txt)")
        if not path:
            return
        rels = self._session_subfolders()
        if not rels:
            QMessageBox.warning(self._parent, "Scoped logbook",
                                "No subfolder detected in the session.\n"
                                "Use 'Load logbook (global)' instead.")
            return
        chosen = self._pick_subfolders_dialog(rels, Path(path).name)
        if not chosen:
            return
        try:
            records, mapping, sheet_name = self.read(Path(path))
        except Exception as exc:
            QMessageBox.warning(self._parent, "Scoped logbook", str(exc))
            self._status(f"Warning: scoped logbook: {exc}")
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
                    rc["_sheet_name"] = sheet_name
                    added.append(rc)
            self._session.scoped_logbooks[rel] = {
                "path": path, "sheet": sheet_name, "n": len(records),
                "mapping": dict(mapping),
            }
        self._session.logbook_records = keep + added
        self._ensure_sample_params_after_logbook(added, mapping)
        self._session.save()
        scope_txt = ", ".join(chosen)
        self._status(f"Scoped logbook '{Path(path).name}' → {scope_txt} | {len(records)} rows ×{len(chosen)}")
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"scoped logbook → {len(chosen)} subfolder(s)")
        QMessageBox.information(
            self._parent, "Scoped logbook loaded",
            f"File: {Path(path).name}\nSubfolders: {scope_txt}\n{len(records)} rows each."
        )
        if self._parent._current_path:
            self.apply_to_controls(self._parent._current_path)
        self._browser.refresh()

    # ---------------------------------------------------- auto-attach scopes
    def auto_attach_scoped_logbooks_xlsx(self) -> None:
        """Scanne toutes les sheets d'un xlsx, auto-attache par Folder Name.

        Robuste aux variations de template :
        - Search for "Folder Name" / "Folder" / localized folder-label variations
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
                                "pandas is required to scan xlsx files.")
            return
        if self._session.folder is None:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                "Open a session folder first.")
            return
        start = str(self._session.folder)
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "xlsx file to scan", start,
            "Excel (*.xlsx *.xls *.xlsm)")
        if not path:
            return
        rels = self._session_subfolders()
        if not rels:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                "No subfolder detected in the session.")
            return
        try:
            results = scan_xlsx_for_scoped_logbooks(pd, path, rels)
        except Exception as exc:
            QMessageBox.warning(self._parent, "Auto-scope logbook",
                                f"Scan failed: {exc}")
            self._status(f"✗ Auto-scope: {exc}")
            return
        if not results:
            QMessageBox.information(
                self._parent, "Auto-scope logbook",
                "No usable sheet found:\n"
                "- No \"Folder Name\" cell in the sheets, or\n"
                "- The declared name does not match any session subfolder, or\n"
                "- Sheets have no detectable file+hv columns.\n\n"
                f"Session subfolders: {', '.join(rels[:10])}"
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
                self._status(f"✗ Auto-scope: {sheet} → {exc}")
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
                    rc["_sheet_name"] = sheet
                    self._session.logbook_records.append(rc)
            self._session.scoped_logbooks[rel] = {
                "path": path, "sheet": sheet, "n": len(records),
                "mapping": dict(mapping),
            }
        self._ensure_sample_params_after_logbook(
            self._session.logbook_records,
            self._session.logbook_mapping,
        )
        self._session.save()
        attached_txt = ", ".join(f"{e['sheet']}→{e['subfolder_rel']}" for e in chosen)
        self._status(f"✓ Auto-scope: {len(chosen)} sheets attached ({attached_txt})")
        if self._parent._current_path:
            self.apply_to_controls(self._parent._current_path)
        self._browser.refresh()

    def _confirm_auto_scoped_dialog(self, results: list[dict]) -> list[dict]:
        """Dialog previewing sheet-to-subfolder matches for user selection."""
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Auto-attach scoped logbooks - confirmation")
        lay = QVBoxLayout(dlg)
        lbl = QLabel(
            f"<b>{len(results)} match(es) detected</b><br>"
            "Uncheck the ones to ignore. Existing scoped logbooks on these "
            "subfolders will be <b>replaced</b>."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lst = QListWidget()
        for r in results:
            txt = (f"{r['sheet']:25s} → {r['subfolder_rel']}  "
                   f"(declared folder: {r['folder_declared']}, {r['n_rows']} rows)")
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

    # ------------------------------------------------------ sample parameters
    def _ensure_sample_params_after_logbook(
        self, records: list[dict], mapping: dict[str, str]
    ) -> None:
        """Prompt once for sample constants missing from the loaded logbook.

        `current_sample` is the durable source. Per-file logbook values still
        win through SampleConfig.from_meta when present.
        """
        sample = SampleConfig.from_dict(self._session.current_sample)
        detected = self._sample_values_from_logbook(records, mapping)
        merged = sample.merge_missing_from(detected)

        missing = [
            key for key, value in (
                ("a", merged.a_angstrom),
                ("b", merged.b_angstrom),
                ("c", merged.c_angstrom),
                ("work", merged.work_function_eV),
            )
            if float(value or 0.0) <= 0.0
        ]
        if not missing:
            self._save_and_sync_sample(merged, source="logbook")
            return

        chosen = self._sample_params_dialog(merged, missing)
        if chosen is None:
            self._save_and_sync_sample(merged, source=merged.lattice_source or "logbook")
            self._status(
                "Sample parameters still incomplete; detected constants were kept, "
                "but FS/KZ may need manual a/b/c/φ."
            )
            return
        self._save_and_sync_sample(chosen, source="manual")

    @staticmethod
    def _sample_values_from_logbook(
        records: list[dict], mapping: dict[str, str]
    ) -> SampleConfig:
        def first_positive(key: str) -> float:
            col = mapping.get(key, "")
            if not col:
                return 0.0
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                val = _cell_float(rec.get(col))
                if val is not None and val > 0:
                    return float(val)
            return 0.0

        return SampleConfig(
            a_angstrom=first_positive("crystal_a_angstrom"),
            b_angstrom=first_positive("crystal_b_angstrom"),
            c_angstrom=first_positive("crystal_c_angstrom"),
            work_function_eV=first_positive("work_function_eV"),
            lattice_source="logbook",
        )

    def _sample_params_dialog(
        self, sample: SampleConfig, missing: list[str]
    ) -> SampleConfig | None:
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Sample constants")
        lay = QVBoxLayout(dlg)
        label = QLabel(
            "The logbook does not provide all constants needed for physical "
            "k-space conversion. Enter the sample values once for this session."
        )
        label.setWordWrap(True)
        lay.addWidget(label)
        form = QFormLayout()
        lay.addLayout(form)

        def spin(value: float, *, lo: float, hi: float, step: float, dec: int):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(dec)
            w.setLocale(QLocale(QLocale.Language.C))
            w.setValue(float(value or 0.0))
            w.setKeyboardTracking(False)
            return w

        a0 = float(sample.a_angstrom or 0.0)
        b0 = float(sample.b_angstrom or a0 or 0.0)
        c0 = float(sample.c_angstrom or 0.0)
        w0 = float(sample.work_function_eV or 0.0)
        sp_a = spin(a0, lo=0.0, hi=30.0, step=0.01, dec=4)
        sp_b = spin(b0, lo=0.0, hi=30.0, step=0.01, dec=4)
        sp_c = spin(c0, lo=0.0, hi=80.0, step=0.01, dec=4)
        sp_w = spin(w0, lo=0.0, hi=10.0, step=0.01, dec=3)
        if float(sample.b_angstrom or 0.0) <= 0.0:
            sp_a.valueChanged.connect(
                lambda v: sp_b.setValue(float(v)) if sp_b.value() <= 0.0 else None
            )
        form.addRow("a (Å):", sp_a)
        form.addRow("b (Å):", sp_b)
        form.addRow("c (Å):", sp_c)
        form.addRow("φ work function (eV):", sp_w)

        if missing:
            miss_txt = ", ".join(missing)
            hint = QLabel(f"Missing from logbook/session: {miss_txt}.")
            hint.setWordWrap(True)
            lay.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return SampleConfig(
            formula=sample.formula,
            a_angstrom=float(sp_a.value()),
            b_angstrom=float(sp_b.value()),
            c_angstrom=float(sp_c.value()),
            work_function_eV=float(sp_w.value()),
            space_group=sample.space_group,
            mp_id=sample.mp_id,
            lattice_source="manual",
        )

    def _save_and_sync_sample(self, sample: SampleConfig, *, source: str) -> None:
        if sample.a_angstrom <= 0 and sample.b_angstrom <= 0 and sample.c_angstrom <= 0 and sample.work_function_eV <= 0:
            return
        if source:
            sample.lattice_source = source
        self._session.current_sample = sample.to_dict()
        self._sync_sample_widgets(sample)
        self._status(
            "Sample constants: "
            f"a={sample.a_angstrom:g} Å, b={sample.b_angstrom:g} Å, "
            f"c={sample.c_angstrom:g} Å, φ={sample.work_function_eV:g} eV."
        )

    def _sync_sample_widgets(self, sample: SampleConfig) -> None:
        def set_spin(obj, name: str, value: float) -> None:
            sp = getattr(obj, name, None) if obj is not None else None
            if sp is None or float(value or 0.0) <= 0.0:
                return
            if abs(float(sp.value()) - float(value)) <= 1e-9:
                return
            sp.blockSignals(True)
            sp.setValue(float(value))
            sp.blockSignals(False)

        set_spin(self._params, "sp_phi", sample.work_function_eV)
        set_spin(self._params, "sp_crystal_a", sample.a_angstrom)
        fs_controls = getattr(self._parent, "_fs_controls", None)
        set_spin(fs_controls, "sp_a", sample.a_angstrom)
        set_spin(fs_controls, "sp_b", sample.b_angstrom or sample.a_angstrom)
        kz_controls = getattr(self._parent, "_kz_controls", None)
        set_spin(kz_controls, "sp_a", sample.a_angstrom)
        set_spin(kz_controls, "sp_c", sample.c_angstrom)

    # --------------------------------------------------------- detach / state
    def attached_logbooks(self) -> list[tuple[str, str, str]]:
        """Return [(scope_label, filename, key)] for menu/display.

        key = "" for the global logbook, otherwise the subfolder rel path.
        """
        out: list[tuple[str, str, str]] = []
        if self._session.logbook_path:
            out.append(("(global)", Path(self._session.logbook_path).name, ""))
        for rel, meta in sorted(self._session.scoped_logbooks.items()):
            out.append((rel, Path(meta.get("path", "")).name or "?", rel))
        return out

    def detach_logbook(self, key: str) -> None:
        """Detach the global logbook (key='') or a scoped one (key=rel)."""
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
            label = f"scoped '{key}'"
        self._session.save()
        self._status(f"Logbook {label} detached.")
        self._browser.refresh()

    def open_dialog(self) -> None:
        """Ouvre un QFileDialog pour choisir un logbook puis le charge."""
        start = str(self._session.folder or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Logbook ARPES", start,
            "All files (*);;Logbook (*.xlsx *.xls *.xlsm *.csv *.tsv *.txt)")
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
            self._ensure_sample_params_after_logbook(records, mapping)
            self._session.save()
            used = ", ".join(f"{k}={v or '—'}" for k, v in mapping.items())
            sheet_txt = f" [{sheet_name}]" if sheet_name else ""
            self._status(f"Logbook loaded: {Path(path).name}{sheet_txt} | {len(records)} rows | {used}")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"logbook loaded ({len(records)} rows)")
            QMessageBox.information(
                self._parent, "Logbook loaded",
                f"{Path(path).name}{sheet_txt}\n{len(records)} rows read.\n\nDetected columns:\n{used}"
            )
            if self._parent._current_path:
                self.apply_to_controls(self._parent._current_path)
            self._browser.refresh()
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook", str(exc))
            self._status(f"Warning: Logbook: {exc}")

    def read(
        self,
        path: Path,
        *,
        sheet_override: str | None = None,
        silent_mapping: bool = False,
    ) -> tuple[list[dict], dict[str, str], str]:
        """Lit le logbook via `read_logbook_file` en injectant les sélecteurs UI.

        ``sheet_override`` : force la sheet utilisée (court-circuite le dialog
        de sélection). ``silent_mapping`` : garde le mapping heuristique sans
        ouvrir le dialog de mapping (sample setup : jamais de dialog imbriqué).
        """
        sheet_selector = self._choose_excel_sheet
        if sheet_override is not None:
            sheet_selector = lambda _names: sheet_override
        result = read_logbook_file(
            path,
            sheet_selector=sheet_selector,
            table_selector=self._choose_excel_table,
            mapping_selector=None if silent_mapping else self._choose_logbook_mapping,
        )
        return result.records, result.mapping, result.sheet_name

    def attach_scoped_silent(
        self,
        path,
        sheet: str,
        rels: list[str],
        *,
        mapping_override: dict | None = None,
    ) -> int:
        """Attach (path, sheet) to subfolder scopes with ZERO dialog.

        Called from the Sample setup flow after its dialog has closed — any
        interactive selector here would nest dialogs (council redteam case).
        ``rel == ""`` targets the session-wide (global) logbook. Returns the
        number of rows read; raises on unreadable files (caller surfaces it).
        """
        path = Path(path)
        records, mapping, sheet_name = self.read(
            path, sheet_override=(sheet or None), silent_mapping=True,
        )
        if mapping_override:
            mapping = dict(mapping_override)
        scoped_rels = [r for r in rels if r]
        if any(not r for r in rels):
            self._session.logbook_path = str(path)
            self._session.logbook_sheet = sheet_name
            self._session.logbook_mapping = dict(mapping)
            keep_scoped = [r for r in self._session.logbook_records
                           if isinstance(r, dict) and r.get("_subfolder_rel")]
            self._session.logbook_records = keep_scoped + list(records)
        if scoped_rels:
            keep = [r for r in self._session.logbook_records
                    if not (isinstance(r, dict) and r.get("_subfolder_rel") in scoped_rels)]
            added = []
            for rel in scoped_rels:
                for r in records:
                    if isinstance(r, dict):
                        rc = dict(r)
                        rc["_subfolder_rel"] = rel
                        rc["_sheet_name"] = sheet_name
                        added.append(rc)
                self._session.scoped_logbooks[rel] = {
                    "path": str(path), "sheet": sheet_name, "n": len(records),
                    "mapping": dict(mapping),
                }
            self._session.logbook_records = keep + added
        self._session.save()
        scope_txt = ", ".join(scoped_rels) or "session"
        self._status(f"✓ Logbook {path.name} [{sheet_name}] → {scope_txt} | {len(records)} rows")
        if self._parent._current_path:
            self.apply_to_controls(self._parent._current_path)
        self._browser.refresh()
        return len(records)

    # Interactive Excel choosers live in `logbook_excel_choose.py` (split to keep
    # this file under the 700-LOC cap). Thin wrappers preserve callers/tests.
    def _choose_excel_sheet(self, sheet_names: list[str]) -> str:
        from arpes.ui.controllers.logbook_excel_choose import choose_excel_sheet
        return choose_excel_sheet(self, sheet_names)

    def _choose_excel_table(self, raw, candidates: list[int]):
        from arpes.ui.controllers.logbook_excel_choose import choose_excel_table
        return choose_excel_table(self, raw, candidates)

    def _choose_logbook_mapping(self, columns: list[str], mapping: dict[str, str]) -> dict[str, str]:
        from arpes.ui.controllers.logbook_excel_choose import choose_logbook_mapping
        return choose_logbook_mapping(self, columns, mapping)

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
            # Provenance: surface which scoped sheet/subfolder fed the values.
            if values.matched_subfolder:
                self._status(
                    f"Logbook: {Path(path).name} ← sheet "
                    f"'{values.matched_sheet or '?'}' [{values.matched_subfolder}]"
                )
            return True
        # No record matched. Stay SILENT for the global/no-logbook case (a file
        # may legitimately sit outside the plan). Surface ONLY when the session
        # has scoped records covering this file's subfolder yet none matched —
        # a real gap the user must see (no silent fallback to another sheet).
        if manager.has_scoped_records_for_path(path):
            self._status(
                f"Logbook: no record for {Path(path).name} in its scoped sheet "
                "— values left uncalibrated (check the sheet's File column)."
            )
        return False
