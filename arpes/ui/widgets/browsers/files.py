"""Panneau navigateur de fichiers ARPES — discovery + groupement + summary."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QStringListModel, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from arpes.core.session import Session, normalize_tags, session_tags
from arpes.io.loaders import detect_format, detect_scan_kind, loader_label
from arpes.io.logbook import (
    LogbookManager,
    _cell_float,
    _cell_text,
    _format_direction_label,
)


class FileBrowserPanel(QWidget):
    file_selected = pyqtSignal(str)   # émet le chemin complet
    session_reloaded = pyqtSignal()   # émet après chargement .arpes_session.json depuis disque

    STATUS_ICONS = {"unloaded": "[ ]", "loaded": "[L]", "fitted": "[F]"}
    STATUS_COLORS = {"unloaded": "#888", "loaded": "#f0c040", "fitted": "#60e080"}

    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._folder: Path | None = None
        self._collapsed_groups: set[str] = set()
        self._group_mode = "Dossier"
        self._group_fields: list[str] = ["Dossier"]
        self._items_cache: list[Path] | None = None
        self._loader_label_cache: dict[str, tuple[tuple[int, int] | None, str]] = {}
        self._scan_kind_cache: dict[str, tuple[tuple[int, int] | None, str]] = {}
        self._logbook_record_cache: dict[str, dict | None] = {}
        from arpes.ui.controllers.browser_controller import BrowserController
        self._browser_ctrl = BrowserController(self)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        btn = QPushButton("Dossier")
        btn.clicked.connect(self._open_folder)
        top.addWidget(btn)
        btn_refresh = QPushButton("Actualiser")
        btn_refresh.setFixedWidth(78)
        btn_refresh.setToolTip("Rafraîchir la liste des fichiers")
        btn_refresh.clicked.connect(self.refresh)
        top.addWidget(btn_refresh)
        self._lbl_folder = QLabel("—")
        self._lbl_folder.setWordWrap(True)
        self._lbl_folder.setStyleSheet("font-size:10px; color:#aaa;")
        lay.addLayout(top)
        lay.addWidget(self._lbl_folder)

        self._lbl_summary = QLabel("Aucun dossier chargé")
        self._lbl_summary.setWordWrap(True)
        self._lbl_summary.setStyleSheet("font-size:10px; color:#aaa;")
        lay.addWidget(self._lbl_summary)

        self._tag_filter = QLineEdit()
        self._tag_filter.setPlaceholderText("Filtre tag")
        self._tag_filter.setToolTip("Filtre l'affichage aux fichiers contenant ce tag.")
        self._tag_filter_model = QStringListModel([])
        filter_completer = QCompleter(self._tag_filter_model, self._tag_filter)
        filter_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._tag_filter.setCompleter(filter_completer)
        self._tag_filter.textChanged.connect(self._on_tag_filter_changed)
        lay.addWidget(self._tag_filter)

        mode_row = QVBoxLayout()
        mode_title = QLabel("Organiser par:")
        mode_title.setStyleSheet("font-size:10px; color:#aaa;")
        mode_row.addWidget(mode_title)
        checks_row_1 = QHBoxLayout()
        checks_row_2 = QHBoxLayout()
        self._group_checks: dict[str, QCheckBox] = {}
        group_defs = [
            ("Dossier", "Dossier"),
            ("Type", "Type"),
            ("hν", "hν"),
            ("Température", "T"),
            ("Chemin", "Chemin"),
            ("Polarisation", "Pol"),
            ("Labo", "Labo"),
        ]
        for i, (field, label) in enumerate(group_defs):
            chk = QCheckBox(label)
            chk.setChecked(field == "Dossier")
            chk.setToolTip(
                "Critère cumulable d'organisation visuelle.\n"
                "N'applique aucune correction EF/Γ et ne prouve pas que les "
                "fichiers sont directement comparables."
            )
            chk.stateChanged.connect(self._on_group_checks_changed)
            self._group_checks[field] = chk
            (checks_row_1 if i < 4 else checks_row_2).addWidget(chk)
        checks_row_1.addStretch(1)
        checks_row_2.addStretch(1)
        mode_row.addLayout(checks_row_1)
        mode_row.addLayout(checks_row_2)
        lay.addLayout(mode_row)

        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget { background:#222; color:#ddd; font-size:11px; }
            QListWidget::item:selected { background:#2a6099; }
        """)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_change)
        lay.addWidget(self._list, stretch=1)

        self._lbl_selection = QLabel("Sélectionne un fichier à charger")
        self._lbl_selection.setWordWrap(True)
        self._lbl_selection.setStyleSheet(
            "font-size:10px; color:#c8c8c8; background:#1c1c1c; "
            "border:1px solid #333; padding:5px; border-radius:3px;"
        )
        lay.addWidget(self._lbl_selection)

        self._btn_load = QPushButton("Charger la sélection")
        self._btn_load.clicked.connect(self._load_selected)
        self._btn_load.setEnabled(False)
        self._btn_load.setToolTip("Choisir un fichier ou un groupe dans la liste pour activer cette action.")
        lay.addWidget(self._btn_load)

        self.setMinimumWidth(180)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier données ARPES",
                                                   str(Path.home()))
        if not folder:
            return
        fresh = False
        existing = Path(folder) / ".arpes_session.json"
        if existing.exists():
            box = QMessageBox(self)
            box.setWindowTitle("Session existante")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"Ce dossier contient déjà une session enregistrée "
                f"(.arpes_session.json — fits, calibrations EF, logbook, tags…)."
            )
            box.setInformativeText("Reprendre cette session, ou repartir de zéro ?")
            b_resume = box.addButton("Reprendre", QMessageBox.ButtonRole.AcceptRole)
            b_fresh = box.addButton("Nouvelle session", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Annuler", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(b_resume)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_fresh:
                fresh = True
            elif clicked is not b_resume:
                return
        self.set_folder(Path(folder), fresh=fresh)

    def set_folder(self, folder: Path, *, fresh: bool = False):
        if fresh:
            # purge l'état en mémoire (fits/logbook/calibs d'un dossier précédent)
            self._session.reset(keep_folder=False)
        self._folder = folder
        self._session.folder = folder
        self._lbl_folder.setText(folder.name)
        self._items_cache = None
        self._loader_label_cache.clear()
        self._scan_kind_cache.clear()
        self._logbook_record_cache.clear()
        loaded = False
        json_path = self._session.json_path
        if fresh and json_path and json_path.exists():
            # archive l'ancienne session plutôt que la perdre
            from datetime import datetime
            bak = json_path.with_suffix(f".json.bak-{datetime.now():%Y%m%d-%H%M%S}")
            try:
                json_path.rename(bak)
            except OSError:
                pass
        if not fresh and json_path and json_path.exists():
            try:
                self._session.load(json_path)
                loaded = True
            except Exception:
                pass
        self.refresh_tag_completions()
        self._populate()
        # émet aussi en mode 'fresh' pour que les widgets (hν, EF, notes, sections
        # de fit…) soient remis aux valeurs par défaut de la session vide.
        if (loaded or fresh) and hasattr(self, "session_reloaded"):
            self.session_reloaded.emit()

    def refresh(self):
        self._items_cache = None
        self._loader_label_cache.clear()
        self._scan_kind_cache.clear()
        self._logbook_record_cache.clear()
        self.refresh_tag_completions()
        self._populate()

    def refresh_tag_completions(self):
        tags = session_tags(self._session)
        if hasattr(self, "_tag_filter_model"):
            self._tag_filter_model.setStringList(tags)

    def _on_tag_filter_changed(self, _text: str):
        self._populate()

    def _is_cls_dataset_dir(self, p: Path) -> bool:
        if not p.is_dir():
            return False
        for param_file in p.glob("*_param.txt"):
            prefix = param_file.name.removesuffix("_param.txt")
            if any(p.glob(f"{prefix}_Cycle_*_Step_*.txt")):
                return True
        return False

    def _is_data_file(self, p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name.endswith("_param.txt"):
            return False
        if p.suffix.lower() in {".pxt", ".ibw", ".zip"}:
            return True
        # CLS BM : fichier sans extension avec un fichier voisin <nom>_param.txt
        return p.suffix == "" and (p.parent / f"{p.name}_param.txt").exists()

    def _discover_items(self) -> list[Path]:
        if self._items_cache is not None:
            return list(self._items_cache)
        if not self._folder:
            return []
        out: list[Path] = []
        for p in sorted(self._folder.rglob("*")):
            if p.name.startswith("."):
                continue
            if self._is_cls_dataset_dir(p):
                out.append(p)
                # ne pas lister aussi tous les Cycle/Step comme fichiers séparés
                continue
            if self._is_data_file(p):
                # Si le fichier est à l'intérieur d'un dataset CLS FS, on l'ignore
                if any(parent != self._folder and self._is_cls_dataset_dir(parent)
                       for parent in p.parents if self._folder in parent.parents or parent == self._folder):
                    continue
                out.append(p)
        self._items_cache = sorted(set(out), key=lambda x: str(x.relative_to(self._folder)).lower())
        return list(self._items_cache)

    def _group_label(self, group: str) -> str:
        if group == ".":
            return self._folder.name if self._folder else "."
        return group

    def _on_group_checks_changed(self):
        fields = [name for name, chk in self._group_checks.items() if chk.isChecked()]
        if not fields:
            fields = ["Dossier"]
            self._group_checks["Dossier"].blockSignals(True)
            self._group_checks["Dossier"].setChecked(True)
            self._group_checks["Dossier"].blockSignals(False)
        self._group_fields = fields
        self._group_mode = fields[0] if len(fields) == 1 else " + ".join(fields)
        self._collapsed_groups.clear()
        self._populate()

    def _loader_suffix_for_path(self, path: str | Path, key: str | None = None) -> str:
        label = self._loader_label_for_path(path, key)
        return f" ({label})" if label else ""

    def _path_signature(self, path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
            return (int(st.st_mtime_ns), int(st.st_size if path.is_file() else -1))
        except OSError:
            return None

    def _loader_label_for_path(self, path: str | Path, key: str | None = None) -> str:
        p = Path(path)
        key = key or self._session.key_for_path(p)
        entry = self._session.files.get(key)
        label = ""
        if entry is not None:
            label = entry.meta.loader_label or loader_label(entry.meta.source_format)
        if label:
            return label
        cache_key = str(p)
        sig = self._path_signature(p)
        cached = self._loader_label_cache.get(cache_key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        if not label and detect_format is not None:
            try:
                label = loader_label(detect_format(p))
            except Exception:
                label = ""
        self._loader_label_cache[cache_key] = (sig, label)
        return label

    def _fs_suffix_for_path(self, path: str | Path) -> str:
        return "  [FS]" if self._file_kind_for_path(path) == "FS" else ""

    def _item_label(self, path: str | Path, status: str, key: str | None = None) -> str:
        p = Path(path)
        icon = self.STATUS_ICONS[status]
        extra = self._item_context_suffix(p, key)
        tags = self._tags_for_path(p, key)
        tag_txt = f"  #{', #'.join(tags[:3])}" if tags else ""
        return f"  {icon}  {p.name}{self._loader_suffix_for_path(p, key)}{self._fs_suffix_for_path(p)}{tag_txt}{extra}"

    def _tags_for_path(self, path: str | Path, key: str | None = None) -> list[str]:
        p = Path(path)
        key = key or self._session.key_for_path(p)
        entry = self._session.files.get(key)
        if entry is None:
            return []
        return normalize_tags(getattr(entry.meta, "tags", []))

    def _tag_filter_text(self) -> str:
        return self._tag_filter.text().strip() if hasattr(self, "_tag_filter") else ""

    def _tag_filter_matches(self, path: str | Path) -> bool:
        wanted = [tag.casefold() for tag in normalize_tags(self._tag_filter_text())]
        if not wanted:
            return True
        have = {tag.casefold() for tag in self._tags_for_path(path)}
        return all(tag in have for tag in wanted)

    def _scoped_mappings(self) -> dict[str, dict]:
        return {
            rel: meta.get("mapping", {})
            for rel, meta in (self._session.scoped_logbooks or {}).items()
            if isinstance(meta, dict) and meta.get("mapping")
        }

    def _logbook_manager(self) -> LogbookManager:
        return LogbookManager(
            self._session.logbook_records or [],
            self._session.logbook_mapping or {},
            self._session.folder,
            scoped_mappings=self._scoped_mappings(),
        )

    def _logbook_record_for_path(self, path: str | Path) -> dict | None:
        records = self._session.logbook_records or []
        if not records:
            return None
        p = Path(path)
        cache_key = str(p)
        if cache_key in self._logbook_record_cache:
            return self._logbook_record_cache[cache_key]
        rec_out = self._logbook_manager().find_record_for_path(p)
        self._logbook_record_cache[cache_key] = rec_out
        return rec_out

    def _meta_value_for_path(self, path: str | Path, field: str):
        p = Path(path)
        key = self._session.key_for_path(p)
        entry = self._session.files.get(key)
        if entry is not None:
            meta = entry.meta
            if field == "hv" and meta.hv and meta.hv > 0:
                return float(meta.hv), "session"
            if field == "temperature" and meta.temperature and meta.temperature > 0:
                return float(meta.temperature), "session"
            if field == "polarization" and meta.polarization:
                return meta.polarization, "session"
            if field == "direction" and meta.direction:
                return _format_direction_label(meta.direction), "session"
            if field == "azi" and meta.azi is not None:
                return float(meta.azi), "session"
            if field == "polar" and meta.polar is not None:
                return float(meta.polar), "session"
            if field == "tilt" and meta.tilt is not None:
                return float(meta.tilt), "session"

        rec = self._logbook_record_for_path(p)
        if rec is None:
            return None, ""
        mapping = self._logbook_manager()._mapping_for_record(rec)
        col = mapping.get(field, "")
        if not col:
            return None, ""
        if field in {"hv", "temperature", "azi", "polar", "tilt"}:
            val = _cell_float(rec.get(col))
            if val is not None and np.isfinite(val):
                return float(val), "logbook"
            return None, ""
        val = _format_direction_label(rec.get(col)) if field == "direction" else _cell_text(rec.get(col))
        return (val, "logbook") if val else (None, "")

    def _fmt_float_group(self, label: str, value, unit: str = "", step: float = 0.1) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "Métadonnées inconnues"
        if not np.isfinite(v):
            return "Métadonnées inconnues"
        if step > 0:
            v = round(v / step) * step
        suffix = f" {unit}" if unit else ""
        return f"{label} {v:.1f}{suffix}"

    def _group_part_for_field(self, path: Path, field: str) -> str:
        if field == "Dossier":
            if not self._folder:
                return "."
            rel = path.relative_to(self._folder)
            group = str(rel.parent) if str(rel.parent) != "." else "."
            return self._group_label(group)
        if field == "Labo":
            return self._loader_label_for_path(path) or "Labo inconnu"
        if field == "Type":
            return self._file_kind_for_path(path)
        if field == "hν":
            hv, source = self._meta_value_for_path(path, "hv")
            group = self._fmt_float_group("hν", hv, "eV", step=0.1)
            return f"{group} ({source})" if source and group != "Métadonnées inconnues" else group
        if field == "Température":
            temp, source = self._meta_value_for_path(path, "temperature")
            group = self._fmt_float_group("T", temp, "K", step=0.1)
            return f"{group} ({source})" if source and group != "Métadonnées inconnues" else group
        if field in {"Chemin", "Géométrie"}:
            direction, source = self._meta_value_for_path(path, "direction")
            if not direction:
                return "Chemin inconnu"
            return f"{direction} ({source})" if source else str(direction)
        if field == "Polarisation":
            pol, source = self._meta_value_for_path(path, "polarization")
            if not pol:
                return "Polarisation inconnue"
            return f"Pol {pol} ({source})" if source else f"Pol {pol}"
        return "."

    def _file_kind_for_path(self, path: str | Path) -> str:
        p = Path(path)
        cache_key = str(p)
        sig = self._path_signature(p)
        cached = self._scan_kind_cache.get(cache_key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        kind = "unknown"
        entry = self._session.files.get(self._session.key_for_path(p))
        if entry is not None and entry.meta.source_format == "cls_txt":
            kind = "FS" if self._is_cls_dataset_dir(p) else "BM"
        if kind == "unknown" and detect_scan_kind is not None:
            try:
                kind = detect_scan_kind(p, format_hint=None)
            except Exception:
                kind = "unknown"
        if kind == "unknown":
            if self._is_cls_dataset_dir(p) or p.suffix.lower() == ".zip":
                kind = "FS"
            else:
                kind = "BM"
        self._scan_kind_cache[cache_key] = (sig, kind)
        return kind

    def _group_key_for_path(self, path: Path) -> str:
        fields = list(getattr(self, "_group_fields", None) or [self._group_mode or "Dossier"])
        parts = [self._group_part_for_field(path, field) for field in fields]
        return " / ".join(parts) if parts else "."

    def _group_sort_key(self, group: str):
        if group in {self._folder.name if self._folder else ".", ".", "BM", "FS"}:
            priority = {"BM": 0, "FS": 1, ".": 0, self._folder.name if self._folder else ".": 0}.get(group, 5)
            return (priority, -1.0, group.lower())
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", group)
        if m:
            try:
                return (2, float(m.group(1)), group.lower())
            except ValueError:
                pass
        unknown = "inconn" in group.lower() or "métadonnées" in group.lower()
        return (9 if unknown else 3, -1.0, group.lower())

    def _item_context_suffix(self, path: Path, key: str | None = None) -> str:
        fields = set(getattr(self, "_group_fields", None) or [self._group_mode or "Dossier"])
        if fields == {"Dossier"}:
            return ""
        bits: list[str] = []
        if "hν" not in fields:
            hv, _ = self._meta_value_for_path(path, "hv")
            if hv is not None:
                bits.append(f"hν={float(hv):.1f}")
        if "Température" not in fields:
            temp, _ = self._meta_value_for_path(path, "temperature")
            if temp is not None:
                bits.append(f"T={float(temp):.1f}")
        if "Chemin" not in fields and "Géométrie" not in fields:
            direction, _ = self._meta_value_for_path(path, "direction")
            if direction:
                bits.append(str(direction))
        if "Polarisation" not in fields:
            pol, _ = self._meta_value_for_path(path, "polarization")
            if pol:
                bits.append(f"Pol={pol}")
        if not bits:
            return ""
        return "  " + "  ".join(bits[:2])

    def _update_summary(self, paths: list[Path]):
        total = len(paths)
        counts = {"unloaded": 0, "loaded": 0, "fitted": 0}
        loaders: dict[str, int] = {}
        for p in paths:
            key = self._session.key_for_path(p)
            counts[self._file_status(key)] += 1
            label = self._loader_label_for_path(p, key) or "?"
            loaders[label] = loaders.get(label, 0) + 1
        loader_txt = ", ".join(f"{k}:{v}" for k, v in sorted(loaders.items())) if loaders else "—"
        self._lbl_summary.setText(
            f"{total} éléments  •  "
            f"{counts['loaded']} chargés  •  {counts['fitted']} fittés  •  {loader_txt}"
        )

    def _describe_item(self, item: QListWidgetItem | None) -> str:
        if item is None:
            return "Sélectionne un fichier à charger"
        group = item.data(Qt.ItemDataRole.UserRole + 2)
        if group is not None:
            return "Dossier de groupe : double-clic ou Charger pour ouvrir/réduire"
        path_txt = item.data(Qt.ItemDataRole.UserRole)
        if not path_txt:
            return "Sélectionne un fichier à charger"
        p = Path(path_txt)
        key = item.data(Qt.ItemDataRole.UserRole + 1) or self._session.key_for_path(p)
        status = self._file_status(key)
        entry = self._session.files.get(key)
        loader = self._loader_label_for_path(p, key) or "inconnu"
        kind = "FS" if self._fs_suffix_for_path(p) else "BM"
        try:
            rel = str(p.relative_to(self._folder)) if self._folder else str(p)
        except Exception:
            rel = str(p)
        bits = [f"{p.name}", rel, f"{kind} {loader}", f"état: {status}"]
        hv, hv_src = self._meta_value_for_path(p, "hv")
        temp, temp_src = self._meta_value_for_path(p, "temperature")
        pol, pol_src = self._meta_value_for_path(p, "polarization")
        direction, dir_src = self._meta_value_for_path(p, "direction")
        azi, azi_src = self._meta_value_for_path(p, "azi")
        polar, p_src = self._meta_value_for_path(p, "polar")
        tilt, t_src = self._meta_value_for_path(p, "tilt")
        if hv is not None:
            bits.append(f"hν={float(hv):.1f} eV ({hv_src})")
        if temp is not None:
            bits.append(f"T={float(temp):.1f} K ({temp_src})")
        if pol:
            bits.append(f"pol={pol} ({pol_src})")
        if direction:
            bits.append(f"direction={direction} ({dir_src})")
        geom = []
        if azi is not None and abs(float(azi)) > 1e-9:
            geom.append(f"azi={float(azi):.1f}°")
        if polar is not None and abs(float(polar)) > 1e-9:
            geom.append(f"polar={float(polar):.1f}°")
        if tilt is not None and abs(float(tilt)) > 1e-9:
            geom.append(f"tilt={float(tilt):.1f}°")
        if geom:
            sources = sorted({s for s in (azi_src, p_src, t_src) if s})
            src_txt = f" ({'+'.join(sources)})" if sources else ""
            bits.append("géom: " + ", ".join(geom) + src_txt)
        if entry is not None:
            if entry.fit_result:
                bits.append("fit enregistré")
            tags = self._tags_for_path(p, key)
            if tags:
                bits.append("tags: " + ", ".join(tags))
        return "\n".join(bits)

    def _refresh_selection_state(self):
        item = self._list.currentItem()
        has_path = bool(item and item.data(Qt.ItemDataRole.UserRole))
        has_group = bool(item and item.data(Qt.ItemDataRole.UserRole + 2) is not None)
        if has_path:
            self._btn_load.setText("Charger ce fichier")
            self._btn_load.setEnabled(True)
            self._btn_load.setToolTip("Charge le fichier sélectionné dans la session courante.")
        elif has_group:
            self._btn_load.setText("Ouvrir/réduire le groupe")
            self._btn_load.setEnabled(True)
            self._btn_load.setToolTip("Ouvre ou réduit le groupe sélectionné dans le navigateur.")
        else:
            self._btn_load.setText("Charger la sélection")
            self._btn_load.setEnabled(False)
            self._btn_load.setToolTip("Choisir un fichier ou un groupe dans la liste pour activer cette action.")
        self._lbl_selection.setText(self._describe_item(item))

    def _add_header(self, group: str, n_items: int):
        label = self._group_label(group)
        collapsed = group in self._collapsed_groups
        arrow = ">" if collapsed else "v"
        item = QListWidgetItem(f"{arrow}  {label}  ({n_items})")
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setData(Qt.ItemDataRole.UserRole + 2, group)
        item.setToolTip("Double-cliquer pour ouvrir/réduire ce dossier")
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setForeground(QColor("#9ab"))
        self._list.addItem(item)

    def _populate(self):
        return self._browser_ctrl._populate()
    def _file_status(self, key: str) -> str:
        if key not in self._session.files:
            return "unloaded"
        return self._session.files[key].status

    def refresh_item(self, filename_or_key: str):
        return self._browser_ctrl.refresh_item(filename_or_key)
    def _toggle_group(self, group: str):
        return self._browser_ctrl._toggle_group(group)
    def _on_double_click(self, item):
        return self._browser_ctrl._on_double_click(item)
    def _on_selection_change(self, current, _):
        return self._browser_ctrl._on_selection_change(current, _)
    def _load_selected(self):
        return self._browser_ctrl._load_selected()
    def navigate(self, delta: int):
        return self._browser_ctrl.navigate(delta)
    def select_file(self, path: str) -> bool:
        return self._browser_ctrl.select_file(path)
