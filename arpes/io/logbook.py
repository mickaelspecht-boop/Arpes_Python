"""Helpers purs pour parsing et matching de logbooks ARPES."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
import re
import unicodedata

import numpy as np


@dataclass
class LogbookAppliedValues:
    """Valeurs extraites d'une ligne logbook, sans effet UI."""

    hv: float | None = None
    temperature: float | None = None
    polarization: str = ""
    direction: str = ""
    azi: float | None = None
    polar: float | None = None
    tilt: float | None = None
    formula: str = ""
    mp_id: str = ""
    crystal_a_angstrom: float | None = None
    crystal_b_angstrom: float | None = None
    crystal_c_angstrom: float | None = None
    work_function_eV: float | None = None
    sources: dict[str, str] = field(default_factory=dict)
    # Provenance (NOT data — excluded from has_any): which scoped logbook sheet /
    # subfolder fed this record. Empty for a global (unscoped) logbook.
    matched_sheet: str = ""
    matched_subfolder: str = ""

    def has_any(self) -> bool:
        return any([
            self.hv is not None,
            self.temperature is not None,
            bool(self.polarization),
            bool(self.direction),
            self.azi is not None,
            self.polar is not None,
            self.tilt is not None,
            bool(self.formula),
            bool(self.mp_id),
            self.crystal_a_angstrom is not None,
            self.crystal_b_angstrom is not None,
            self.crystal_c_angstrom is not None,
            self.work_function_eV is not None,
        ])


class LogbookManager:
    """Decision pure autour du logbook : matching et extraction de valeurs."""

    def __init__(
        self,
        records: list[dict] | None = None,
        mapping: dict[str, str] | None = None,
        session_folder: str | Path | None = None,
        scoped_mappings: dict[str, dict] | None = None,
    ):
        self.records = list(records or [])
        self.mapping = dict(mapping or {})
        # rel_subdir -> mapping propre à ce logbook scopé (noms de colonnes
        # potentiellement différents de ceux du logbook global).
        self.scoped_mappings = {str(k): dict(v) for k, v in (scoped_mappings or {}).items() if v}
        self.session_folder = Path(session_folder) if session_folder is not None else None

    def _mapping_for_record(self, record: dict | None) -> dict[str, str]:
        if record:
            rel = str(record.get("_subfolder_rel") or "").strip()
            if rel and rel in self.scoped_mappings:
                return self.scoped_mappings[rel]
        return self.mapping

    def find_record_for_path(self, path: str | Path) -> dict | None:
        # 1. Pass : records avec scope subfolder (prioritaire si présents),
        #    matchés avec le mapping 'file' propre à leur logbook scopé.
        scoped_matches: list[tuple[dict, str]] = []
        for rec in self.records:
            rel = str(rec.get("_subfolder_rel") or "").strip()
            if not rel:
                continue
            if not _record_in_subfolder_scope(rec, path, self.session_folder):
                continue
            file_col = self._mapping_for_record(rec).get("file", "")
            if file_col and _record_matches_path(rec.get(file_col), path, self.session_folder):
                scoped_matches.append((rec, file_col))
        if scoped_matches:
            return _best_record_for_path(scoped_matches, path, self.session_folder)
        # 2. Pass : records sans scope (global, fallback)
        file_col = self.mapping.get("file", "")
        if not file_col:
            return None
        global_matches: list[tuple[dict, str]] = []
        for rec in self.records:
            if str(rec.get("_subfolder_rel") or "").strip():
                continue
            if _record_matches_path(rec.get(file_col), path, self.session_folder):
                global_matches.append((rec, file_col))
        return _best_record_for_path(global_matches, path, self.session_folder)

    def has_scoped_records_for_path(self, path: str | Path) -> bool:
        """True iff ≥1 scoped record (with `_subfolder_rel`) covers `path`'s
        subfolder. Distinguishes "scoped session, file unmatched" (→ surface a
        visible warning) from "global/no logbook" (→ stay silent). Key primitive
        for the no-silent-fallback robustness rule."""
        for rec in self.records:
            if not str(rec.get("_subfolder_rel") or "").strip():
                continue
            if _record_in_subfolder_scope(rec, path, self.session_folder):
                return True
        return False

    def values_from_record(self, record: dict | None) -> LogbookAppliedValues:
        if not record:
            return LogbookAppliedValues()
        m = self._mapping_for_record(record)
        out = LogbookAppliedValues()
        # Provenance: surface which scoped sheet/subfolder this record came from.
        out.matched_subfolder = str(record.get("_subfolder_rel") or "")
        out.matched_sheet = str(record.get("_sheet_name") or "")

        hv = _cell_float(record.get(m.get("hv", "")))
        if hv is not None and hv > 0:
            out.hv = float(hv)
            out.sources["hv"] = "logbook"

        temp_col = m.get("temperature", "")
        temp = _cell_float(record.get(temp_col)) if temp_col else None
        if temp is not None:
            out.temperature = float(temp)
            out.sources["temperature"] = "logbook"

        pol_col = m.get("polarization", "")
        pol = _cell_text(record.get(pol_col)) if pol_col else ""
        if pol:
            out.polarization = pol
            out.sources["polarization"] = "logbook"

        azi_col = m.get("azi", "")
        azi = _cell_float(record.get(azi_col)) if azi_col else None
        if azi is not None and np.isfinite(azi):
            out.azi = float(azi)
            out.sources["azi"] = "logbook"

        dir_col = m.get("direction", "")
        direction = _cell_text(record.get(dir_col)) if dir_col else ""
        if direction:
            out.direction = _format_direction_label(direction)
            out.sources["direction"] = "logbook"

        polar_col = m.get("polar", "")
        polar = _cell_float(record.get(polar_col)) if polar_col else None
        if polar is not None and np.isfinite(polar):
            out.polar = float(polar)
            out.sources["polar"] = "logbook"

        tilt_col = m.get("tilt", "")
        tilt = _cell_float(record.get(tilt_col)) if tilt_col else None
        if tilt is not None and np.isfinite(tilt):
            out.tilt = float(tilt)
            out.sources["tilt"] = "logbook"

        formula_col = m.get("formula", "")
        formula = _cell_text(record.get(formula_col)) if formula_col else ""
        if formula:
            out.formula = formula
            out.sources["formula"] = "logbook"

        mp_id_col = m.get("mp_id", "")
        mp_id_raw = _cell_text(record.get(mp_id_col)) if mp_id_col else ""
        mp_id = _normalize_mp_id(mp_id_raw)
        if mp_id:
            out.mp_id = mp_id
            out.sources["mp_id"] = "logbook"

        a_col = m.get("crystal_a_angstrom", "")
        a_val = _cell_float(record.get(a_col)) if a_col else None
        if a_val is not None and np.isfinite(a_val) and a_val > 0:
            out.crystal_a_angstrom = float(a_val)
            out.sources["crystal_a_angstrom"] = "logbook"

        b_col = m.get("crystal_b_angstrom", "")
        b_val = _cell_float(record.get(b_col)) if b_col else None
        if b_val is not None and np.isfinite(b_val) and b_val > 0:
            out.crystal_b_angstrom = float(b_val)
            out.sources["crystal_b_angstrom"] = "logbook"

        c_col = m.get("crystal_c_angstrom", "")
        c_val = _cell_float(record.get(c_col)) if c_col else None
        if c_val is not None and np.isfinite(c_val) and c_val > 0:
            out.crystal_c_angstrom = float(c_val)
            out.sources["crystal_c_angstrom"] = "logbook"

        wf_col = m.get("work_function_eV", "")
        wf_val = _cell_float(record.get(wf_col)) if wf_col else None
        if wf_val is not None and np.isfinite(wf_val) and wf_val > 0:
            out.work_function_eV = float(wf_val)
            out.sources["work_function_eV"] = "logbook"

        return out

    def values_for_path(self, path: str | Path) -> LogbookAppliedValues:
        return self.values_from_record(self.find_record_for_path(path))

    def apply_to_entry(self, entry: Any, path: str | Path) -> LogbookAppliedValues:
        values = self.values_for_path(path)
        if values.hv is not None:
            entry.meta.hv = values.hv
        if values.temperature is not None:
            entry.meta.temperature = values.temperature
        if values.polarization:
            entry.meta.polarization = values.polarization
        if values.direction:
            entry.meta.direction = values.direction
        if values.azi is not None:
            entry.meta.azi = values.azi
        if values.polar is not None:
            entry.meta.polar = values.polar
        if values.tilt is not None:
            entry.meta.tilt = values.tilt
        if values.formula:
            entry.meta.formula = values.formula
        if values.mp_id:
            entry.meta.mp_id = values.mp_id
        if values.crystal_a_angstrom is not None:
            entry.meta.crystal_a_angstrom = float(values.crystal_a_angstrom)
        if values.crystal_b_angstrom is not None:
            entry.meta.crystal_b_angstrom = float(values.crystal_b_angstrom)
        if values.crystal_c_angstrom is not None:
            entry.meta.crystal_c_angstrom = float(values.crystal_c_angstrom)
        if values.work_function_eV is not None:
            entry.meta.work_function_eV = float(values.work_function_eV)
        return values


_MP_ID_PATTERN = re.compile(r"^mp-\d+$")


def _normalize_mp_id(raw: str) -> str:
    """Normalise un MPID logbook : 'mp-149', 'MP-149', '149' → 'mp-149'.

    Retourne chaîne vide si format invalide (silencieusement, pas de raise).
    """
    text = (raw or "").strip().lower()
    if not text:
        return ""
    if _MP_ID_PATTERN.match(text):
        return text
    digits = re.match(r"^(\d+)$", text)
    if digits:
        return f"mp-{digits.group(1)}"
    return ""


def _norm_text(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("ν", "nu").replace("Ν", "nu")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _cell_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", ".")
        m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if not m:
            return None
        value = m.group(0)
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _pick_column(columns: list[str], groups: list[list[str]]) -> str:
    normalized = {c: _norm_text(c) for c in columns}
    for group in groups:
        keys = [_norm_text(k) for k in group]
        for col, name in normalized.items():
            if all(k in name for k in keys):
                return col
    return ""


def _pick_exact_column(columns: list[str], aliases: set[str]) -> str:
    normalized_aliases = {_norm_text(a) for a in aliases}
    for col in columns:
        if _norm_text(col) in normalized_aliases:
            return col
    return ""


def _pick_direction_column(columns: list[str]) -> str:
    col = _pick_column(columns, [
        ["direction"], ["direct"], ["cut"], ["coupe"], ["chemin"],
        ["zdb"], ["zone", "boundary"], ["brillouin", "zone"],
        ["high", "symmetry"], ["symmetry", "path"], ["symmetry"],
        ["k", "path"], ["scan", "path"], ["path", "bz"],
        ["orientation"], ["geometry"], ["geometrie"], ["geom"],
        ["trajectory"], ["traj"], ["ligne"], ["line"],
        ["gamma"], ["gamme"],
    ])
    if col:
        return col
    aliases = {
        "g", "gamma", "gammapath", "gammaline", "gammacut",
        "highsymmetry", "highsymmetrypath", "zdb", "bzpath",
    }
    for candidate in columns:
        if _norm_text(candidate) in aliases:
            return candidate
    return ""


def _format_direction_label(value: Any) -> str:
    """Canonicalize a cut-direction cell (delegates to hs_directions)."""
    from arpes.physics.hs_directions import normalize_direction_label
    return normalize_direction_label(_cell_text(value))


def _best_record_for_path(
    matches: list[tuple[dict, str]],
    path: str | Path,
    session_folder: Path | None,
) -> dict | None:
    """Disambiguate duplicate File cells with another field matching the path.

    Some folder logbooks use a constant file basename for all rows and put the
    real per-scan label in another column, e.g. ``Spectrum Name = kz_100.0``.
    Returning the first file match would assign the first hν to every KZ file.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    path_text = " ".join(_path_match_tokens(path, session_folder)).lower()
    best_rec = matches[0][0]
    best_score = -1
    for rec, file_col in matches:
        score = 0
        for col, value in rec.items():
            if col == file_col or str(col).startswith("_"):
                continue
            text = _cell_text(value).lower()
            if len(text) < 3:
                continue
            if text in path_text:
                weight = len(text)
                col_norm = _norm_text(col)
                if any(ch.isdigit() for ch in text) and ("_" in text or "." in text or "-" in text):
                    weight += 1000
                if "spectrum" in col_norm or "scan" in col_norm or "name" in col_norm:
                    weight += 100
                score = max(score, weight)
        if score > best_score:
            best_rec = rec
            best_score = score
    return best_rec


def _infer_legacy_measurement_plan_mapping(columns: list[str]) -> dict[str, str]:
    normalized = {c: _norm_text(c) for c in columns}
    by_norm = {v: k for k, v in normalized.items()}
    has_legacy_shape = "num" in by_norm and (
        "energy" in by_norm or "measurementtype" in by_norm or "measurement type" in by_norm
    )
    if not has_legacy_shape:
        return {}
    return {
        "file": by_norm.get("num", ""),
        "hv": by_norm.get("energy", ""),
        "temperature": by_norm.get("temp", ""),
        "polarization": by_norm.get("pol", ""),
        "azi": "",
        "polar": by_norm.get("polar", ""),
        "tilt": "",
        "direction": by_norm.get("direction", ""),
    }


# `_sniff_columns_by_content` lives in `logbook_mapping.py` (split to keep this
# file under the 700-LOC cap). Imported here; no import cycle (it is numpy-only).
from arpes.io.logbook_mapping import _sniff_columns_by_content  # noqa: E402


def _coerce_float(value):
    """Conversion numérique tolérante (utilisée par sniffer + parser)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            n = float(value)
        except (TypeError, ValueError):
            return None
        return n if np.isfinite(n) else None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        n = float(m.group(0))
    except ValueError:
        return None
    return n if np.isfinite(n) else None


def _infer_logbook_mapping(columns: list[str], df=None) -> dict[str, str]:
    mapping = {
        "file": _pick_column(columns, [
            ["file"], ["filename"], ["fichier"], ["scan"], ["measurement"],
            ["measure"], ["name"], ["nom"], ["run"], ["sample"],
        ]),
        "hv": _pick_column(columns, [
            ["hv"], ["hnu"], ["photon", "energy"], ["photon", "energie"],
            ["energy", "ev"], ["energie", "ev"], ["hn"],
        ]),
        "temperature": _pick_exact_column(columns, {
            "temp", "TEMP", "Temp",
            "t", "T", "t_k", "T_K", "t(k)", "T(K)", "t [k]", "T [K]",
            "tk", "TK", "Tk",
            "temperature", "Temperature", "TEMPERATURE",
            "sample temperature", "Sample Temperature", "sample_temperature",
        }) or _pick_column(columns, [
            ["sample", "temperature"], ["temperature"], ["temp"], ["t", "sample"], ["t", "k"],
        ]),
        "polarization": _pick_exact_column(columns, {
            "pol", "polarization", "polarisation", "light polarization",
            "light polarisation", "photon polarization", "photon polarisation",
            "e vector", "e-vector", "lv", "lh",
        }) or _pick_column(columns, [
            ["polarization"], ["polarisation"],
            ["pol"], ["light", "polarization"], ["light", "polarisation"],
            ["photon", "polarization"], ["photon", "polarisation"],
            ["e", "vector"], ["e-vector"], ["lv"], ["lh"],
        ]),
        "azi": _pick_exact_column(columns, {
            "az", "Az", "AZ",
        }) or _pick_column(columns, [
            ["azi"], ["azimuth"], ["azimut"], ["phi", "azimuth"],
        ]),
        "polar": _pick_exact_column(columns, {
            "polar", "p", "p axis", "p-axis", "p axis deg", "p-axis deg",
        }) or _pick_column(columns, [
            ["theta"], ["polar", "angle"], ["polar", "deg"],
            ["manip", "p"],
        ]),
        "tilt": _pick_column(columns, [
            ["phi"], ["tilt"], ["manip", "t"],
        ]),
        "direction": _pick_direction_column(columns),
        "formula": _pick_exact_column(columns, {
            "formula", "formule", "compound", "compose", "material", "materiau",
        }) or _pick_column(columns, [
            ["formula"], ["formule"], ["compound"], ["material"],
        ]),
        "mp_id": _pick_exact_column(columns, {
            "mp_id", "mp-id", "mpid", "materials project id", "mp",
        }) or _pick_column(columns, [
            ["mp", "id"], ["materials", "project", "id"],
        ]),
        "crystal_a_angstrom": _pick_exact_column(columns, {
            "a", "a (a)", "a (angstrom)", "a_angstrom", "a_a", "lattice_a",
            "parametre a", "parametre_a", "lattice a", "a [a]",
        }) or _pick_column(columns, [
            ["lattice", "a"], ["parametre", "a"], ["a", "angstrom"],
        ]),
        "crystal_b_angstrom": _pick_exact_column(columns, {
            "b", "b (a)", "b (angstrom)", "b_angstrom", "b_a", "lattice_b",
            "parametre b", "parametre_b", "lattice b", "b [a]",
        }) or _pick_column(columns, [
            ["lattice", "b"], ["parametre", "b"], ["b", "angstrom"],
        ]),
        "crystal_c_angstrom": _pick_exact_column(columns, {
            "c", "c (a)", "c (angstrom)", "c_angstrom", "c_a", "lattice_c",
            "parametre c", "parametre_c", "lattice c", "c [a]",
        }) or _pick_column(columns, [
            ["lattice", "c"], ["parametre", "c"], ["c", "angstrom"],
        ]),
        "work_function_eV": _pick_exact_column(columns, {
            "phi", "φ", "work function", "work_function", "workfunction",
            "work function ev", "work_function_ev", "fonction travail",
            "fonction de travail", "wf", "wf_ev",
        }) or _pick_column(columns, [
            ["work", "function"], ["fonction", "travail"], ["phi"],
            ["workfunction"], ["wf"],
        ]),
    }
    legacy = _infer_legacy_measurement_plan_mapping(columns)
    for key, val in legacy.items():
        current_norm = _norm_text(mapping.get(key, ""))
        legacy_should_override = key == "file" and current_norm in {"measurementtype", "sample"}
        if val and (not mapping.get(key) or legacy_should_override):
            mapping[key] = val
    if df is not None:
        # Sniff par contenu uniquement pour les clés où aucun keyword n'a matché.
        # On ne second-guess pas les keyword matches : un user peut nommer
        # "Temp" une colonne d'enums ("Low T", "High T") et c'est légitime.
        sniffed = _sniff_columns_by_content(df, columns)
        for key, val in sniffed.items():
            if not mapping.get(key) and val:
                mapping[key] = val
    return mapping


def _drop_implausible_mappings(mapping: dict[str, str], df) -> dict[str, str]:
    """Rejette les colonnes mappées dont le contenu ne correspond pas au type.

    Évite p.ex. de mapper hv sur une colonne de strings juste parce que le nom
    contient un mot-clé sémantiquement proche.
    """
    try:
        n_rows = len(df)
    except TypeError:
        return mapping
    if n_rows < 3:
        return mapping
    out = dict(mapping)

    def values_of(col):
        try:
            return [v for v in df[col].tolist() if v is not None and str(v).strip()]
        except Exception:
            return []

    def strict_numeric(v) -> bool:
        """True si v est un float pur (pas de lettre/code, signes/virgule ok)."""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                return np.isfinite(float(v))
            except Exception:
                return False
        s = str(v).strip().replace(",", ".")
        if not s:
            return False
        try:
            n = float(s)
        except ValueError:
            return False
        return bool(np.isfinite(n))

    def numeric_ratio(values):
        if not values:
            return 0.0
        n = sum(1 for v in values if strict_numeric(v))
        return n / len(values)

    NUMERIC_KEYS = {
        "hv", "temperature", "azi", "polar", "tilt",
        "crystal_a_angstrom", "crystal_b_angstrom", "crystal_c_angstrom",
        "work_function_eV",
    }
    for key in list(out.keys()):
        col = out[key]
        if not col:
            continue
        values = values_of(col)
        if not values:
            out[key] = ""
            continue
        if key in NUMERIC_KEYS:
            if numeric_ratio(values) < 0.4:
                out[key] = ""
    return out


from arpes.io.logbook_matching import (  # noqa: E402
    _alnum_label_key,
    _extract_measurement_numbers,
    _path_match_tokens,
    _path_measurement_numbers,
    _record_in_subfolder_scope,
    _record_matches_path as _record_matches_path_impl,
)


def _record_matches_path(record_value: Any, path: str | Path, session_folder: Path | None) -> bool:
    return _record_matches_path_impl(
        record_value,
        path,
        session_folder,
        cell_text=_cell_text,
        cell_float=_cell_float,
    )
