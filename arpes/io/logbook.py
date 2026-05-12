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
    sources: dict[str, str] = field(default_factory=dict)

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
        for rec in self.records:
            rel = str(rec.get("_subfolder_rel") or "").strip()
            if not rel:
                continue
            if not _record_in_subfolder_scope(rec, path, self.session_folder):
                continue
            file_col = self._mapping_for_record(rec).get("file", "")
            if file_col and _record_matches_path(rec.get(file_col), path, self.session_folder):
                return rec
        # 2. Pass : records sans scope (global, fallback)
        file_col = self.mapping.get("file", "")
        if not file_col:
            return None
        for rec in self.records:
            if str(rec.get("_subfolder_rel") or "").strip():
                continue
            if _record_matches_path(rec.get(file_col), path, self.session_folder):
                return rec
        return None

    def values_from_record(self, record: dict | None) -> LogbookAppliedValues:
        if not record:
            return LogbookAppliedValues()
        m = self._mapping_for_record(record)
        out = LogbookAppliedValues()

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
    s = _cell_text(value)
    if not s:
        return ""
    s = re.sub(r"(?i)\bgamma\b", "Γ", s)
    s = re.sub(r"(?i)\bgamma(?=[A-Z0-9])", "Γ", s)
    s = re.sub(r"(?i)\bg(?=\s*(?:$|[-_/→> ]))", "Γ", s)
    if re.fullmatch(r"(?i)g[mkxy][a-z0-9_-]*", s.strip()):
        s = "Γ" + s.strip()[1:]
    return s.strip()


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


def _sniff_columns_by_content(df, columns: list[str]) -> dict[str, str]:
    """Devine les colonnes à partir des valeurs quand keyword match échoue.

    Règles :
      - hv : numérique, médiane dans [4, 200] eV, ≥60% de valeurs finies > 0
      - temperature : numérique, médiane dans [3, 500] K, ≥60% de valeurs finies > 0
      - direction : strings contenant Γ/G/M/X/K/Σ avec '-' ou '→'
      - polarization : strings dans {LH, LV, RC, LC, σ, π, s, p, ...}
                       ou numérique dans [0, 360]
      - file : strings avec extension (.ibw/.zip/.txt) OU haute cardinalité
               (≥80% valeurs uniques)

    La fonction ne retourne que des mappings non vides.
    """
    if df is None:
        return {}
    try:
        n_rows = len(df)
    except TypeError:
        n_rows = 0
    if n_rows < 3:
        # Pas assez de lignes pour deviner avec confiance.
        return {}
    out: dict[str, str] = {}
    POL_TOKENS = {"LH", "LV", "RC", "LC", "σ", "π", "s", "p", "S", "P",
                  "C+", "C-", "RCP", "LCP", "lin", "circ"}
    DIR_HINTS = ("Γ", "GAMMA", "M", "X", "K", "Σ", "SIGMA")
    FILE_EXT = (".ibw", ".zip", ".txt", ".dat", ".h5", ".hdf5", ".nxs", ".pxt")

    def col_values(col):
        try:
            return df[col].dropna().tolist()
        except Exception:
            return []

    def strict_numeric_value(v):
        """Retourne float seulement si la cellule est un nombre pur."""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
            except Exception:
                return None
            return f if np.isfinite(f) else None
        s = str(v).strip().replace(",", ".")
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if np.isfinite(f) else None

    def numeric_stats(values):
        nums = []
        for v in values:
            n = strict_numeric_value(v)
            if n is not None:
                nums.append(n)
        if not nums:
            return None, 0.0
        finite_pos = [n for n in nums if n > 0]
        if not finite_pos:
            return None, 0.0
        return float(np.median(finite_pos)), len(finite_pos) / max(len(values), 1)

    for col in columns:
        values = col_values(col)
        if not values:
            continue
        # hv : numérique [4, 200]
        if "hv" not in out:
            med, ratio = numeric_stats(values)
            if med is not None and 4.0 <= med <= 200.0 and ratio >= 0.6:
                out["hv"] = col
                continue
        # temperature : numérique [3, 500] (mais pas hv)
        if "temperature" not in out and out.get("hv") != col:
            med, ratio = numeric_stats(values)
            if med is not None and 3.0 <= med <= 500.0 and ratio >= 0.6:
                out["temperature"] = col
                continue
        # direction : strings avec Γ/M/X
        if "direction" not in out:
            txts = [str(v).upper() for v in values if str(v).strip()]
            if txts:
                hits = sum(1 for t in txts if any(h in t for h in DIR_HINTS) and ("-" in t or "→" in t))
                if hits / max(len(txts), 1) >= 0.4:
                    out["direction"] = col
                    continue
        # polarization
        if "polarization" not in out:
            txts = [str(v).strip() for v in values if str(v).strip()]
            if txts:
                hits = sum(1 for t in txts if t in POL_TOKENS or t.upper() in POL_TOKENS)
                if hits / max(len(txts), 1) >= 0.5:
                    out["polarization"] = col
                    continue
        # file : strings avec extension fichier (.ibw, .zip, .txt...) requise.
        # Cardinalité seule ne suffit pas — trop de faux positifs sur codes échantillon.
        if "file" not in out:
            txts = [str(v) for v in values if str(v).strip()]
            if txts:
                ext_hits = sum(1 for t in txts if any(t.lower().endswith(e) for e in FILE_EXT))
                if ext_hits / max(len(txts), 1) >= 0.5:
                    out["file"] = col
                    continue
    return out


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
        "temperature": _pick_column(columns, [
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
        "azi": _pick_column(columns, [
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

    NUMERIC_KEYS = {"hv", "temperature", "azi", "polar", "tilt", "crystal_a_angstrom"}
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


def _path_match_tokens(path: str | Path, session_folder: Path | None) -> list[str]:
    p = Path(path)
    tokens = [p.name, p.stem]
    if session_folder is not None:
        try:
            rel = p.resolve().relative_to(session_folder.resolve())
            tokens.extend([str(rel), rel.name, rel.stem])
        except Exception:
            pass
    return sorted({t for t in tokens if t}, key=len, reverse=True)


def _extract_measurement_numbers(text: str) -> set[int]:
    out: set[int] = set()
    if not text:
        return out
    for pat in (r"(\d{3,4})w", r"_(\d{1,4})(?:\D|$)"):
        for m in re.finditer(pat, text):
            try:
                out.add(int(m.group(1)))
            except ValueError:
                pass
    return out


def _path_measurement_numbers(path: str | Path) -> set[int]:
    p = Path(path)
    out: set[int] = set()
    for text in (p.name, p.stem):
        out |= _extract_measurement_numbers(text)
    return out


def _record_in_subfolder_scope(record: Any, path: str | Path, session_folder: Path | None) -> bool:
    """Si le record a un champ `_subfolder_rel`, vérifie que `path` y est inclus.

    Permet d'attacher des logbooks à un sous-dossier précis (ex: CA041 vs CA046)
    sans confusion entre fichiers de même nom dans deux subdirs.
    """
    if not isinstance(record, dict):
        return True
    subfolder = str(record.get("_subfolder_rel") or "").strip()
    if not subfolder:
        return True
    if session_folder is None:
        return True
    try:
        p_resolved = Path(path).resolve()
        scope_resolved = (Path(session_folder) / subfolder).resolve()
    except Exception:
        return True
    try:
        p_resolved.relative_to(scope_resolved)
        return True
    except ValueError:
        return False


_ALNUM_KEY_RE = re.compile(r"^\s*([A-Za-z]+)\s*0*(\d+)\s*$")


def _alnum_label_key(text: Any) -> tuple[str, int] | None:
    """('BM3' | 'BM03' | 'bm 3') -> ('bm', 3). Sinon None.

    Permet de matcher un nom de scan logbook ('BM3') avec un fichier disque
    zéro-paddé ('BM03') ou inversement — convention fréquente côté beamline.
    """
    m = _ALNUM_KEY_RE.match(str(text or ""))
    if not m:
        return None
    try:
        return m.group(1).lower(), int(m.group(2))
    except ValueError:
        return None


def _record_matches_path(record_value: Any, path: str | Path, session_folder: Path | None) -> bool:
    value = _cell_text(record_value)
    if not value:
        return False
    path_nums = _path_measurement_numbers(path)
    num = _cell_float(record_value)
    if num is not None and abs(num - round(num)) < 1e-9:
        if int(round(num)) in path_nums:
            return True
    cell_nums = _extract_measurement_numbers(value)
    if cell_nums and path_nums and (cell_nums & path_nums):
        return True
    value_norm = value.lower()
    value_key = _alnum_label_key(value)
    for token in _path_match_tokens(path, session_folder):
        token_norm = token.lower()
        if value_norm == token_norm:
            return True
        if value_key is not None and _alnum_label_key(token) == value_key:
            return True
        pat = r"(?<![A-Za-z0-9])" + re.escape(token_norm) + r"(?![A-Za-z0-9])"
        if re.search(pat, value_norm):
            return True
    return False
