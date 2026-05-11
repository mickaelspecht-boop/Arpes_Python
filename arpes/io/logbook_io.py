"""Lecture de logbooks ARPES — parsing pur, sans PyQt.

La sélection interactive de feuille/table/colonnes reste dans l'UI. Ce module
contient uniquement les heuristiques de lecture, détection de header, mapping
et propagation de contexte entre lignes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from arpes.io.logbook import _cell_float, _cell_text, _infer_logbook_mapping


@dataclass
class LogbookReadResult:
    records: list[dict]
    mapping: dict[str, str]
    sheet_name: str = ""


def read_delimited_logbook_raw(pd, path: Path):
    """Lecture brute CSV/TSV pour anciens logbooks avec titre avant header."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except Exception:
        lines = []
    best_manual = None
    for sep in (";", ",", "\t"):
        rows = [line.split(sep) for line in lines if line.strip()]
        width = max((len(r) for r in rows), default=0)
        if width > 1:
            padded = [r + [None] * (width - len(r)) for r in rows]
            raw = pd.DataFrame(padded)
            if best_manual is None or len(raw.columns) > len(best_manual.columns):
                best_manual = raw
    if best_manual is not None:
        return best_manual

    attempts = [
        {"sep": ";", "header": None},
        {"sep": ",", "header": None},
        {"sep": "\t", "header": None},
        {"sep": None, "engine": "python", "header": None},
    ]
    best = None
    for kwargs in attempts:
        try:
            raw = pd.read_csv(path, **kwargs)
            if len(raw.columns) > 1:
                if best is None or len(raw.columns) > len(best.columns):
                    best = raw
        except Exception:
            continue
    return best


def inherit_logbook_context(records: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Propage direction/polarisation/azi quand les cellules suivantes sont vides."""
    azi_col = mapping.get("azi", "")
    dir_col = mapping.get("direction", "")
    pol_col = mapping.get("polarization", "")
    last_azi = None
    last_dir = ""
    last_pol = ""
    out = []
    for rec in records:
        rec = dict(rec)

        if dir_col:
            direct = _cell_text(rec.get(dir_col))
            if direct:
                last_dir = direct
            elif last_dir:
                rec[dir_col] = last_dir

        if pol_col:
            pol = _cell_text(rec.get(pol_col))
            if pol:
                last_pol = pol
            elif last_pol:
                rec[pol_col] = last_pol

        if azi_col:
            azi = _cell_float(rec.get(azi_col))
            if azi is not None and np.isfinite(azi):
                last_azi = float(azi)
            elif last_azi is not None:
                rec[azi_col] = last_azi

        out.append(rec)
    return out


def excel_header_candidates(raw) -> list[int]:
    candidates: list[int] = []
    for row_idx in range(min(len(raw), 120)):
        values = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
        nonempty = [v for v in values if v]
        if len(nonempty) >= 2:
            candidates.append(row_idx)
    return candidates


def excel_table_from_header(raw, row_idx: int):
    headers = [_cell_text(v) for v in raw.iloc[row_idx].tolist()]
    cols = [h if h else f"column_{i}" for i, h in enumerate(headers)]
    seen: dict[str, int] = {}
    unique_cols = []
    for col in cols:
        n = seen.get(col, 0)
        seen[col] = n + 1
        unique_cols.append(col if n == 0 else f"{col}_{n + 1}")
    df = raw.iloc[row_idx + 1:].copy()
    df.columns = unique_cols
    df = df.dropna(how="all")
    mapping = _infer_logbook_mapping(list(df.columns), df=df)
    return df, mapping


def best_excel_table(raw, candidates: list[int]):
    best = None
    best_score = -1
    for row_idx in candidates:
        df, mapping = excel_table_from_header(raw, row_idx)
        score = int(bool(mapping.get("file"))) * 3 + int(bool(mapping.get("hv"))) * 3
        score += int(bool(mapping.get("temperature"))) + int(bool(mapping.get("polarization")))
        score += int(bool(mapping.get("direction"))) + int(bool(mapping.get("azi")))
        score += int(bool(mapping.get("polar"))) + int(bool(mapping.get("tilt")))
        score += min(len(df), 20) / 1000
        if score > best_score:
            best = (df, mapping, row_idx)
            best_score = score
    if best is None or best_score < 6:
        return None
    return best[0], best[1]


def _records_from_df(pd, df, mapping: dict[str, str]) -> list[dict]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return inherit_logbook_context(records, mapping)


def read_logbook(
    path: str | Path,
    *,
    sheet_selector: Callable[[list[str]], str] | None = None,
    table_selector: Callable[[object, list[int]], tuple[object, dict[str, str]] | None] | None = None,
    mapping_selector: Callable[[list[str], dict[str, str]], dict[str, str]] | None = None,
) -> LogbookReadResult:
    """Lit un logbook Excel/CSV/TSV et retourne records + mapping.

    Les callbacks optionnels permettent à l'UI de demander une feuille, une
    ligne d'en-tête ou un mapping manuel quand l'heuristique ne suffit pas.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas est nécessaire pour lire les logbooks Excel/CSV.") from exc

    path = Path(path)
    suffix = path.suffix.lower()
    sheet_name = ""

    if suffix in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        if sheet_selector is None:
            sheet_name = book.sheet_names[0] if len(book.sheet_names) == 1 else ""
        else:
            sheet_name = sheet_selector(book.sheet_names)
        if not sheet_name:
            raise ValueError("Aucune feuille Excel sélectionnée.")
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if raw.dropna(how="all").empty:
            raise ValueError("Le logbook ne contient aucune ligne exploitable.")
        candidates = excel_header_candidates(raw)
        guessed = best_excel_table(raw, candidates)
        if guessed is None and table_selector is not None:
            guessed = table_selector(raw, candidates)
        if guessed is None:
            raise ValueError("Aucune ligne d'en-tête sélectionnée pour le logbook.")
        df, mapping = guessed
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("Le logbook ne contient aucune ligne exploitable.")
        df.columns = [str(c).strip() for c in df.columns]
        mapping = _infer_logbook_mapping(list(df.columns), df=df)
    else:
        if suffix == ".csv":
            try:
                df = pd.read_csv(path, sep=";")
            except Exception:
                df = pd.read_csv(path)
        else:
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path)
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("Le logbook ne contient aucune ligne exploitable.")
        df.columns = [str(c).strip() for c in df.columns]
        mapping = _infer_logbook_mapping(list(df.columns), df=df)
        if len(df.columns) <= 1 or not mapping.get("file") or not mapping.get("hv"):
            raw = read_delimited_logbook_raw(pd, path)
            if raw is not None and not raw.dropna(how="all").empty:
                candidates = excel_header_candidates(raw)
                guessed = best_excel_table(raw, candidates)
                if guessed is not None:
                    df, mapping = guessed

    if not mapping.get("file") or not mapping.get("hv"):
        if mapping_selector is not None:
            mapping = mapping_selector(list(df.columns), mapping)
    if not mapping.get("file") or not mapping.get("hv"):
        raise ValueError("Les colonnes fichier et hν sont obligatoires pour appliquer un logbook.")

    return LogbookReadResult(
        records=_records_from_df(pd, df, mapping),
        mapping=mapping,
        sheet_name=sheet_name,
    )
