"""Read ARPES logbooks with pure parsing, without PyQt.

Interactive sheet/table/column selection stays in the UI. This module contains
only read heuristics, header detection, mapping, and context propagation between
rows.
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
    """Raw CSV/TSV read for older logbooks with a title before the header."""
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
    """Propagate direction/polarization/azi when following cells are empty."""
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


# Labels that mark a "Folder Name" cell (on the left) whose adjacent cell (to
# the right, on the same row) is the subfolder name.
# Normalized through `_norm_text` (lowercase + alphanumeric).
_FOLDER_NAME_LABELS = {
    "foldername", "folder", "dossier", "nomdossier", "subfolder",
    "nameoffolder", "foldernameroot", "datafolder",
}

# Scan height (rows at the top of the sheet) used to find the cell.
_FOLDER_NAME_SCAN_ROWS = 15


def find_folder_name_in_sheet(raw) -> str:
    """Find the "Folder Name" cell and return its adjacent value.

    Robust to varying templates:
    - Scan the first ``_FOLDER_NAME_SCAN_ROWS`` rows.
    - Searched label: "Folder Name" / "Folder" / "Dossier" / "Subfolder" / ...
      (case-insensitive, spaces/punctuation ignored through ``_norm_text``).
    - Value = non-empty cell to the right of the label on the same row (skips up
      to 3 empty columns).
    - Return ``""`` if nothing relevant is found.

    Args:
        raw: raw ``pandas.DataFrame`` (header=None) for the sheet.
    """
    from arpes.io.logbook import _norm_text
    try:
        n_rows = min(int(len(raw)), _FOLDER_NAME_SCAN_ROWS)
    except TypeError:
        return ""
    for row_idx in range(n_rows):
        try:
            row = raw.iloc[row_idx].tolist()
        except Exception:
            continue
        for col_idx, cell in enumerate(row):
            label = _norm_text(cell)
            if label and label in _FOLDER_NAME_LABELS:
                # Value = first non-empty cell to the right (up to 4 columns).
                for j in range(col_idx + 1, min(col_idx + 5, len(row))):
                    val = _cell_text(row[j])
                    if val:
                        return val
                break  # label found but value empty: another row may contain it
    return ""


def match_folder_to_subfolder(
    folder_name: str,
    candidate_subfolders: list[str],
) -> str:
    """Match a declared folder name (sheet) with a session subfolder.

    Strategies (priority order):
    1. Exact match (case-sensitive) on the subfolder name (basename).
    2. Case-insensitive match.
    3. Normalized match through ``_norm_text`` (alphanumeric only).
    4. Subfolder basename match (rel = ``parent/BNA_S2`` → basename ``BNA_S2``).
    5. Substring: declared name contained in rel or reciprocal (normalized).

    Return the matched ``rel`` or ``""``.
    """
    from arpes.io.logbook import _norm_text
    target_norm = _norm_text(folder_name)
    if not target_norm:
        return ""

    # 1. exact
    for rel in candidate_subfolders:
        if rel == folder_name:
            return rel
    # 2. case-insensitive
    target_low = folder_name.lower()
    for rel in candidate_subfolders:
        if rel.lower() == target_low:
            return rel
    # 3. normalized
    for rel in candidate_subfolders:
        if _norm_text(rel) == target_norm:
            return rel
    # 4. basename (last part) match
    for rel in candidate_subfolders:
        parts = rel.replace("\\", "/").split("/")
        base = parts[-1] if parts else rel
        if _norm_text(base) == target_norm:
            return rel
    # 5. substring (normalized) - cautious match: >=3 chars, avoids false positives.
    #    Ambiguity-safe: if the declared name substring-matches >=2 DISTINCT
    #    subfolders (e.g. truncated "YNS" hitting both "YNS_S1" and "YNS_S6"),
    #    refuse (return "") instead of first-winner — silently scoping another
    #    sample's params onto this folder would corrupt Γ/φ/a downstream.
    if len(target_norm) >= 3:
        hits = [
            rel for rel in candidate_subfolders
            if _norm_text(rel)
            and (target_norm in _norm_text(rel) or _norm_text(rel) in target_norm)
        ]
        if len(hits) == 1:
            return hits[0]
        # 0 matches, or >=2 (ambiguous) -> no scoping (fail-loud at caller).
    return ""


def scan_xlsx_for_scoped_logbooks(
    pd,
    path,
    candidate_subfolders: list[str],
) -> list[dict]:
    """Scan every sheet in an xlsx file -> scoped-logbook candidates.

    For each sheet:
    - Read the first rows.
    - Look for a "Folder Name" cell via ``find_folder_name_in_sheet``.
    - If found, match against ``candidate_subfolders`` via
      ``match_folder_to_subfolder``.
    - Check that the sheet contains the expected columns (file + hv) via
      ``best_excel_table`` (otherwise ignore it: it is not a real data sheet).

    Return a list of ``{"sheet", "folder_declared", "subfolder_rel",
    "mapping", "df", "n_rows"}`` for each usable sheet.
    """
    out: list[dict] = []
    try:
        book = pd.ExcelFile(path)
    except Exception:
        return out
    for sheet in book.sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
        except Exception:
            continue
        if raw.dropna(how="all").empty:
            continue
        folder_declared = find_folder_name_in_sheet(raw)
        if not folder_declared:
            continue
        matched_rel = match_folder_to_subfolder(folder_declared, candidate_subfolders)
        if not matched_rel:
            continue
        candidates = excel_header_candidates(raw)
        guessed = best_excel_table(raw, candidates)
        if guessed is None:
            continue
        df, mapping = guessed
        if not mapping.get("file") or not mapping.get("hv"):
            continue
        out.append({
            "sheet": sheet,
            "folder_declared": folder_declared,
            "subfolder_rel": matched_rel,
            "mapping": mapping,
            "df": df,
            "n_rows": int(len(df)),
        })
    return out


_TITLE_TOKENS = {"plan", "measurement", "for", "title", "page", "sheet", "data"}


def _looks_like_title(column: str) -> bool:
    """True if the column name looks like a title (multiple words, keywords)."""
    if not column:
        return False
    words = [w.lower() for w in str(column).split() if w]
    if len(words) >= 3:
        return True
    return bool(_TITLE_TOKENS.intersection(words))


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
        # Penalty if the detected file column looks like a section title
        # ("Measurement Plan for ..." etc) - often a header-row error.
        if _looks_like_title(mapping.get("file", "")):
            score -= 3
        if score > best_score:
            best = (df, mapping, row_idx)
            best_score = score
    if best is None or best_score < 6:
        return None
    # Extra guard: if the file/hv column looks like a title, reject it so the
    # table_selector can ask the user.
    if _looks_like_title(best[1].get("file", "")) or _looks_like_title(best[1].get("hv", "")):
        return None
    return best[0], best[1]


def _records_from_df(pd, df, mapping: dict[str, str]) -> list[dict]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return inherit_logbook_context(records, mapping)


def get_xlsx_sheet_names(path: str | Path) -> list[str]:
    """Sheet names of an Excel workbook, without reading any data.

    Raises ValueError on unreadable/corrupt files (never silent) and
    ImportError when pandas is missing.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required to read Excel logbooks.") from exc
    try:
        return [str(s) for s in pd.ExcelFile(Path(path)).sheet_names]
    except Exception as exc:
        raise ValueError(f"Cannot read Excel file {Path(path).name}: {exc}") from exc


def read_logbook(
    path: str | Path,
    *,
    sheet_selector: Callable[[list[str]], str] | None = None,
    table_selector: Callable[[object, list[int]], tuple[object, dict[str, str]] | None] | None = None,
    mapping_selector: Callable[[list[str], dict[str, str]], dict[str, str]] | None = None,
) -> LogbookReadResult:
    """Read an Excel/CSV/TSV logbook and return records + mapping.

    Optional callbacks let the UI request a sheet, a header row, or a manual
    mapping when heuristics are not enough.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required to read Excel/CSV logbooks.") from exc

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
            raise ValueError("No Excel sheet selected.")
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if raw.dropna(how="all").empty:
            raise ValueError("The logbook contains no usable rows.")
        candidates = excel_header_candidates(raw)
        guessed = best_excel_table(raw, candidates)
        if guessed is None and table_selector is not None:
            guessed = table_selector(raw, candidates)
        if guessed is None:
            raise ValueError("No header row selected for the logbook.")
        df, mapping = guessed
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("The logbook contains no usable rows.")
        df.columns = [str(c).strip() for c in df.columns]
        mapping = _infer_logbook_mapping(list(df.columns), df=df)
    else:
        # Separator auto-detection: try several and keep the one producing the
        # most columns (>=2). Order: TAB, ;, ,, |.
        df = None
        best_ncols = 0
        for sep in ("\t", ";", ",", "|"):
            try:
                candidate = pd.read_csv(path, sep=sep)
            except Exception:
                continue
            if candidate is None:
                continue
            ncols = len(candidate.columns)
            if ncols > best_ncols:
                best_ncols = ncols
                df = candidate
        if df is None or best_ncols < 2:
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path)
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("The logbook contains no usable rows.")
        df.columns = [str(c).strip() for c in df.columns]
        mapping = _infer_logbook_mapping(list(df.columns), df=df)
        if len(df.columns) <= 1 or not mapping.get("file") or not mapping.get("hv"):
            raw = read_delimited_logbook_raw(pd, path)
            if raw is not None and not raw.dropna(how="all").empty:
                candidates = excel_header_candidates(raw)
                guessed = best_excel_table(raw, candidates)
                if guessed is None and table_selector is not None:
                    guessed = table_selector(raw, candidates)
                if guessed is not None:
                    df, mapping = guessed

    if not mapping.get("file") or not mapping.get("hv"):
        if mapping_selector is not None:
            mapping = mapping_selector(list(df.columns), mapping)
    if not mapping.get("file") or not mapping.get("hv"):
        raise ValueError("The file and hν columns are required to apply a logbook.")

    return LogbookReadResult(
        records=_records_from_df(pd, df, mapping),
        mapping=mapping,
        sheet_name=sheet_name,
    )
