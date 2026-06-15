"""Content-based column sniffing for logbook mapping.

Split out of ``logbook.py`` to keep that file under the 700-LOC cap. Pure
helper: depends only on numpy and a dataframe-like object, no logbook
internals, so there is no import cycle with ``logbook.py``.
"""
from __future__ import annotations

import numpy as np


def _sniff_columns_by_content(df, columns: list[str]) -> dict[str, str]:
    """Devine les colonnes à partir des valeurs quand keyword match échoue.

    Règles :
      - hv : numérique, médiane dans [4, 10000] eV (XPS/HAXPES), ≥60% finies > 0
      - temperature : numérique, médiane dans [0.001, 1000] K (mK→HT), ≥60% finies > 0
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
    DIR_HINTS = ("Γ", "GAMMA", "M", "X", "Y", "K", "S", "Σ", "SIGMA")
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
        # hv : numérique [4, 10000] (XPS/HAXPES). P4.4
        # NB : ce sniff numérique est un repli après le matching par nom de
        # colonne (_pick_exact_column) ; sur les plages élargies hv/T se
        # recouvrent, donc le nom reste l'indice primaire.
        if "hv" not in out:
            med, ratio = numeric_stats(values)
            if med is not None and 4.0 <= med <= 10000.0 and ratio >= 0.6:
                out["hv"] = col
                continue
        # temperature : numérique [0.001, 1000] K (mK→HT), mais pas hv. P4.4
        if "temperature" not in out and out.get("hv") != col:
            med, ratio = numeric_stats(values)
            if med is not None and 0.001 <= med <= 1000.0 and ratio >= 0.6:
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
