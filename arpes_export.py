"""Exports de resultats ARPES sans dependance UI."""
from __future__ import annotations

from typing import Any
import csv
import numpy as np


BASE_RESULT_COLUMNS = [
    "file",
    "hv",
    "T_K",
    "direction",
    "E_eV",
    "dE_meV",
    "dk_inv_a",
    "resolution_source",
]


def _value_at(series: Any, index: int):
    try:
        return series[index] if index < len(series) else ""
    except TypeError:
        return ""


def result_rows(session) -> list[dict]:
    """Produit les lignes d'export CSV depuis une session."""
    rows: list[dict] = []
    for name, entry in session.files.items():
        if entry.fit_result is None:
            continue
        fr = entry.fit_result
        meta = entry.meta
        ev_f = np.asarray(fr.get("e_fitted", []))
        n = entry.fit_params.n_pairs
        res = fr.get("resolution", {}) or {}
        for ie, ev in enumerate(ev_f):
            row = {
                "file": name,
                "hv": meta.hv,
                "T_K": meta.temperature,
                "direction": meta.direction,
                "E_eV": ev,
                "dE_meV": res.get("dE_meV", ""),
                "dk_inv_a": res.get("dk_inv_a", ""),
                "resolution_source": res.get("source", ""),
            }
            for i in range(n):
                km_arr = fr.get("kF_minus", [])[i] if i < len(fr.get("kF_minus", [])) else []
                kp_arr = fr.get("kF_plus", [])[i] if i < len(fr.get("kF_plus", [])) else []
                row[f"kF_minus_{i+1}"] = _value_at(km_arr, ie)
                row[f"kF_plus_{i+1}"] = _value_at(kp_arr, ie)
                for key in ("gamma_brut", "gamma_min", "gamma_corrige"):
                    arr = fr.get(key, [])
                    vals = arr[i] if i < len(arr) else []
                    row[f"{key}_{i+1}"] = _value_at(vals, ie)
            rows.append(row)
    return rows


def result_columns(rows: list[dict]) -> list[str]:
    """Retourne un ordre de colonnes stable en preservant les colonnes dynamiques."""
    if not rows:
        return []
    dynamic: list[str] = []
    for row in rows:
        for key in row:
            if key not in BASE_RESULT_COLUMNS and key not in dynamic:
                dynamic.append(key)
    return BASE_RESULT_COLUMNS + dynamic


def write_results_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_columns(rows))
        writer.writeheader()
        writer.writerows(rows)
