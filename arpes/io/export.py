"""Exports de resultats ARPES sans dependance UI."""
from __future__ import annotations

from typing import Any
import csv
import numpy as np

from arpes.core.sample import require_lattice_a, sample_for_entry


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
    """Produit les lignes d'export CSV depuis une session.

    Inclut les incertitudes statistiques par slice (sigma_kF_*, sigma_gamma)
    si disponibles dans le fit_result.
    """
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
                for sigma_key in ("sigma_kF_minus", "sigma_kF_plus", "sigma_gamma"):
                    arr = fr.get(sigma_key, [])
                    vals = arr[i] if i < len(arr) else []
                    row[f"{sigma_key}_{i+1}"] = _value_at(vals, ie)
                for key in ("gamma_brut", "gamma_min", "gamma_corrige"):
                    arr = fr.get(key, [])
                    vals = arr[i] if i < len(arr) else []
                    row[f"{key}_{i+1}"] = _value_at(vals, ie)
            rows.append(row)
    return rows


PHYSICS_RESULT_COLUMNS = [
    "file", "hv", "T_K", "direction", "formula", "mp_id", "crystal_a_angstrom",
    "pair", "branch", "n_points",
    "kF_pi_a", "kF_pi_a_sigma",
    "kF_inv_A", "kF_inv_A_sigma",
    "vF_eV_pi_a", "vF_eV_pi_a_sigma",
    "m_star_over_me", "m_star_over_me_sigma",
    "luttinger_density", "luttinger_density_sigma",
    "gamma_zero", "gamma_zero_sigma",
    "coef_E2", "coef_E2_sigma",
    "asym_delta_kF", "asym_delta_kF_sigma", "asym_is_symmetric",
]


def physics_rows(session, *, e_window_kF: float = 0.10, e_window_gamma: float = 0.30) -> list[dict]:
    """Produit une ligne par (fichier, paire, branche) avec résultats physiques ± σ.

    Utilise ``arpes.analysis.results.compute_results`` avec ``crystal_a`` lu
    depuis ``SampleConfig``. Refuse l'export physique si la maille manque.
    """
    from arpes.analysis.results import compute_results
    import math
    rows: list[dict] = []
    for name, entry in session.files.items():
        if entry.fit_result is None:
            continue
        meta = entry.meta
        sample = sample_for_entry(session, entry)
        a_val = require_lattice_a(sample, context=name)
        bundle = compute_results(
            entry.fit_result,
            e_window_kF=e_window_kF, e_window_gamma=e_window_gamma,
            crystal_a_angstrom=a_val,
        )
        gamma_by_pair = {g.pair_index: g for g in bundle.gamma_fl}
        asym_by_pair = {a.pair_index: a for a in bundle.asymmetry}
        for br in bundle.branches:
            kF_inv_A = br.kF_at_EF * math.pi / a_val if a_val > 0 else float("nan")
            kF_inv_A_sigma = br.kF_at_EF_sigma * math.pi / a_val if a_val > 0 else float("nan")
            g = gamma_by_pair.get(br.pair_index)
            asym = asym_by_pair.get(br.pair_index)
            row = {
                "file": name, "hv": meta.hv, "T_K": meta.temperature,
                "direction": meta.direction, "formula": sample.formula,
                "mp_id": sample.mp_id, "crystal_a_angstrom": a_val,
                "pair": br.pair_index + 1,
                "branch": br.branch.replace("kF_", ""),
                "n_points": br.n_points_used,
                "kF_pi_a": br.kF_at_EF, "kF_pi_a_sigma": br.kF_at_EF_sigma,
                "kF_inv_A": kF_inv_A, "kF_inv_A_sigma": kF_inv_A_sigma,
                "vF_eV_pi_a": br.vF_eV_pi_a, "vF_eV_pi_a_sigma": br.vF_sigma,
                "m_star_over_me": br.m_star_over_me, "m_star_over_me_sigma": br.m_star_sigma,
                "luttinger_density": br.luttinger_density_pi_a2,
                "luttinger_density_sigma": br.luttinger_density_sigma,
                "gamma_zero": g.gamma_zero if g else float("nan"),
                "gamma_zero_sigma": g.gamma_zero_sigma if g else float("nan"),
                "coef_E2": g.coef_E2 if g else float("nan"),
                "coef_E2_sigma": g.coef_E2_sigma if g else float("nan"),
                "asym_delta_kF": asym.delta_kF if asym else float("nan"),
                "asym_delta_kF_sigma": asym.delta_kF_sigma if asym else float("nan"),
                "asym_is_symmetric": int(asym.is_symmetric) if asym else "",
            }
            rows.append(row)
    return rows


def physics_to_latex(rows: list[dict]) -> str:
    """Génère une table LaTeX booktabs des résultats physiques par branche.

    Format compact pour publication : `kF (Å⁻¹), v_F (eV·π/a), m*/m_e, Γ₀ (π/a)`
    avec incertitudes ± σ. Ligne par (fichier, paire, branche).
    """
    if not rows:
        return "% Aucun résultat physique disponible.\n"

    def fmt(value, sigma, dec=4):
        try:
            v = float(value); s = float(sigma)
        except (TypeError, ValueError):
            return "--"
        if not (v == v and s == s):  # NaN guard
            return "--"
        return f"${v:.{dec}f} \\pm {s:.{dec}f}$"

    lines = [
        "% Table générée par ARPES Explorer — physics_to_latex",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Résultats physiques ARPES par branche.}",
        "\\label{tab:arpes_physics}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Fichier & Paire/Branche & $T$ (K) & Dir. & $k_F$ (\\AA$^{-1}$) & "
        "$v_F$ (eV·$\\pi/a$) & $m^*/m_e$ & $\\Gamma_0$ ($\\pi/a$) \\\\",
        "\\midrule",
    ]
    for r in rows:
        f_safe = str(r.get("file", "")).replace("_", "\\_")
        d_safe = str(r.get("direction", "")).replace("_", "\\_")
        label = f"P{r.get('pair', '?')} {r.get('branch', '')}"
        try:
            t = f"{float(r.get('T_K', 0.0)):.0f}"
        except (TypeError, ValueError):
            t = "--"
        lines.append(
            " & ".join([
                f_safe, label, t, d_safe,
                fmt(r.get("kF_inv_A"), r.get("kF_inv_A_sigma"), dec=4),
                fmt(r.get("vF_eV_pi_a"), r.get("vF_eV_pi_a_sigma"), dec=2),
                fmt(r.get("m_star_over_me"), r.get("m_star_over_me_sigma"), dec=2),
                fmt(r.get("gamma_zero"), r.get("gamma_zero_sigma"), dec=4),
            ]) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_physics_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PHYSICS_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


def _rows_to_aligned_txt(rows: list[dict], columns: list[str]) -> str:
    """Table texte à colonnes alignées (fixed width). Pour lecture humaine."""
    if not rows:
        return "# Aucune donnée.\n"
    str_rows = [[("" if r.get(c) is None else str(r.get(c))) for c in columns] for r in rows]
    widths = [
        max(len(c), max((len(s[i]) for s in str_rows), default=0))
        for i, c in enumerate(columns)
    ]
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    lines.append("  ".join("-" * w for w in widths))
    for s in str_rows:
        lines.append("  ".join(s[i].ljust(widths[i]) for i in range(len(columns))))
    return "\n".join(lines) + "\n"


def write_physics_txt(path: str, rows: list[dict]) -> None:
    text = _rows_to_aligned_txt(rows, PHYSICS_RESULT_COLUMNS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_results_txt(path: str, rows: list[dict]) -> None:
    cols = result_columns(rows)
    text = _rows_to_aligned_txt(rows, cols)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
