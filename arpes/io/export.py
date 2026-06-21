"""ARPES result exports without a UI dependency."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arpes.core.sample import require_lattice_a, sample_for_entry
from arpes.core.session import Session


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
    """Build CSV export rows from a session.

    Include per-slice statistical uncertainties (sigma_kF_*, sigma_gamma) when
    they are available in the fit_result.
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
    "pair", "branch", "n_points", "linear_ok", "refused_reason",
    "kF_pi_a", "kF_pi_a_sigma",
    "kF_inv_A", "kF_inv_A_sigma",
    "vF_eV_pi_a", "vF_eV_pi_a_sigma",
    "m_star_over_me", "m_star_over_me_sigma",
    "luttinger_density", "luttinger_density_sigma", "luttinger_units",
    "gamma_zero", "gamma_zero_sigma",
    "coef_E2", "coef_E2_sigma",
    "asym_delta_kF", "asym_delta_kF_sigma", "asym_is_symmetric",
]


def git_commit_hash(repo_root: str | Path | None = None) -> str:
    """Return the current git commit hash, or a clear fallback outside git."""
    cwd = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return "unversioned"
    return proc.stdout.strip() or "unversioned"


def export_timestamp_utc() -> str:
    """UTC timestamp in ISO-8601 format for reproducible exports."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_session_file(session, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute() and getattr(session, "folder", None):
        path = Path(session.folder) / path
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def input_fingerprint(session, file_names: list[str] | set[str] | None = None) -> tuple[str, list[dict]]:
    """Hash the input file identity/mtime/size used by the session."""
    items: list[dict] = []
    names = sorted(file_names if file_names is not None else getattr(session, "files", {}))
    for name in names:
        path = _resolve_session_file(session, name)
        record: dict[str, Any] = {
            "file": name,
            "path": str(path),
            "exists": path.exists(),
        }
        try:
            stat = path.stat()
        except OSError:
            record.update({"mtime_ns": None, "size": None})
        else:
            record.update({"mtime_ns": stat.st_mtime_ns, "size": stat.st_size})
        items.append(record)
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), items


def export_provenance(
    session,
    *,
    content: str = "results",
    file_names: list[str] | set[str] | None = None,
) -> dict:
    """Build machine-readable provenance attached to CSV and figure exports."""
    files = getattr(session, "files", {})
    names = sorted(file_names if file_names is not None else files)
    input_hash, inputs = input_fingerprint(session, names)
    samples = {
        name: sample_for_entry(session, entry, name).to_dict()
        for name, entry in sorted(files.items())
        if name in names
    }
    file_state = {
        name: {
            "hv": getattr(entry.meta, "hv", 0.0),
            "temperature": getattr(entry.meta, "temperature", 0.0),
            "polarization": getattr(entry.meta, "polarization", ""),
            "direction": getattr(entry.meta, "direction", ""),
            "instrument": getattr(entry.meta, "loader_label", ""),
            "source_format": getattr(entry.meta, "source_format", ""),
            "fit_params": _json_safe(getattr(entry.fit_params, "__dict__", {})),
            "ef_correction": _json_safe(getattr(entry, "ef_correction", {})),
            "bm_distortion": _json_safe(getattr(entry, "bm_distortion", {})),
            "pocket_settings": _json_safe(getattr(entry, "fs_pockets", [])),
            "gamma_state": _json_safe(getattr(entry, "meta_gamma_state", {})),
            "angle_offsets": _json_safe(getattr(session, "angle_offsets", {})),
        }
        for name, entry in sorted(files.items())
        if name in names
    }
    return {
        "app": "ARPES Explorer",
        "session_version": getattr(session, "VERSION", Session.VERSION),
        "git_commit": git_commit_hash(),
        "timestamp_utc": export_timestamp_utc(),
        "content": content,
        "input_hash": input_hash,
        "inputs": inputs,
        "sample_config": samples,
        "files": file_state,
    }


def physics_rows(session, *, e_window_kF: float = 0.10, e_window_gamma: float = 0.30) -> list[dict]:
    """Build one row per (file, pair, branch) with physical results ± σ.

    Use ``arpes.analysis.results.compute_results`` with ``crystal_a`` read from
    ``SampleConfig``. Refuse the physics export if the lattice is missing.
    """
    from arpes.analysis.results import compute_results
    import math
    rows: list[dict] = []
    for name, entry in session.files.items():
        if entry.fit_result is None:
            continue
        meta = entry.meta
        sample = sample_for_entry(session, entry, name)
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
                "linear_ok": int(getattr(br, "linear_ok", True)),
                "refused_reason": getattr(br, "refused_reason", ""),
                "kF_pi_a": br.kF_at_EF, "kF_pi_a_sigma": br.kF_at_EF_sigma,
                "kF_inv_A": kF_inv_A, "kF_inv_A_sigma": kF_inv_A_sigma,
                "vF_eV_pi_a": br.vF_eV_pi_a, "vF_eV_pi_a_sigma": br.vF_sigma,
                "m_star_over_me": br.m_star_over_me, "m_star_over_me_sigma": br.m_star_sigma,
                "luttinger_density": br.luttinger_density_pi_a2,
                "luttinger_density_sigma": br.luttinger_density_sigma,
                "luttinger_units": getattr(br, "luttinger_units", ""),
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
    """Generate a booktabs LaTeX table of physical results by branch.

    Compact publication format: `kF (Å⁻¹), v_F (eV·π/a), m*/m_e, Γ₀ (π/a)`
    with uncertainties ± σ. One row per (file, pair, branch).
    """
    if not rows:
        return "% No physical results available.\n"

    def fmt(value, sigma, dec=4):
        try:
            v = float(value); s = float(sigma)
        except (TypeError, ValueError):
            return "--"
        if not (v == v and s == s):  # NaN guard
            return "--"
        return f"${v:.{dec}f} \\pm {s:.{dec}f}$"

    lines = [
        "% Table generated by ARPES Explorer - physics_to_latex",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{ARPES physical results by branch.}",
        "\\label{tab:arpes_physics}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "File & Pair/Branch & $T$ (K) & Dir. & $k_F$ (\\AA$^{-1}$) & "
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


def _write_csv_provenance(f, provenance: dict | None) -> None:
    if not provenance:
        return
    compact = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    f.write(
        "# ARPES Explorer "
        f"v{provenance.get('session_version', '?')} "
        f"commit {provenance.get('git_commit', 'unversioned')} "
        f"exported {provenance.get('timestamp_utc', '')}\n"
    )
    f.write(f"# provenance_json: {compact}\n")


def write_provenance_sidecar(path: str, provenance: dict | None) -> None:
    """Write a .meta.json next to an export for tools that cannot read CSV comments."""
    if not provenance:
        return
    meta_path = Path(path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")


def write_physics_csv(path: str, rows: list[dict], *, provenance: dict | None = None) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        _write_csv_provenance(f, provenance)
        writer = csv.DictWriter(f, fieldnames=PHYSICS_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def result_columns(rows: list[dict]) -> list[str]:
    """Return a stable column order while preserving dynamic columns."""
    if not rows:
        return []
    dynamic: list[str] = []
    for row in rows:
        for key in row:
            if key not in BASE_RESULT_COLUMNS and key not in dynamic:
                dynamic.append(key)
    return BASE_RESULT_COLUMNS + dynamic


def write_results_csv(path: str, rows: list[dict], *, provenance: dict | None = None) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        _write_csv_provenance(f, provenance)
        writer = csv.DictWriter(f, fieldnames=result_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def _rows_to_aligned_txt(rows: list[dict], columns: list[str]) -> str:
    """Aligned-column text table (fixed width) for human reading."""
    if not rows:
        return "# No data.\n"
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
