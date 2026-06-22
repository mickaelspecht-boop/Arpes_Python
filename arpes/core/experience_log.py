"""Readable processing log for one ARPES signal.

The log is derived from the persisted session state. It does not store a second
history stream, so old sessions remain readable and there is no risk of the log
drifting away from the actual parameters used by the app.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


def build_experience_log(entry: Any, *, name: str = "") -> str:
    """Return a Markdown report for a BM or FS session entry."""
    kind = _scan_kind(entry)
    lines: list[str] = [
        f"# Processing log - {name or 'current signal'}",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"- Signal kind: {kind.upper()}",
    ]
    lines.extend(_metadata_lines(entry))
    lines.append("")
    lines.extend(_load_section(entry))
    lines.extend(_axis_energy_section(entry))
    if kind == "fs":
        lines.extend(_fs_section(entry))
    else:
        lines.extend(_bm_section(entry))
    lines.extend(_analysis_section(entry))
    return "\n".join(lines).rstrip() + "\n"


def _scan_kind(entry: Any) -> str:
    meta = getattr(entry, "meta", None)
    kind = str(getattr(meta, "scan_kind", "") or "").lower()
    if kind in {"bm", "fs", "kz", "edc"}:
        return kind
    if getattr(entry, "fs_center_kx", None) is not None or getattr(entry, "fs_pockets", None):
        return "fs"
    return "bm"


def _metadata_lines(entry: Any) -> list[str]:
    meta = getattr(entry, "meta", None)
    if meta is None:
        return []
    vals = [
        ("hv", getattr(meta, "hv", 0.0), "eV"),
        ("temperature", getattr(meta, "temperature", 0.0), "K"),
        ("direction", getattr(meta, "direction", ""), ""),
        ("polarization", getattr(meta, "polarization", ""), ""),
        ("loader", getattr(meta, "loader_label", ""), ""),
        ("format", getattr(meta, "source_format", ""), ""),
        ("formula", getattr(meta, "formula", ""), ""),
        ("MP id", getattr(meta, "mp_id", ""), ""),
    ]
    out = []
    for label, value, unit in vals:
        if value in ("", None, 0, 0.0):
            continue
        suffix = f" {unit}" if unit else ""
        out.append(f"- {label}: {value}{suffix}")
    return out


def _load_section(entry: Any) -> list[str]:
    meta = getattr(entry, "meta", None)
    loader = getattr(meta, "loader_label", "") or getattr(meta, "source_format", "") or "session loader"
    return [
        "",
        "## 1. Input",
        f"- Raw signal loaded through `{loader}`.",
        "- Stored axes convention: energy is E - EF in eV when calibrated; momentum is in pi/a for BM views.",
    ]


def _axis_energy_section(entry: Any) -> list[str]:
    out = [
        "",
        "## 2. Axes and energy reference",
    ]
    ef_offset = float(getattr(entry, "ef_offset", 0.0) or 0.0)
    out.append(f"- EF offset applied in display/session: {ef_offset:+.6f} eV.")
    ef_corr = dict(getattr(entry, "ef_correction", {}) or {})
    if ef_corr:
        mode = ef_corr.get("mode", "unknown")
        if mode == "poly":
            out.append("- EF calibration: polynomial correction stored for this file.")
            out.append("  Formula: `E_corrected = E_raw - delta_E(k)`.")
        else:
            out.append(f"- EF calibration: `{mode}` correction stored.")
        out.extend(_dict_lines(ef_corr, indent="  "))
    else:
        out.append("- EF calibration: no stored correction.")
    gamma = dict(getattr(entry, "meta_gamma_state", {}) or {})
    if gamma:
        out.append("- Gamma-axis recentering metadata is stored.")
        for key in sorted(gamma):
            if "shift" in key or "centered" in key or "reference" in key:
                out.append(f"  - {key}: {gamma[key]}")
    return out


def _bm_section(entry: Any) -> list[str]:
    out = [
        "",
        "## 3. BM display transformations",
    ]
    view = str(getattr(entry, "view_mode", "Raw") or "Raw")
    edcnorm = bool(getattr(entry, "edcnorm", False) or view == "EDCnorm")
    if edcnorm:
        out.append("- EDC normalization active.")
        out.append("  Formula: `I_norm(k, E) = I(k, E) / mean_k I(k, E)` — the map is "
                   "divided by the k-averaged EDC at each energy, removing the "
                   "energy-dependent intensity envelope to enhance momentum contrast.")
    else:
        out.append(f"- View mode: {view}.")
    grid = dict(getattr(entry, "grid_correction", {}) or {})
    if grid.get("enabled"):
        out.append("- Detector-grid correction active.")
        out.append("  Formula: automatic 2D Fourier mask suppresses detector-grid peaks, blended by `strength`.")
        out.extend(_dict_lines(grid, indent="  "))
    else:
        out.append("- Detector-grid correction: disabled.")
    dist = dict(getattr(entry, "bm_distortion", {}) or {})
    if _distortion_active(dist):
        out.append("- BM distortion correction active.")
        out.append("  Formula: k-axis is remapped by trapezoid/parabola calibration before plotting/fitting.")
        out.extend(_dict_lines(dist, indent="  ", max_items=14))
    else:
        out.append("- BM distortion correction: disabled.")
    fit = getattr(entry, "fit_result", None)
    if fit:
        out.append("- MDC fit result stored for this BM.")
        out.extend(_fit_summary(fit, getattr(entry, "fit_params", None), indent="  "))
    else:
        out.append("- MDC fit: no stored fit result.")
    zones = list(getattr(entry, "fit_zones", []) or [])
    if zones:
        out.append(f"- Fit zones: {len(zones)} zone(s); active zone `{getattr(entry, 'active_zone_id', None)}`.")
        for z in zones:
            label = z.get("label") or z.get("id")
            zfr = z.get("fit_result") or {}
            out.append(f"  - {label}: active={bool(z.get('active', True))}, fitted={bool(zfr)}")
    return out


def _fs_section(entry: Any) -> list[str]:
    out = [
        "",
        "## 3. FS transformations",
    ]
    cx = getattr(entry, "fs_center_kx", None)
    cy = getattr(entry, "fs_center_ky", None)
    if cx is not None or cy is not None:
        out.append(f"- FS center: kx={_fmt(cx)}, ky={_fmt(cy)}.")
        out.append("  Formula: centered axes use `k_centered = k_raw - k_gamma`.")
    else:
        out.append("- FS center: not stored.")
    rot = float(getattr(entry, "fs_rotation_deg", 0.0) or 0.0)
    if abs(rot) > 1e-12:
        out.append(f"- FS display rotation: {rot:+.3f} deg.")
        out.append("  Formula: 2D rotation matrix applied to display coordinates.")
    out.append(
        f"- kz settings: plane={getattr(entry, 'fs_kz_plane', 'Auto')}, "
        f"V0={float(getattr(entry, 'fs_v0', 0.0) or 0.0):.3f} eV, "
        f"phi_c={float(getattr(entry, 'fs_phi_c_deg', 0.0) or 0.0):+.3f} deg."
    )
    if bool(getattr(entry, "propagate_distortion_to_fs", False)):
        out.append("- BM distortion propagation to FS volume: enabled.")
    else:
        out.append("- BM distortion propagation to FS volume: disabled.")
    lattice = dict(getattr(entry, "fs_lattice", {}) or {})
    if lattice:
        out.append("- Crystal/BZ lattice cache:")
        out.extend(_dict_lines(lattice, indent="  ", max_items=10))
    pockets = list(getattr(entry, "fs_pockets", []) or [])
    if pockets:
        out.append(f"- FS pockets extracted: {len(pockets)} pocket(s).")
        for i, p in enumerate(pockets[:8], start=1):
            out.append(
                f"  - P{i}: kF_mean={_fmt(p.get('kF_mean'))}, "
                f"area={_fmt(p.get('area_pct_bz'))}% BZ, "
                f"topology={p.get('topology', '?')}"
            )
    else:
        out.append("- FS pockets: none stored.")
    return out


def _analysis_section(entry: Any) -> list[str]:
    out = [
        "",
        "## 4. Downstream analyses",
    ]
    theory = dict(getattr(entry, "theory_overlay", {}) or {})
    out.append(f"- Theory overlay: {'stored' if theory else 'none'}.")
    band = dict(getattr(entry, "band_analysis", {}) or {})
    if band:
        out.append("- Band analysis stored:")
        out.extend(_dict_lines(band, indent="  ", max_items=12))
    else:
        out.append("- Band analysis: none stored.")
    annotations = getattr(entry, "annotations", {}) or {}
    n_ann = sum(len(v or []) for v in annotations.values())
    out.append(f"- Fit-point annotations: {n_ann}.")
    return out


def _fit_summary(fr: dict, fp: Any, *, indent: str) -> list[str]:
    out: list[str] = []
    e = np.asarray(fr.get("e_fitted", []), dtype=float)
    n_e = int(e.size)
    n_pairs = int(fr.get("n_pairs") or getattr(fp, "n_pairs", 0) or len(fr.get("kF_plus") or []))
    out.append(f"{indent}- slices: {n_e}; pairs: {n_pairs}.")
    out.append(f"{indent}- model: MDC Lorentzian/Voigt peak-pair fit on each energy slice.")
    out.append(f"{indent}- formula: `I(k) = background + sum_i Lorentzian(k; xg +/- kF_i, gamma_i)`.")
    if fr.get("ensemble"):
        ens = fr.get("ensemble") or {}
        out.append(
            f"{indent}- ensemble: {int(ens.get('n_ok', 0) or 0)}/"
            f"{int(ens.get('n_runs', 0) or 0)} runs, jitter={float(ens.get('jitter_pct', 0.0) or 0.0):.3f}."
        )
        out.append(f"{indent}- uncertainty: run-to-run spread combined with per-run covariance sigma.")
    elif fr.get("sigma_kF_plus") or fr.get("sigma_gamma"):
        out.append(f"{indent}- uncertainty: per-slice covariance sigma stored.")
    else:
        out.append(f"{indent}- uncertainty: no sigma arrays stored.")
    for key in ("distorted", "grid_active", "params_hash", "asymmetric_warning"):
        if key in fr:
            out.append(f"{indent}- {key}: {fr.get(key)}")
    return out


def _dict_lines(d: dict, *, indent: str, max_items: int = 20) -> list[str]:
    out: list[str] = []
    for i, key in enumerate(sorted(d)):
        if i >= max_items:
            out.append(f"{indent}- ... {len(d) - max_items} more field(s)")
            break
        value = d[key]
        if isinstance(value, dict):
            value = _compact_dict(value)
        out.append(f"{indent}- {key}: {value}")
    return out


def _compact_dict(value: dict) -> str:
    parts = []
    for i, key in enumerate(sorted(value)):
        if i >= 6:
            parts.append("...")
            break
        parts.append(f"{key}={value[key]}")
    return "{" + ", ".join(parts) + "}"


def _distortion_active(cfg: dict) -> bool:
    if not cfg:
        return False
    trap = cfg.get("trapezoid") or {}
    para = cfg.get("parabola") or {}
    return bool(
        (trap.get("enabled") and (
            abs(float(trap.get("slope_left", 0.0) or 0.0)) > 0
            or abs(float(trap.get("slope_right", 0.0) or 0.0)) > 0
        ))
        or (para.get("enabled") and abs(float(para.get("a", 0.0) or 0.0)) > 0)
    )


def _fmt(value) -> str:
    try:
        if value is None:
            return "not set"
        f = float(value)
        return f"{f:.5g}" if np.isfinite(f) else "not finite"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ts(ts: str) -> str:
    """Compact display of an ISO timestamp: keep date + time, drop the 'T'/'Z'."""
    if not ts:
        return "?"
    return str(ts).replace("T", " ").replace("Z", "")


def _params_str(params: dict) -> str:
    if not params:
        return ""
    return ", ".join(f"{key}={params[key]}" for key in params)


def build_timeline(entry: Any, *, name: str = "", limit: int = 0) -> str:
    """Return a Markdown chronological journal from entry.processing_history.

    This is the time-ordered audit trail ("what we did, and when"), the
    complement of build_experience_log (which is the current-state snapshot).
    `limit` > 0 keeps only the most recent events.
    """
    hist = list(getattr(entry, "processing_history", []) or [])
    out = [f"# Processing timeline - {name or 'current signal'}", ""]
    if not hist:
        out.append("_No recorded operations yet._")
        return "\n".join(out) + "\n"
    shown = hist
    if limit and len(hist) > limit:
        shown = hist[-limit:]
        out.append(f"- Showing last {limit} of {len(hist)} events.")
    else:
        out.append(f"- Total events: {len(hist)}.")
    out.append("")
    out.append("| Time (UTC) | Step | Action | Details |")
    out.append("|---|---|---|---|")
    for ev in shown:
        ts = _fmt_ts(ev.get("ts", ""))
        cat = str(ev.get("category", "")).upper()
        action = str(ev.get("action", "")).replace("|", "/")
        detail = str(ev.get("summary") or _params_str(ev.get("params") or {})).replace("|", "/")
        out.append(f"| {ts} | {cat} | {action} | {detail} |")
    return "\n".join(out) + "\n"


def build_full_report(entry: Any, *, name: str = "") -> str:
    """Chronological timeline + current-state snapshot (for Markdown export)."""
    return (
        build_timeline(entry, name=name)
        + "\n---\n\n"
        + build_experience_log(entry, name=name)
    )


def entry_to_plain_dict(entry: Any) -> dict:
    """Testing/debug helper: convert a FileEntry-like object to a plain dict."""
    if is_dataclass(entry):
        return asdict(entry)
    return dict(getattr(entry, "__dict__", {}) or {})
