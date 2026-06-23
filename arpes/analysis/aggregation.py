"""Aggregation of physical results across multiple files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from arpes.analysis.results import compute_results
from arpes.core.sample import require_lattice_a, sample_for_entry
from arpes.core.session import FileEntry, Session
from arpes.theory.models import normalize_direction_label


@dataclass(frozen=True)
class MultiFilePoint:
    filename: str
    x_value: float
    x_label: str
    kF: float
    kF_sigma: float
    m_star: float
    m_star_sigma: float
    vF: float = float("nan")        # Fermi velocity (eV·π/a)
    vF_sigma: float = float("nan")
    gamma_zero: float = float("nan")
    gamma_zero_sigma: float = float("nan")
    direction: str = ""
    compound: str = ""              # sample/compound label (grouping & colour)


@dataclass(frozen=True)
class MultiFileSeries:
    points: tuple[MultiFilePoint, ...]
    skipped: int = 0
    warning: str = ""


def aggregate_session_entries(
    session: Session,
    filenames: Iterable[str] | None = None,
    *,
    x_axis: str = "T (K)",
    direction_filter: str = "",
    crystal_a_default: float = 0.0,
) -> MultiFileSeries:
    """Extract kF, m*, Γ0 for a selection of session entries."""
    selected = list(filenames) if filenames is not None else list(session.files)
    points: list[MultiFilePoint] = []
    skipped = 0
    warnings: list[str] = []
    a_values: set[float] = set()
    category_map: dict[str, int] = {}
    norm_filter = normalize_direction_label(direction_filter)
    for name in selected:
        entry = session.files.get(name)
        if entry is None or not entry.fit_result:
            skipped += 1
            continue
        if norm_filter:
            direction = normalize_direction_label(entry.meta.direction)
            if norm_filter not in direction:
                skipped += 1
                continue
        sample = sample_for_entry(session, entry, name)
        if crystal_a_default and not sample.has_lattice_a:
            sample = sample.merge_missing_from(type(sample)(a_angstrom=float(crystal_a_default)))
        try:
            a_val = require_lattice_a(sample, context=name)
        except ValueError as exc:
            skipped += 1
            warnings.append(str(exc))
            continue
        a_values.add(round(a_val, 6))
        point = _point_from_entry(
            name, entry, x_axis=x_axis, a_val=a_val,
            category_map=category_map,
        )
        if point is None:
            skipped += 1
            continue
        points.append(point)
    if len(a_values) > 1:
        warnings.append("Heterogeneous a parameter across files.")
    points.sort(key=lambda p: (p.x_value, p.filename))
    return MultiFileSeries(points=tuple(points), skipped=skipped, warning=" ".join(warnings))


def _point_from_entry(
    filename: str,
    entry: FileEntry,
    *,
    x_axis: str,
    a_val: float,
    category_map: dict[str, int],
) -> MultiFilePoint | None:
    bundle = compute_results(
        entry.fit_result, crystal_a_angstrom=a_val,
        gamma_max=getattr(getattr(entry, "fit_params", None), "gamma_max", None),
    )
    branch = next((br for br in bundle.branches if np.isfinite(br.kF_at_EF)), None)
    if branch is None:
        return None
    gamma = bundle.gamma_fl[branch.pair_index] if branch.pair_index < len(bundle.gamma_fl) else None
    compound = compound_label(filename)
    x_value, x_label = _x_value(entry, x_axis, category_map, compound)
    if not np.isfinite(x_value):
        return None
    return MultiFilePoint(
        filename=filename,
        x_value=float(x_value),
        x_label=x_label,
        kF=float(branch.kF_at_EF),
        kF_sigma=float(branch.kF_at_EF_sigma),
        m_star=float(branch.m_star_over_me),
        m_star_sigma=float(branch.m_star_sigma),
        vF=float(branch.vF_eV_pi_a),
        vF_sigma=float(branch.vF_sigma),
        gamma_zero=float(gamma.gamma_zero) if gamma is not None else float("nan"),
        gamma_zero_sigma=float(gamma.gamma_zero_sigma) if gamma is not None else float("nan"),
        direction=str(entry.meta.direction or ""),
        compound=compound,
    )


def compound_label(filename: str) -> str:
    """Short compound/sample label for grouping & colour.

    Uses the session-key prefix (the sample folder), e.g.
    ``"Ba122Cu_C13/BM2" -> "Cu_C13"``. Strips a leading ``Ba122`` so the legend
    reads ``Cu_C13``, ``Cr_C10``, ``C05_2`` … Falls back to the raw name when
    there is no ``/`` (synthetic test entries).
    """
    prefix = str(filename).split("/", 1)[0].strip()
    if not prefix:
        return str(filename)
    for tag in ("Ba122_", "Ba122"):
        if prefix.startswith(tag):
            return prefix[len(tag):] or prefix
    return prefix


def _x_value(entry: FileEntry, x_axis: str, category_map: dict[str, int],
             compound: str = "") -> tuple[float, str]:
    meta = entry.meta
    if x_axis == "hν":
        value = float(meta.hv or np.nan)
        return value, f"{value:g}"
    if x_axis == "polarisation":
        label = str(meta.polarization or "").strip() or "?"
        if label not in category_map:
            category_map[label] = len(category_map)
        return float(category_map[label]), label
    if x_axis == "dopant":
        label = compound or "?"
        if label not in category_map:
            category_map[label] = len(category_map)
        return float(category_map[label]), label
    value = float(meta.temperature or np.nan)
    return value, f"{value:g}"
