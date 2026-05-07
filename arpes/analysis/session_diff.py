"""Comparaison pure de deux payloads de session ARPES."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from arpes.analysis.results import BranchResult, compute_results
from arpes.core.session import FileEntry, Session


@dataclass(frozen=True)
class SessionDiffRow:
    filename: str
    branch: str
    pair_index: int
    status: str
    delta_kF: float = float("nan")
    delta_vF: float = float("nan")
    delta_m_star: float = float("nan")
    a_kF: float = float("nan")
    a_kF_sigma: float = float("nan")
    b_kF: float = float("nan")
    b_kF_sigma: float = float("nan")
    a_vF: float = float("nan")
    a_vF_sigma: float = float("nan")
    b_vF: float = float("nan")
    b_vF_sigma: float = float("nan")


def compare_session_payloads(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> list[SessionDiffRow]:
    """Compare deux payloads JSON de session.

    Retourne une ligne par fichier commun et par branche/pair disponible. Les
    fichiers absents d'un côté produisent une ligne ``status`` explicite.
    """
    session_a = Session()
    session_b = Session()
    session_a.load_from_payload(payload_a or {})
    session_b.load_from_payload(payload_b or {})

    rows: list[SessionDiffRow] = []
    names = sorted(set(session_a.files) | set(session_b.files))
    for name in names:
        entry_a = session_a.files.get(name)
        entry_b = session_b.files.get(name)
        if entry_a is None:
            rows.append(SessionDiffRow(name, "", -1, "absent A"))
            continue
        if entry_b is None:
            rows.append(SessionDiffRow(name, "", -1, "absent B"))
            continue
        rows.extend(_compare_entries(name, entry_a, entry_b))
    return rows


def _compare_entries(filename: str, entry_a: FileEntry, entry_b: FileEntry) -> list[SessionDiffRow]:
    a_bundle = compute_results(
        entry_a.fit_result,
        crystal_a_angstrom=float(entry_a.meta.crystal_a_angstrom or 0.0),
    )
    b_bundle = compute_results(
        entry_b.fit_result,
        crystal_a_angstrom=float(entry_b.meta.crystal_a_angstrom or 0.0),
    )
    a_map = {(br.branch, br.pair_index): br for br in a_bundle.branches}
    b_map = {(br.branch, br.pair_index): br for br in b_bundle.branches}
    keys = sorted(set(a_map) | set(b_map), key=lambda item: (item[1], item[0]))
    if not keys:
        return [SessionDiffRow(filename, "", -1, "pas de fit commun")]

    rows: list[SessionDiffRow] = []
    for branch, pair_index in keys:
        br_a = a_map.get((branch, pair_index))
        br_b = b_map.get((branch, pair_index))
        if br_a is None:
            rows.append(SessionDiffRow(filename, branch, pair_index, "branche absente A"))
            continue
        if br_b is None:
            rows.append(SessionDiffRow(filename, branch, pair_index, "branche absente B"))
            continue
        rows.append(_diff_branch(filename, br_a, br_b))
    return rows


def _diff_branch(filename: str, a: BranchResult, b: BranchResult) -> SessionDiffRow:
    return SessionDiffRow(
        filename=filename,
        branch=a.branch,
        pair_index=a.pair_index,
        status="OK" if _finite_any(a.kF_at_EF, b.kF_at_EF) else "resultat non fini",
        delta_kF=_sub(b.kF_at_EF, a.kF_at_EF),
        delta_vF=_sub(b.vF_eV_pi_a, a.vF_eV_pi_a),
        delta_m_star=_sub(b.m_star_over_me, a.m_star_over_me),
        a_kF=a.kF_at_EF,
        a_kF_sigma=a.kF_at_EF_sigma,
        b_kF=b.kF_at_EF,
        b_kF_sigma=b.kF_at_EF_sigma,
        a_vF=a.vF_eV_pi_a,
        a_vF_sigma=a.vF_sigma,
        b_vF=b.vF_eV_pi_a,
        b_vF_sigma=b.vF_sigma,
    )


def _sub(b_value: float, a_value: float) -> float:
    if math.isfinite(a_value) and math.isfinite(b_value):
        return float(b_value - a_value)
    return float("nan")


def _finite_any(*values: float) -> bool:
    return any(math.isfinite(v) for v in values)
