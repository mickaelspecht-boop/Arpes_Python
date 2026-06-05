"""Sample metadata model used by physics and export code.

The goal is to make lattice/work-function assumptions explicit instead of
silently falling back to one material.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


def _finite_positive(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out > 0:
        return out
    return 0.0


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class SampleConfig:
    """Physical sample parameters with provenance.

    Zero/empty values mean "unknown". Code that needs a real lattice parameter
    should check ``has_lattice_a`` and fail loudly when it is false.
    """

    formula: str = ""
    a_angstrom: float = 0.0
    c_angstrom: float = 0.0
    work_function_eV: float = 0.0
    space_group: str = ""
    mp_id: str = ""
    lattice_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SampleConfig":
        data = raw or {}
        return cls(
            formula=_text(data.get("formula")),
            a_angstrom=_finite_positive(
                data.get("a_angstrom", data.get("crystal_a_angstrom"))
            ),
            c_angstrom=_finite_positive(
                data.get("c_angstrom", data.get("crystal_c_angstrom"))
            ),
            work_function_eV=_finite_positive(
                data.get("work_function_eV", data.get("work_function"))
            ),
            space_group=_text(data.get("space_group")),
            mp_id=_text(data.get("mp_id")),
            lattice_source=_text(data.get("lattice_source", data.get("source"))),
        )

    @classmethod
    def from_meta(cls, meta: Any) -> "SampleConfig":
        explicit = cls.from_dict(getattr(meta, "sample_config", {}) or {})
        legacy = cls(
            formula=_text(getattr(meta, "formula", "")),
            a_angstrom=_finite_positive(getattr(meta, "crystal_a_angstrom", 0.0)),
            c_angstrom=_finite_positive(getattr(meta, "crystal_c_angstrom", 0.0)),
            work_function_eV=_finite_positive(getattr(meta, "work_function_eV", 0.0)),
            space_group=_text(getattr(meta, "space_group", "")),
            mp_id=_text(getattr(meta, "mp_id", "")),
            lattice_source=_text(getattr(meta, "lattice_source", "")),
        )
        return legacy.merge_missing_from(explicit)

    def merge_missing_from(self, other: "SampleConfig") -> "SampleConfig":
        """Return a config where this object's known fields take priority."""
        return SampleConfig(
            formula=self.formula or other.formula,
            a_angstrom=self.a_angstrom or other.a_angstrom,
            c_angstrom=self.c_angstrom or other.c_angstrom,
            work_function_eV=self.work_function_eV or other.work_function_eV,
            space_group=self.space_group or other.space_group,
            mp_id=self.mp_id or other.mp_id,
            lattice_source=self.lattice_source or other.lattice_source,
        )

    @property
    def has_lattice_a(self) -> bool:
        return self.a_angstrom > 0

    @property
    def has_work_function(self) -> bool:
        return self.work_function_eV > 0


def sample_for_entry(session: Any, entry: Any) -> SampleConfig:
    """Resolve sample metadata with per-file values before session defaults."""
    file_sample = SampleConfig.from_meta(getattr(entry, "meta", None))
    session_sample = SampleConfig.from_dict(getattr(session, "current_sample", {}) or {})
    return file_sample.merge_missing_from(session_sample)


def require_lattice_a(sample: SampleConfig, *, context: str = "sample") -> float:
    """Return lattice a or raise a user-facing error for publishable physics."""
    if sample.has_lattice_a:
        return float(sample.a_angstrom)
    label = _text(sample.formula or sample.mp_id or context) or context
    raise ValueError(
        f"Paramètre de maille a manquant pour {label}. "
        "Renseigne crystal_a_angstrom/SampleConfig avant un calcul physique publiable."
    )
