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
    b_angstrom: float = 0.0
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
            b_angstrom=_finite_positive(
                data.get("b_angstrom", data.get("crystal_b_angstrom"))
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
            b_angstrom=_finite_positive(getattr(meta, "crystal_b_angstrom", 0.0)),
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
            b_angstrom=self.b_angstrom or other.b_angstrom,
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


def sample_for_entry(session: Any, entry: Any, entry_key: str | None = None) -> SampleConfig:
    """Resolve sample metadata: file meta → per-subfolder config → session default.

    ``entry_key`` is the session file key (folder-relative path); when given,
    the per-subfolder ``session.sample_configs`` of its top-level folder is the
    user's explicit override and takes priority over the file meta and over
    ``session.current_sample``. Priority: per-subfolder config → file meta →
    session default. Callers without a key keep the historical two-level
    resolution (file meta → session default), since ``folder_sample`` is empty.
    """
    file_sample = SampleConfig.from_meta(getattr(entry, "meta", None))
    folder_sample = SampleConfig()
    if entry_key:
        from arpes.core.sample_layout import sample_key_for_entry_key
        configs = getattr(session, "sample_configs", {}) or {}
        raw = configs.get(sample_key_for_entry_key(entry_key))
        if raw:
            folder_sample = SampleConfig.from_dict(raw)
    session_sample = SampleConfig.from_dict(getattr(session, "current_sample", {}) or {})
    # Explicit per-subfolder setup wins over file meta (which is often just an
    # echo of a previous UI value or a logbook default); without an explicit
    # config the file meta still wins over the session-wide default.
    return folder_sample.merge_missing_from(file_sample).merge_missing_from(session_sample)


def require_lattice_a(sample: SampleConfig, *, context: str = "sample") -> float:
    """Return lattice a or raise a user-facing error for publishable physics."""
    if sample.has_lattice_a:
        return float(sample.a_angstrom)
    label = _text(sample.formula or sample.mp_id or context) or context
    raise ValueError(
        f"Missing lattice parameter a for {label}. "
        "Set crystal_a_angstrom/SampleConfig before publishable physical calculations."
    )


def work_function_for_entry(session: Any, entry: Any, *, fallback: float,
                            entry_key: str | None = None) -> float:
    """Resolve work function with SampleConfig before transitional UI fallback."""
    sample = sample_for_entry(session, entry, entry_key)
    if sample.has_work_function:
        return float(sample.work_function_eV)
    return float(fallback)


def lattice_a_for_entry(session: Any, entry: Any, *, fallback: float = 0.0,
                        entry_key: str | None = None) -> float:
    """Resolve lattice a with SampleConfig before explicit caller fallback."""
    sample = sample_for_entry(session, entry, entry_key)
    if sample.has_lattice_a:
        return float(sample.a_angstrom)
    return float(fallback)
