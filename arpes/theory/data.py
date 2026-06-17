"""Dataclasses for optional DFT band overlays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

from arpes.theory.labels import _clean_label


@dataclass(frozen=True)
class TheoryBandData:
    source: str
    material_id: str
    formula: str = ""
    efermi: float = 0.0
    k_distance: list[float] = field(default_factory=list)
    bands: list[list[float]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    path_type: str = "setyawan_curtarolo"
    warning: str = ""
    schema_version: int = 3
    band_meta: list[dict[str, Any]] = field(default_factory=list)
    band_character: list[str] = field(default_factory=list)
    branches: list[dict[str, Any]] = field(default_factory=list)
    crystal_system: str = ""
    k_distance_abs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "material_id": self.material_id,
            "formula": self.formula,
            "efermi": self.efermi,
            "k_distance": self.k_distance,
            "bands": self.bands,
            "labels": self.labels,
            "path_type": self.path_type,
            "warning": self.warning,
            "schema_version": self.schema_version,
            "band_meta": self.band_meta,
            "band_character": self.band_character,
            "branches": self.branches,
            "crystal_system": self.crystal_system,
            "k_distance_abs": self.k_distance_abs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TheoryBandData":
        data = data or {}
        return cls(
            source=str(data.get("source") or ""),
            material_id=str(data.get("material_id") or ""),
            formula=str(data.get("formula") or ""),
            efermi=_finite_float(data.get("efermi"), 0.0),
            k_distance=[float(x) for x in data.get("k_distance", [])],
            bands=[[float(v) for v in row] for row in data.get("bands", [])],
            labels=[{**dict(x), "label": _clean_label(x.get("label", ""))} for x in data.get("labels", [])],
            path_type=str(data.get("path_type") or "setyawan_curtarolo"),
            warning=str(data.get("warning") or ""),
            schema_version=int(_finite_float(data.get("schema_version"), 1)),
            band_meta=[dict(x) for x in (data.get("band_meta") or [])],
            band_character=[str(x) for x in (data.get("band_character") or [])],
            branches=[dict(x) for x in (data.get("branches") or [])],
            crystal_system=str(data.get("crystal_system") or ""),
            k_distance_abs=[float(x) for x in (data.get("k_distance_abs") or [])],
        )


@dataclass(frozen=True)
class TheoryOverlayConfig:
    enabled: bool = False
    segment: str = ""
    energy_shift: float = 0.0
    mu_shift: float | None = None
    z_scale: float = 1.0
    k_shift: float = 0.0
    k_scale: float = 1.0
    alpha: float = 0.65
    max_bands: int = 10
    mirror_gamma: bool = False
    band_indices: str = ""
    ef_window: float = 0.0
    color_by_band: bool = True
    crystal_a: float = 0.0
    path_convention: str = "mp_bulk"
    gamma_center: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "segment": self.segment,
            "energy_shift": -float(self.mu_shift) if self.mu_shift is not None else float(self.energy_shift),
            "mu_shift": -float(self.energy_shift) if self.mu_shift is None else float(self.mu_shift),
            "z_scale": float(self.z_scale),
            "k_shift": float(self.k_shift),
            "k_scale": float(self.k_scale),
            "alpha": float(self.alpha),
            "max_bands": int(self.max_bands),
            "mirror_gamma": bool(self.mirror_gamma),
            "band_indices": str(self.band_indices),
            "ef_window": float(self.ef_window),
            "color_by_band": bool(self.color_by_band),
            "crystal_a": float(self.crystal_a),
            "path_convention": str(self.path_convention or "mp_bulk"),
            "gamma_center": (
                None if self.gamma_center is None else float(self.gamma_center)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TheoryOverlayConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            segment=str(data.get("segment") or ""),
            energy_shift=_finite_float(data.get("energy_shift"), 0.0),
            mu_shift=(
                _finite_float(data.get("mu_shift"), 0.0)
                if "mu_shift" in data else None
            ),
            z_scale=max(0.001, _finite_float(data.get("z_scale"), 1.0) or 1.0),
            k_shift=_finite_float(data.get("k_shift"), 0.0),
            k_scale=_finite_float(data.get("k_scale"), 1.0) or 1.0,
            alpha=_finite_float(data.get("alpha"), 0.65),
            max_bands=max(1, int(_finite_float(data.get("max_bands"), 10))),
            mirror_gamma=bool(data.get("mirror_gamma", False)),
            band_indices=str(data.get("band_indices") or ""),
            ef_window=max(0.0, _finite_float(data.get("ef_window"), 0.0)),
            color_by_band=bool(data.get("color_by_band", True)),
            crystal_a=max(0.0, _finite_float(data.get("crystal_a"), 0.0)),
            path_convention=str(data.get("path_convention") or "mp_bulk"),
            gamma_center=(
                _finite_float(data.get("gamma_center"), 0.0)
                if data.get("gamma_center") is not None else None
            ),
        )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)
