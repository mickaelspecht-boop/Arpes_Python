"""Contrats runtime legers pour ARPES Explorer.

Ce module ne change pas le format session existant. Il fournit des dataclasses
compatibles avec les dicts deja stockes dans `.arpes_session.json`, afin que les
prochaines extractions puissent s'appuyer sur des formes explicites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _median_finite(values: Any) -> float | None:
    if values is None:
        return None
    if isinstance(values, (int, float)):
        values = [values]
    flat: list[float] = []
    try:
        iterator = iter(values)
    except TypeError:
        return None
    for item in iterator:
        if isinstance(item, (list, tuple)):
            for sub in item:
                val = _finite_float(sub)
                if val is not None:
                    flat.append(val)
        else:
            val = _finite_float(item)
            if val is not None:
                flat.append(val)
    if not flat:
        return None
    flat.sort()
    mid = len(flat) // 2
    if len(flat) % 2:
        return flat[mid]
    return 0.5 * (flat[mid - 1] + flat[mid])


@dataclass(frozen=True)
class MetadataSource:
    """Valeur metadata avec provenance explicite."""

    key: str
    value: Any = None
    source: str = "unknown"
    unit: str = ""
    detail: str = ""
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source or "unknown",
            "unit": self.unit,
            "detail": self.detail,
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | Any, *, key: str = "") -> "MetadataSource":
        if isinstance(data, dict):
            return cls(
                key=str(data.get("key") or key),
                value=data.get("value"),
                source=str(data.get("source") or "unknown"),
                unit=str(data.get("unit") or ""),
                detail=str(data.get("detail") or ""),
                warning=str(data.get("warning") or ""),
            )
        return cls(key=str(key), value=data)


@dataclass(frozen=True)
class ResolutionInfo:
    """Resolution instrumentale utilisee par le fit MDC."""

    dE_meV: float = 15.0
    dk_inv_a: float = 0.005
    source: str = "default"

    @property
    def dE_eV(self) -> float:
        return float(self.dE_meV) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dE_eV": self.dE_eV,
            "dE_meV": float(self.dE_meV),
            "dk_inv_a": float(self.dk_inv_a),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResolutionInfo":
        data = data or {}
        dE_meV = _finite_float(data.get("dE_meV"))
        if dE_meV is None:
            dE_eV = _finite_float(data.get("dE_eV"))
            dE_meV = 1000.0 * dE_eV if dE_eV is not None else 15.0
        dk_inv_a = _finite_float(data.get("dk_inv_a"), 0.005)
        return cls(
            dE_meV=float(dE_meV),
            dk_inv_a=float(dk_inv_a if dk_inv_a is not None else 0.005),
            source=str(data.get("source") or "default"),
        )


@dataclass(frozen=True)
class FitGammaSummary:
    """Resume stable des largeurs Gamma d'un fit MDC."""

    gamma_brut_median: float | None = None
    gamma_corrige_median: float | None = None
    gamma_min_median: float | None = None
    resolution_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "gamma_brut_median": self.gamma_brut_median,
            "gamma_corrige_median": self.gamma_corrige_median,
            "gamma_min_median": self.gamma_min_median,
            "resolution_limited": bool(self.resolution_limited),
        }

    @classmethod
    def from_fit_result(cls, fit_result: dict[str, Any] | None) -> "FitGammaSummary":
        fr = fit_result or {}
        gamma_brut = _median_finite(fr.get("gamma_brut", fr.get("gamma")))
        gamma_corrige = _median_finite(fr.get("gamma_corrige"))
        gamma_min = _median_finite(fr.get("gamma_min"))
        limited = False
        if gamma_brut is not None and gamma_corrige is not None:
            limited = gamma_corrige < 0.3 * gamma_brut
        return cls(
            gamma_brut_median=gamma_brut,
            gamma_corrige_median=gamma_corrige,
            gamma_min_median=gamma_min,
            resolution_limited=limited,
        )


@dataclass(frozen=True)
class LoadContext:
    """Contexte minimal passe a un chargement de fichier."""

    hv: float | None = None
    temperature: float | None = None
    azi: float | None = None
    pol: str = ""
    angle_offsets: dict[str, Any] = field(default_factory=dict)
    bessy_energy_reference: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hv": self.hv,
            "temperature": self.temperature,
            "azi": self.azi,
            "pol": self.pol,
            "angle_offsets": dict(self.angle_offsets or {}),
            "bessy_energy_reference": self.bessy_energy_reference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LoadContext":
        data = data or {}
        return cls(
            hv=_finite_float(data.get("hv")),
            temperature=_finite_float(data.get("temperature")),
            azi=_finite_float(data.get("azi")),
            pol=str(data.get("pol") or ""),
            angle_offsets=dict(data.get("angle_offsets") or {}),
            bessy_energy_reference=str(data.get("bessy_energy_reference") or "auto"),
        )
