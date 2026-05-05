"""Orchestration de chargement ARPES entre UI/session et loaders.

Les loaders restent dans `arpes_io.py`. Ce module centralise seulement la
preparation du contexte et l'application des metadata de chargement a la
session, sans dependance PyQt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np

from arpes_models import LoadContext, MetadataSource


LoadFunc = Callable[..., dict | None]
BestAngleLoadFunc = Callable[[str, Any, float, dict], tuple[dict | None, dict]]
LoaderLabelFunc = Callable[[str | None, dict | None], str]


@dataclass
class LoaderOrchestratorResult:
    data: dict | None
    angle_offsets: dict = field(default_factory=dict)
    context: LoadContext = field(default_factory=LoadContext)


class LoaderOrchestrator:
    """Prepare et applique un chargement sans connaitre les widgets."""

    def __init__(self, load_func: LoadFunc, loader_label_func: LoaderLabelFunc):
        self.load_func = load_func
        self.loader_label_func = loader_label_func

    def build_context(
        self,
        entry: Any,
        *,
        hv: float | None,
        angle_offsets: dict | None,
        bessy_energy_reference: str,
    ) -> LoadContext:
        return LoadContext(
            hv=float(hv) if hv is not None else None,
            temperature=entry.meta.temperature if entry.meta.temperature and entry.meta.temperature > 0 else None,
            azi=entry.meta.azi,
            pol=entry.meta.polarization,
            angle_offsets=dict(angle_offsets or {}),
            bessy_energy_reference=bessy_energy_reference,
        )

    def load(
        self,
        path: str,
        entry: Any,
        *,
        work_func: float,
        ef_offset: float,
        hv: float | None,
        angle_offsets: dict | None,
        bessy_energy_reference: str,
        best_angle_load_func: BestAngleLoadFunc | None = None,
    ) -> LoaderOrchestratorResult:
        context = self.build_context(
            entry,
            hv=hv,
            angle_offsets=angle_offsets,
            bessy_energy_reference=bessy_energy_reference,
        )
        resolved_offsets = dict(context.angle_offsets or {})
        if resolved_offsets and Path(path).is_file() and best_angle_load_func is not None:
            data, resolved_offsets = best_angle_load_func(path, entry, float(hv or 0.0), resolved_offsets)
        else:
            data = self.load_func(
                path,
                work_func,
                ef_offset,
                hv=context.hv,
                temperature=context.temperature,
                azi=context.azi,
                pol=context.pol,
                angle_offsets=resolved_offsets,
                bessy_energy_reference=context.bessy_energy_reference,
            )
        return LoaderOrchestratorResult(
            data=data,
            angle_offsets=resolved_offsets,
            context=context,
        )

    def apply_loaded_metadata(self, data: dict, entry: Any) -> dict:
        """Met a jour FileEntry depuis les metadata du loader."""
        md = data.get("metadata", {}) or {}
        source_format = str(data.get("source_format") or md.get("source_format") or "")
        entry.meta.source_format = source_format
        entry.meta.loader_label = self.loader_label_func(source_format, md)
        t_md = md.get("temperature")
        try:
            t_md = float(t_md) if t_md is not None else None
        except (TypeError, ValueError):
            t_md = None
        if t_md is not None and np.isfinite(t_md) and t_md > 0:
            entry.meta.temperature = t_md
        return md

    def resolve_hv_after_load(
        self,
        data: dict,
        entry: Any,
        *,
        hv_for_load: float | None,
        hv_from_logbook: bool,
    ) -> MetadataSource:
        hv_in_data = data.get("hv")
        try:
            hv_file = float(hv_in_data) if hv_in_data is not None else None
        except (TypeError, ValueError):
            hv_file = None
        if hv_file is not None and np.isfinite(hv_file) and hv_file > 0:
            entry.meta.hv = hv_file
            return MetadataSource("hv", hv_file, "file", "eV")
        if hv_for_load and hv_for_load > 0:
            hv_val = float(hv_for_load)
            entry.meta.hv = hv_val
            return MetadataSource("hv", hv_val, "logbook" if hv_from_logbook else "manual", "eV")
        return MetadataSource("hv", None, "unknown", "eV")
