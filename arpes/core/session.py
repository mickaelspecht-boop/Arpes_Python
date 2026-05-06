"""Session JSON et etat persistant pour ARPES Explorer.

Ce module preserve le format `.arpes_session.json` existant. Il est separe de
l'interface PyQt pour permettre des tests sans lancer l'application.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

import numpy as np


@dataclass
class FitParams:
    n_pairs: int = 1
    ev_start: float = -0.90
    ev_end: float = -0.005
    k_min: float = -0.80
    k_max: float = 0.80
    smooth_fit: float = 2.0
    smooth_detect: float = 3.0
    gamma_init: float = 0.08
    gamma_max: float = 0.30
    xg_range: float = 0.10
    center_init: float = 0.0
    k0_max: Optional[float] = None
    width_mode: str = "symmetric"
    min_amplitude: float = 0.01
    max_jump: float = 0.20
    scan_direction: str = "up"
    dE_meV: float = 15.0
    dk_inv_a: float = 0.005
    pairs: list = field(default_factory=lambda: [
        {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
    ])


@dataclass
class FileMeta:
    hv: float = 0.0
    temperature: float = 0.0
    direction: str = ""
    polarization: str = ""
    meas_no: int = 0
    azi: Optional[float] = None
    polar: Optional[float] = None
    tilt: Optional[float] = None
    source_format: str = ""
    loader_label: str = ""
    formula: str = ""
    mp_id: str = ""


@dataclass
class FileEntry:
    ef_offset: float = 0.052
    edcnorm: bool = False
    view_mode: str = "Raw"
    fit_params: FitParams = field(default_factory=FitParams)
    fit_result: Optional[dict] = None
    meta: FileMeta = field(default_factory=FileMeta)
    fs_center_kx: Optional[float] = None
    fs_center_ky: Optional[float] = None
    grid_correction: dict = field(default_factory=dict)
    ef_correction: dict = field(default_factory=dict)
    theory_overlay: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.fit_result:
            return "fitted"
        return "loaded"


def _known_kwargs(cls, values: dict) -> dict:
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: val for key, val in (values or {}).items() if key in allowed}


def _to_serial(obj):
    """Convertit recursivement np.ndarray / np.floating en types JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serial(v) for v in obj]
    return obj


class Session:
    VERSION = 1

    def __init__(self, folder: Path | None = None, work_func: float = 4.031):
        self.folder: Path | None = folder
        self.work_func: float = work_func
        self.files: dict[str, FileEntry] = {}
        self.logbook_path: str = ""
        self.logbook_sheet: str = ""
        self.logbook_mapping: dict[str, str] = {}
        self.logbook_records: list[dict] = []
        self.kz_logbook_path: str = ""
        self.kz_logbook_sheet: str = ""
        self.kz_logbook_mapping: dict[str, str] = {}
        self.kz_logbook_records: list[dict] = []
        self.gamma_reference: dict = {}
        self.angle_offsets: dict = {}
        self.ef_reference: dict = {}

    @property
    def json_path(self) -> Path | None:
        return self.folder / ".arpes_session.json" if self.folder else None

    def save(self):
        if not self.json_path:
            return
        data = {
            "version": self.VERSION,
            "folder": str(self.folder),
            "work_func": self.work_func,
            "logbook_path": self.logbook_path,
            "logbook_sheet": self.logbook_sheet,
            "logbook_mapping": _to_serial(self.logbook_mapping),
            "logbook_records": _to_serial(self.logbook_records),
            "kz_logbook_path": self.kz_logbook_path,
            "kz_logbook_sheet": self.kz_logbook_sheet,
            "kz_logbook_mapping": _to_serial(self.kz_logbook_mapping),
            "kz_logbook_records": _to_serial(self.kz_logbook_records),
            "gamma_reference": _to_serial(self.gamma_reference),
            "angle_offsets": _to_serial(self.angle_offsets),
            "ef_reference": _to_serial(self.ef_reference),
            "files": {
                name: _to_serial(asdict(entry))
                for name, entry in self.files.items()
            },
        }
        self.json_path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path):
        raw = json.loads(path.read_text())
        self.work_func = raw.get("work_func", 4.031)
        self.logbook_path = raw.get("logbook_path", "")
        self.logbook_sheet = raw.get("logbook_sheet", "")
        self.logbook_mapping = raw.get("logbook_mapping", {})
        self.logbook_records = raw.get("logbook_records", [])
        self.kz_logbook_path = raw.get("kz_logbook_path", "")
        self.kz_logbook_sheet = raw.get("kz_logbook_sheet", "")
        self.kz_logbook_mapping = raw.get("kz_logbook_mapping", {})
        self.kz_logbook_records = raw.get("kz_logbook_records", [])
        self.gamma_reference = raw.get("gamma_reference", {})
        self.angle_offsets = raw.get("angle_offsets", {}) or {}
        self.ef_reference = raw.get("ef_reference", {}) or {}
        self.files = {}
        for name, edict in raw.get("files", {}).items():
            fp = FitParams(**_known_kwargs(FitParams, edict.get("fit_params", {})))
            mt = FileMeta(**_known_kwargs(FileMeta, edict.get("meta", {})))
            entry = FileEntry(
                ef_offset=edict.get("ef_offset", 0.052),
                edcnorm=edict.get("edcnorm", False),
                view_mode=edict.get("view_mode", "Raw"),
                fit_params=fp,
                fit_result=edict.get("fit_result"),
                meta=mt,
                fs_center_kx=edict.get("fs_center_kx"),
                fs_center_ky=edict.get("fs_center_ky"),
                grid_correction=edict.get("grid_correction", {}) or {},
                ef_correction=edict.get("ef_correction", {}) or {},
                theory_overlay=edict.get("theory_overlay", {}) or {},
            )
            self.files[name] = entry

    def get_or_create(self, filename: str) -> FileEntry:
        if filename not in self.files:
            self.files[filename] = FileEntry()
        return self.files[filename]

    def set_fit_result(self, filename: str, fr: dict):
        entry = self.get_or_create(filename)
        entry.fit_result = _to_serial(fr)
        self.save()

    def key_for_path(self, path: str | Path) -> str:
        """Cle stable de session : chemin relatif au dossier racine si possible."""
        p = Path(path)
        if self.folder is not None:
            try:
                return str(p.resolve().relative_to(self.folder.resolve()))
            except Exception:
                pass
        return p.name
