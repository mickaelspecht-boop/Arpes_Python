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

from arpes.core.sample import SampleConfig


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
    shape: str = "lorentzian"  # "lorentzian" | "voigt" (pseudo-Voigt η_global)
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
    crystal_a_angstrom: float = 0.0
    crystal_c_angstrom: float = 0.0
    work_function_eV: float = 0.0
    space_group: str = ""
    lattice_source: str = ""
    sample_config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    # A.1 — type de scan inféré au load (BM/FS/KZ/EDC/unknown). Source de
    # vérité unique pour pairing BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md).
    scan_kind: str = "unknown"


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
    fs_v0: float = 12.0                     # inner potential (eV) pour calcul kz
    fs_kz_plane: str = "Auto"               # "Gamma" | "Z" | "Auto"
    fs_phi_c_deg: float = 0.0               # rotation cristal/détecteur (deg)
    fs_bz_crystal_visible: bool = False     # overlay BZ cristal MP
    fs_hs_crystal_visible: bool = False     # labels HS cristal (Γ/X/M ou Z/R/A)
    fs_bz_crystal_force_override: bool = False  # override mismatch a_ARPES/a_MP > 2%
    fs_lattice: dict = field(default_factory=dict)  # cache lattice MP {a,b,c,alpha,beta,gamma,bravais,space_group,mp_id}
    propagate_distortion_to_fs: bool = False  # opt-in : applique distortion BM aux coupes du volume FS
    grid_correction: dict = field(default_factory=dict)
    ef_correction: dict = field(default_factory=dict)
    bm_distortion: dict = field(default_factory=dict)
    theory_overlay: dict = field(default_factory=dict)
    band_analysis: dict = field(default_factory=dict)  # TB fit / kink / gap results
    fs_pockets: list[dict] = field(default_factory=list)
    dft_grid_path: str = ""  # chemin npz DFT 3D pour comparaison poches
    fit_zones: list[dict] = field(default_factory=list)
    # each zone : {id, label, color_idx, active, fit_params, fit_result|None}
    active_zone_id: Optional[str] = None
    annotations: dict[str, list[dict]] = field(default_factory=dict)
    # Survit save/load : flags meta gamma déposés par apply_bm_gamma_axis_shift
    # (bm_gamma_axis_centered, bm_gamma_axis_shift, fs_gamma_axis_*,
    # bm_gamma_reference_*). Restauré dans raw_data["metadata"] au load.
    meta_gamma_state: dict = field(default_factory=dict)
    # A.3 — override manuel pairing BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md).
    # Si défini : force le rattachement de cette BM à la FS au path donné,
    # court-circuitant l'auto-discovery par métadonnées.
    parent_fs_path: Optional[str] = None

    @property
    def status(self) -> str:
        if self.fit_result:
            return "fitted"
        return "loaded"


def _known_kwargs(cls, values: dict) -> dict:
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: val for key, val in (values or {}).items() if key in allowed}


def normalize_tags(value) -> list[str]:
    """Nettoie une saisie tags separee par virgules, sans changer la casse."""
    if value is None:
        raw: list[str] = []
    elif isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for item in value:
            raw.extend(str(item).split(","))
    else:
        raw = [str(value)]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = str(item).strip()
        if not tag:
            continue
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def session_tags(session: "Session") -> list[str]:
    """Retourne les tags existants, tries sans sensibilite a la casse."""
    seen: dict[str, str] = {}
    for entry in getattr(session, "files", {}).values():
        for tag in normalize_tags(getattr(entry.meta, "tags", [])):
            seen.setdefault(tag.casefold(), tag)
    return sorted(seen.values(), key=lambda x: x.casefold())


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
        # rel_subdir -> {"path": str, "sheet": str, "n": int} ; records eux-mêmes
        # restent dans logbook_records taggés "_subfolder_rel".
        self.scoped_logbooks: dict[str, dict] = {}
        self.kz_logbook_path: str = ""
        self.kz_logbook_sheet: str = ""
        self.kz_logbook_mapping: dict[str, str] = {}
        self.kz_logbook_records: list[dict] = []
        self.gamma_reference: dict = {}
        self.angle_offsets: dict = {}
        self.ef_reference: dict = {}
        self.fit_panel_sections: dict[str, bool] = {}
        self.fit_panel_preset: str = "Custom"
        self.session_notes: str = ""
        self.current_sample: dict = {}

    def reset(self, *, keep_folder: bool = True) -> None:
        """Remet la session à zéro (fits, logbook, calibs…). Garde le dossier."""
        folder = self.folder if keep_folder else None
        wf = self.work_func
        self.__init__(folder=folder, work_func=wf)

    @property
    def json_path(self) -> Path | None:
        return self.folder / ".arpes_session.json" if self.folder else None

    def to_payload(self) -> dict:
        """Serialize session state to a portable JSON-safe dict.

        The folder path is stored as a hint only; on load the recipient
        relocates to their own data folder and file keys (relative paths)
        are resolved against it.
        """
        return {
            "version": self.VERSION,
            "folder": str(self.folder) if self.folder else "",
            "folder_hint": self.folder.name if self.folder else "",
            "work_func": self.work_func,
            "logbook_path": self.logbook_path,
            "logbook_sheet": self.logbook_sheet,
            "logbook_mapping": _to_serial(self.logbook_mapping),
            "logbook_records": _to_serial(self.logbook_records),
            "scoped_logbooks": _to_serial(self.scoped_logbooks),
            "kz_logbook_path": self.kz_logbook_path,
            "kz_logbook_sheet": self.kz_logbook_sheet,
            "kz_logbook_mapping": _to_serial(self.kz_logbook_mapping),
            "kz_logbook_records": _to_serial(self.kz_logbook_records),
            "gamma_reference": _to_serial(self.gamma_reference),
            "angle_offsets": _to_serial(self.angle_offsets),
            "ef_reference": _to_serial(self.ef_reference),
            "fit_panel_sections": dict(self.fit_panel_sections),
            "fit_panel_preset": str(self.fit_panel_preset or "Custom"),
            "session_notes": str(self.session_notes or ""),
            "current_sample": _to_serial(self.current_sample),
            "files": {
                name: _to_serial(asdict(entry))
                for name, entry in self.files.items()
            },
        }

    def save(self):
        if not self.json_path:
            return
        self.json_path.write_text(json.dumps(self.to_payload(), indent=2))

    def save_to(self, path: Path) -> None:
        """Write session payload to an arbitrary path (Save As / Export)."""
        Path(path).write_text(json.dumps(self.to_payload(), indent=2))

    def load_from_payload(self, raw: dict) -> None:
        self.work_func = raw.get("work_func", 4.031)
        self.logbook_path = raw.get("logbook_path", "")
        self.logbook_sheet = raw.get("logbook_sheet", "")
        self.logbook_mapping = raw.get("logbook_mapping", {})
        self.logbook_records = raw.get("logbook_records", [])
        self.scoped_logbooks = raw.get("scoped_logbooks", {}) or {}
        self.kz_logbook_path = raw.get("kz_logbook_path", "")
        self.kz_logbook_sheet = raw.get("kz_logbook_sheet", "")
        self.kz_logbook_mapping = raw.get("kz_logbook_mapping", {})
        self.kz_logbook_records = raw.get("kz_logbook_records", [])
        self.gamma_reference = raw.get("gamma_reference", {})
        self.angle_offsets = raw.get("angle_offsets", {}) or {}
        self.ef_reference = raw.get("ef_reference", {}) or {}
        self.fit_panel_sections = dict(raw.get("fit_panel_sections", {}) or {})
        self.fit_panel_preset = str(raw.get("fit_panel_preset", "Custom") or "Custom")
        self.session_notes = str(raw.get("session_notes", "") or "")
        self.current_sample = SampleConfig.from_dict(
            raw.get("current_sample", {}) or {}
        ).to_dict()
        self.files = {}
        for name, edict in raw.get("files", {}).items():
            fp = FitParams(**_known_kwargs(FitParams, edict.get("fit_params", {})))
            mt = FileMeta(**_known_kwargs(FileMeta, edict.get("meta", {})))
            mt.sample_config = SampleConfig.from_meta(mt).to_dict()
            entry = FileEntry(
                ef_offset=edict.get("ef_offset", 0.052),
                edcnorm=edict.get("edcnorm", False),
                view_mode=edict.get("view_mode", "Raw"),
                fit_params=fp,
                fit_result=edict.get("fit_result"),
                meta=mt,
                fs_center_kx=edict.get("fs_center_kx"),
                fs_center_ky=edict.get("fs_center_ky"),
                fs_v0=float(edict.get("fs_v0", 12.0) or 12.0),
                fs_kz_plane=str(edict.get("fs_kz_plane", "Auto") or "Auto"),
                fs_phi_c_deg=float(edict.get("fs_phi_c_deg", 0.0) or 0.0),
                fs_bz_crystal_visible=bool(edict.get("fs_bz_crystal_visible", False)),
                fs_hs_crystal_visible=bool(edict.get("fs_hs_crystal_visible", False)),
                fs_bz_crystal_force_override=bool(edict.get("fs_bz_crystal_force_override", False)),
                fs_lattice=edict.get("fs_lattice", {}) or {},
                propagate_distortion_to_fs=bool(edict.get("propagate_distortion_to_fs", False)),
                grid_correction=edict.get("grid_correction", {}) or {},
                ef_correction=edict.get("ef_correction", {}) or {},
                bm_distortion=edict.get("bm_distortion", {}) or {},
                theory_overlay=edict.get("theory_overlay", {}) or {},
                band_analysis=edict.get("band_analysis", {}) or {},
                fs_pockets=list(edict.get("fs_pockets", []) or []),
                dft_grid_path=str(edict.get("dft_grid_path", "") or ""),
                fit_zones=list(edict.get("fit_zones", []) or []),
                active_zone_id=edict.get("active_zone_id"),
                annotations=edict.get("annotations", {}) or {},
                meta_gamma_state=dict(edict.get("meta_gamma_state", {}) or {}),
                parent_fs_path=edict.get("parent_fs_path"),
            )
            self.files[name] = entry

    def load(self, path: Path):
        self.load_from_payload(json.loads(Path(path).read_text()))

    def get_or_create(self, filename: str) -> FileEntry:
        if filename not in self.files:
            self.files[filename] = FileEntry()
        return self.files[filename]

    def set_fit_result(self, filename: str, fr: dict):
        # Delegate to fit_result_store so the active-zone mirror is kept in
        # sync automatically (single-setter pattern, per architect audit).
        from arpes.core.fit_result_store import set_fit_result as _store_set
        entry = self.get_or_create(filename)
        _store_set(entry, _to_serial(fr))
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
