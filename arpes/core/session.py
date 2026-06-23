"""Session JSON and persistent state for ARPES Explorer.

This module preserves the existing `.arpes_session.json` format. It is
separate from the PyQt interface so tests can run without launching the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional
import json
import os
import warnings

import numpy as np

from arpes.core.sample import SampleConfig


# P4.8: historical default EF offset (eV). Value inherited from an original
# CLS dataset; kept for session compatibility but with no physical scope for
# other formats (the loader neutralizes it; see load_controller).
DEFAULT_EF_OFFSET_EV = 0.052


class SessionVersionError(RuntimeError):
    """Explicit refusal to load a session from a newer schema.

    Raised when `payload["version"] > Session.VERSION`: future fields cannot be
    interpreted reliably, so refuse instead of silently dropping them (P1.6
    audit: 3 fields added without a bump = silent cross-version loss).
    """


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
    hold_center: bool = False
    hold_gamma: bool = False
    min_amplitude: float = 0.01
    max_jump: float = 0.20
    mdc_energy_window: float = 0.02  # eV; integrates ±half over E per MDC = primary noise control (doesn't broaden Γ). 0 = single row.
    mdc_energy_step: float = 0.0  # eV; >0 fits one MDC every `step` (Igor step). 0 = every energy row.
    scan_direction: str = "up"
    dE_meV: float = 15.0
    dk_inv_a: float = 0.005
    pairs: list = field(default_factory=lambda: [
        {"kF_init": 0.30, "gamma_init": 0.08, "gamma_max": 0.30}
    ])


@dataclass
class FitZone:
    """Typed schema for a multi-zone fit area (P3.4).

    Runtime storage remains a ``dict`` (≥6 consumers access it by key plus the
    ``entry.fit_result`` mirror), but this dataclass is the canonical schema and
    validated constructor: ``from_dict`` fills missing defaults and **warns
    loudly** on unknown keys instead of losing them silently at load time (see
    audit P3.4). ``to_dict`` converts back for storage, so no consumer needs to
    change.
    """
    id: str
    label: str
    color_idx: int = 0
    active: bool = True
    fit_model: str = "peak_pair"
    fit_params: dict = field(default_factory=dict)
    fit_result: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> "FitZone":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            warnings.warn(
                f"FitZone: unknown keys ignored on load {sorted(unknown)} "
                "(derived schema? bump Session.VERSION if intentional).",
                stacklevel=2,
            )
        return cls(
            id=str(d.get("id", "")),
            label=str(d.get("label", "")),
            color_idx=int(d.get("color_idx", 0) or 0),
            active=bool(d.get("active", True)),
            fit_model=str(d.get("fit_model", "peak_pair") or "peak_pair"),
            fit_params=dict(d.get("fit_params", {}) or {}),
            fit_result=d.get("fit_result"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_fit_zones(zones: list) -> list[dict]:
    """Ensure every zone has the canonical keys (defaults filled) and log
    unknown keys. Preserve order. P3.4."""
    out: list[dict] = []
    for z in zones or []:
        if isinstance(z, dict):
            out.append(FitZone.from_dict(z).to_dict())
    return out


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
    crystal_b_angstrom: float = 0.0
    crystal_c_angstrom: float = 0.0
    work_function_eV: float = 0.0
    space_group: str = ""
    lattice_source: str = ""
    sample_config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    # A.1: scan type inferred on load (BM/FS/KZ/EDC/unknown). Single source of
    # truth for BM↔FS pairing (see BM_FS_ORGANIZATION_PLAN.md).
    scan_kind: str = "unknown"


@dataclass
class FileEntry:
    ef_offset: float = DEFAULT_EF_OFFSET_EV
    edcnorm: bool = False
    view_mode: str = "Raw"
    fit_params: FitParams = field(default_factory=FitParams)
    fit_result: Optional[dict] = None
    meta: FileMeta = field(default_factory=FileMeta)
    fs_center_kx: Optional[float] = None
    fs_center_ky: Optional[float] = None
    fs_rotation_deg: float = 0.0             # display rotation of centered FS map (deg)
    fs_v0: float = 12.0                     # inner potential (eV) for kz calculation
    fs_kz_plane: str = "Auto"               # "Gamma" | "Z" | "Auto"
    fs_phi_c_deg: float = 0.0               # crystal/detector rotation (deg)
    fs_bz_crystal_visible: bool = False     # MP crystal BZ overlay
    fs_hs_crystal_visible: bool = False     # crystal HS labels (Γ/X/M or Z/R/A)
    fs_bz_crystal_force_override: bool = False  # override mismatch a_ARPES/a_MP > 2%
    fs_lattice: dict = field(default_factory=dict)  # cache lattice MP {a,b,c,alpha,beta,gamma,bravais,space_group,mp_id}
    # BZ label convention for the theoretical overlay: pure display renames
    # ({"M": "Σ"}…) applied by bz_high_symmetry_points AND by the logbook
    # direction matching (a renamed corner must keep matching "Γ-Σ" cuts).
    fs_bz_label_overrides: dict = field(default_factory=dict)
    fs_bz_label_preset: str = ""  # key in bz.BZ_LABEL_CONVENTION_PRESETS
    propagate_distortion_to_fs: bool = False  # opt-in: apply BM distortion to FS volume slices
    grid_correction: dict = field(default_factory=dict)
    ef_correction: dict = field(default_factory=dict)
    bm_distortion: dict = field(default_factory=dict)
    theory_overlay: dict = field(default_factory=dict)
    band_analysis: dict = field(default_factory=dict)  # TB fit / kink / gap results
    fs_pockets: list[dict] = field(default_factory=list)
    dft_grid_path: str = ""  # 3D DFT npz path for pocket comparison
    fit_zones: list[dict] = field(default_factory=list)
    # each zone : {id, label, color_idx, active, fit_params, fit_result|None}
    active_zone_id: Optional[str] = None
    annotations: dict[str, list[dict]] = field(default_factory=dict)
    # Survives save/load: gamma metadata flags written by apply_bm_gamma_axis_shift
    # (bm_gamma_axis_centered, bm_gamma_axis_shift, fs_gamma_axis_*,
    # bm_gamma_reference_*). Restored into raw_data["metadata"] on load.
    meta_gamma_state: dict = field(default_factory=dict)
    # A.3: manual BM↔FS pairing override (see BM_FS_ORGANIZATION_PLAN.md).
    # If set: force this BM to attach to the FS at the given path, bypassing
    # metadata auto-discovery.
    parent_fs_path: Optional[str] = None
    # Append-only provenance journal: chronological data transforms + fit
    # operations applied to this signal (timestamped events). Written via
    # core/processing_history.log_event at each mutation, rendered by the
    # processing-log dock/dialog. Survives save/load. NOT the source of truth
    # for current parameters (those live in the typed fields above) — purely an
    # audit trail, so it can never drift away from the actual state.
    processing_history: list[dict] = field(default_factory=list)
    # Independent kF(E) estimate from the Zhang curvature maxima, computed as a
    # cross-check of the Lorentzian MDC dispersion (never carries Gamma — the
    # curvature distorts widths). Same array schema as fit_result
    # ({e_fitted, kF_minus, kF_plus, method:"curvature", c0_alpha, n_pairs}) so
    # the dispersion plot reuses its consumers. None until the user runs it.
    curvature_dispersion: Optional[dict] = None

    @property
    def status(self) -> str:
        if self.fit_result:
            return "fitted"
        return "loaded"


def _known_kwargs(cls, values: dict) -> dict:
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    return {key: val for key, val in (values or {}).items() if key in allowed}


def normalize_tags(value) -> list[str]:
    """Clean comma-separated tag input without changing case."""
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
    """Return existing tags, sorted case-insensitively."""
    seen: dict[str, str] = {}
    for entry in getattr(session, "files", {}).values():
        for tag in normalize_tags(getattr(entry.meta, "tags", [])):
            seen.setdefault(tag.casefold(), tag)
    return sorted(seen.values(), key=lambda x: x.casefold())


def _to_serial(obj):
    """Recursively convert np.ndarray / np.floating to JSON types."""
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


def _atomic_write_json(path: Path, payload: dict, *, keep_backup: bool) -> None:
    """Write the session JSON without risking a truncated file.

    Strategy: write to a tmp file in the same folder, fsync, then atomic
    os.replace. If keep_backup, the old file is copied to `.bak` before the
    replacement (P1.6 audit: non-atomic save() = corruption on crash).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if keep_backup and path.exists():
        bak = path.with_name(path.name + ".bak")
        try:
            os.replace(path, bak)
        except OSError:
            pass
    os.replace(tmp, path)


class Session:
    # v1 -> v2: added band_analysis, fit_zones, active_zone_id on FileEntry
    # (were added without a bump; v1->v2 migration = absent fields -> defaults).
    # v2 -> v3: added convention_registry (P2.6a, angle sign conventions
    # freezable by beamline). v2->v3 migration = absent field -> empty dict.
    # v3 -> v4: added FileEntry.processing_history (append-only provenance
    # journal). v3->v4 migration = absent field -> empty list.
    # v4 -> v5: added FileEntry.curvature_dispersion (Zhang curvature cross-check
    # of the MDC dispersion). v4->v5 migration = absent field -> None.
    VERSION = 5

    def __init__(self, folder: Path | None = None, work_func: float = 0.0):
        self.folder: Path | None = folder
        self.work_func: float = work_func
        self.files: dict[str, FileEntry] = {}
        self.logbook_path: str = ""
        self.logbook_sheet: str = ""
        self.logbook_mapping: dict[str, str] = {}
        self.logbook_records: list[dict] = []
        # rel_subdir -> {"path": str, "sheet": str, "n": int}; records themselves
        # remain in logbook_records tagged "_subfolder_rel".
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
        # Browse-only: suppress automatic setup prompts (sample setup popup,
        # logbook auto-attach) for this session. Fits still ask when needed.
        self.browse_only: bool = False
        # Per-top-level-subfolder sample parameters (folder-load setup dialog).
        # key = first component of the file's session key ("" = root files),
        # value = SampleConfig.to_dict(). Resolution: file meta ->
        # sample_configs[key] -> current_sample (see core/sample.py).
        self.sample_configs: dict = {}
        # P2.6a: angle sign conventions frozen by beamline (geometric key ->
        # dict BeamlineAngleConvention). Empty = all UNCALIBRATED (data-driven).
        self.convention_registry: dict = {}
        # Version of the payload that was actually loaded (None for a new session).
        self.loaded_version: Optional[int] = None

    def reset(self, *, keep_folder: bool = True) -> None:
        """Reset the session (fits, logbook, calibrations...). Keep the folder."""
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
            "browse_only": bool(self.browse_only),
            "sample_configs": _to_serial(self.sample_configs),
            "convention_registry": _to_serial(self.convention_registry),
            "files": {
                name: _to_serial(asdict(entry))
                for name, entry in self.files.items()
            },
        }

    def save(self):
        if not self.json_path:
            return
        _atomic_write_json(self.json_path, self.to_payload(), keep_backup=True)

    def save_to(self, path: Path) -> None:
        """Write session payload to an arbitrary path (Save As / Export)."""
        _atomic_write_json(Path(path), self.to_payload(), keep_backup=False)

    @classmethod
    def _migrate_payload(cls, raw: dict) -> dict:
        """Normalize a payload to the current schema or explicitly refuse it.

        - missing version -> treated as v1 (pre-versioned sessions).
        - version <= VERSION -> in-place migration (new fields = defaults
          provided by load_from_payload).
        - version > VERSION -> SessionVersionError (unknown future schema).
        """
        raw = dict(raw or {})
        try:
            version = int(raw.get("version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        if version > cls.VERSION:
            raise SessionVersionError(
                f"Session version {version} > supported {cls.VERSION}. "
                "Update ARPES Explorer to open this file."
            )
        # v1 -> v2: no data remapping required; added fields
        # (band_analysis, fit_zones, active_zone_id) take their defaults through
        # load_from_payload when absent.
        raw["version"] = version
        return raw

    def load_from_payload(self, raw: dict) -> None:
        raw = self._migrate_payload(raw)
        self.loaded_version = int(raw.get("version", 1))
        self.work_func = raw.get("work_func", 0.0)
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
        self.browse_only = bool(raw.get("browse_only", False))
        # Absent on pre-feature sessions -> empty (current_sample fallback).
        self.sample_configs = {
            str(k): SampleConfig.from_dict(v or {}).to_dict()
            for k, v in (raw.get("sample_configs", {}) or {}).items()
        }
        # v2->v3: absent field -> empty dict (all conventions data-driven).
        self.convention_registry = dict(raw.get("convention_registry", {}) or {})
        self.files = {}
        for name, edict in raw.get("files", {}).items():
            fp = FitParams(**_known_kwargs(FitParams, edict.get("fit_params", {})))
            mt = FileMeta(**_known_kwargs(FileMeta, edict.get("meta", {})))
            mt.sample_config = SampleConfig.from_meta(mt).to_dict()
            entry = FileEntry(
                ef_offset=edict.get("ef_offset", DEFAULT_EF_OFFSET_EV),
                edcnorm=edict.get("edcnorm", False),
                view_mode=edict.get("view_mode", "Raw"),
                fit_params=fp,
                fit_result=edict.get("fit_result"),
                meta=mt,
                fs_center_kx=edict.get("fs_center_kx"),
                fs_center_ky=edict.get("fs_center_ky"),
                fs_rotation_deg=float(edict.get("fs_rotation_deg", 0.0) or 0.0),
                fs_v0=float(edict.get("fs_v0", 12.0) or 12.0),
                fs_kz_plane=str(edict.get("fs_kz_plane", "Auto") or "Auto"),
                fs_phi_c_deg=float(edict.get("fs_phi_c_deg", 0.0) or 0.0),
                fs_bz_crystal_visible=bool(edict.get("fs_bz_crystal_visible", False)),
                fs_hs_crystal_visible=bool(edict.get("fs_hs_crystal_visible", False)),
                fs_bz_crystal_force_override=bool(edict.get("fs_bz_crystal_force_override", False)),
                fs_lattice=edict.get("fs_lattice", {}) or {},
                fs_bz_label_overrides=edict.get("fs_bz_label_overrides", {}) or {},
                fs_bz_label_preset=str(edict.get("fs_bz_label_preset", "") or ""),
                propagate_distortion_to_fs=bool(edict.get("propagate_distortion_to_fs", False)),
                grid_correction=edict.get("grid_correction", {}) or {},
                ef_correction=edict.get("ef_correction", {}) or {},
                bm_distortion=edict.get("bm_distortion", {}) or {},
                theory_overlay=edict.get("theory_overlay", {}) or {},
                band_analysis=edict.get("band_analysis", {}) or {},
                fs_pockets=list(edict.get("fs_pockets", []) or []),
                dft_grid_path=str(edict.get("dft_grid_path", "") or ""),
                fit_zones=normalize_fit_zones(edict.get("fit_zones", [])),
                active_zone_id=edict.get("active_zone_id"),
                annotations=edict.get("annotations", {}) or {},
                meta_gamma_state=dict(edict.get("meta_gamma_state", {}) or {}),
                parent_fs_path=edict.get("parent_fs_path"),
                processing_history=list(edict.get("processing_history", []) or []),
                curvature_dispersion=edict.get("curvature_dispersion"),
            )
            # One-time upgrade: legacy fits stored the MDC width as FWHM; modern
            # fits store HWHM (tagged width_convention). Normalize loaded fits to
            # HWHM so every consumer (Results, Γ(E), Im Σ) reads one convention.
            from arpes.physics.fit import migrate_fit_result_to_hwhm
            migrate_fit_result_to_hwhm(entry.fit_result)
            for _z in entry.fit_zones:
                if isinstance(_z, dict):
                    migrate_fit_result_to_hwhm(_z.get("fit_result"))
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
        """Stable session key: path relative to the root folder if possible."""
        p = Path(path)
        if self.folder is not None:
            try:
                return str(p.resolve().relative_to(self.folder.resolve()))
            except Exception:
                pass
        return p.name
