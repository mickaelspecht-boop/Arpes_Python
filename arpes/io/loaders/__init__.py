#!/usr/bin/env python3
"""ARPES IO layer - registry + dispatch + public re-exports.

ARPES Explorer Internal Convention
==================================

All loaders must return an :class:`ARPESData` compliant with this convention,
independent of the source laboratory.

- Energy axis: `energy` is always `E - EF` in eV, with `0` at EF when the
  calibration is known. Raw kinetic energies remain in `metadata` when useful
  for diagnostics.
- Band map: `data` always has shape `(n_k, n_E)`, so `data[:, i]` is an
  MDC and `data[j, :]` is an EDC.
- FS / volume: when a volume is available, it is stored in
  `metadata["fs_data"]` with shape `(n_ky, n_kx, n_E)`. Associated axes are
  `metadata["fs_ky"]`, `metadata["fs_kx"]`, `metadata["fs_energy"]`.
- k units: the `kx`/`ky` axes are in `pi/a`. Raw angles remain in `metadata`
  (`theta_par_deg`, `fs_ky_angle_deg`, etc.).
- Current CLS/BESSY convention: `kx` is calculated from
  `theta_raw - static_polar - theta0_deg`. On FS data, a P motor position that
  corresponds to the scanned axis is not reused as static polar.
  `ky` comes from the scanned axis (`tilt`/`P-Axis`) with its own recentering.
  Crystal corrections `azi`/ZDB rotation remain metadata or visualization
  corrections until they are propagated by an explicit geometric model.

Before adding a new loader, it must pass `assert_arpes_data_valid()`.

Package Architecture
--------------------
- `common.py`: `ARPESData`/`LoaderSpec` models, registry, validation, numeric
  helpers + metadata, `load_arpes` dispatcher.
- `solaris.py`: Solaris/DA30 through erlab.
- `bessy.py`: BESSY Scienta/SES R8000 (Igor v5).
- `cls.py`: CLS/LNLS text (BM + FS Cycle/Step).
- `alls_itx.py`: ALLS SpecsLab Prodigy Igor Text exports.

Importing the package triggers registration of each backend.
"""
from __future__ import annotations

from .common import (
    ARPESData,
    ARPESDataValidationError,
    LoaderSpec,
    SUPPORTED_SOLARIS_EXTENSIONS,
    _add_instrument_resolution_metadata,
    _add_loader_diagnostics,
    _append_unique_list,
    _cls_angle_to_k_pi_over_a,
    _first_present,
    _is_monotonic_axis,
    _loadtxt_float32,
    _LOADER_REGISTRY,
    _require_erlab,
    _set_da30_loader,
    _transpose_to_axes,
    _valid_float,
    _valid_positive_float,
    _validation_issue,
    assert_arpes_data_valid,
    detect_format,
    detect_scan_kind,
    load_arpes,
    load_arpes_file,
    loader_label,
    register_loader,
    registered_loaders,
    scan_axis_summary,
    static_polar_for_kx,
)
from .bessy import (
    _IBW5Info,
    _IBW5_BIN_HEADER_SIZE,
    _IBW5_WAVE_HEADER_SIZE,
    _is_bessy_ses_ibw,
    _load_ibw5_numeric,
    _parse_ses_note,
    _read_ibw5_info,
    _read_ibw5_note,
    load_bessy_ses_ibw,
)
from .solaris import (
    _is_solaris_da30_file,
    _load_solaris_from_registry,
    load_solaris_da30_bandmap,
)
from .alls_itx import (
    ITXInfo,
    ITXScale,
    _is_alls_itx_file,
    _load_alls_itx_array,
    _parse_alls_itx_info,
    _read_alls_itx_info,
    load_alls_itx,
)
from .cls import (
    _CLS_CACHE_VERSION,
    _CYCLE_STEP_RE,
    _cls_cycle_step,
    _cls_fs_cache_path,
    _cls_fs_signature,
    _is_cls_bm_file,
    _is_cls_fs_dir,
    _load_cls_fs_cache,
    _load_cls_fs_volume,
    _load_cls_from_registry,
    _parse_cls_param,
    _save_cls_fs_cache,
    load_cls_txt,
)
