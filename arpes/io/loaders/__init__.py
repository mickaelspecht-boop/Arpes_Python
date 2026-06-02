#!/usr/bin/env python3
"""Couche IO ARPES — registre + dispatch + ré-exports publics.

Convention interne ARPES Explorer
=================================

Tous les loaders doivent retourner un :class:`ARPESData` conforme à cette
convention, indépendante du laboratoire source.

- Axe énergie : `energy` est toujours `E - EF` en eV, avec `0` à EF quand la
  calibration est connue. Les énergies cinétiques brutes restent dans
  `metadata` si elles sont utiles au diagnostic.
- Band map : `data` a toujours la shape `(n_k, n_E)`, donc `data[:, i]` est une
  MDC et `data[j, :]` est une EDC.
- FS / volume : quand un volume est disponible, il est stocké dans
  `metadata["fs_data"]` avec la shape `(n_ky, n_kx, n_E)`. Les axes associés
  sont `metadata["fs_ky"]`, `metadata["fs_kx"]`, `metadata["fs_energy"]`.
- Unités k : les axes `kx`/`ky` sont en `pi/a`. Les angles bruts restent dans
  `metadata` (`theta_par_deg`, `fs_ky_angle_deg`, etc.).
- Convention CLS/BESSY actuelle : `kx` est calculé depuis
  `theta_raw - static_polar - theta0_deg`. Sur les FS, une position moteur P
  qui correspond à l'axe scanné n'est pas réutilisée comme polar statique.
  `ky` vient de l'axe scanné (`tilt`/`P-Axis`) avec son recentrage propre.
  Les corrections cristallines `azi`/rotation de ZDB restent des métadonnées
  ou des corrections de visualisation tant qu'elles ne sont pas propagées par
  un modèle géométrique explicite.

Avant d'ajouter un nouveau loader, il doit passer `assert_arpes_data_valid()`.

Architecture du package
-----------------------
- `common.py` : modèles `ARPESData`/`LoaderSpec`, registre, validation,
  helpers numériques + métadonnées, dispatcher `load_arpes`.
- `solaris.py` : Solaris/DA30 via erlab.
- `bessy.py` : BESSY Scienta/SES R8000 (Igor v5).
- `cls.py` : CLS/LNLS texte (BM + FS Cycle/Step).

L'import du package déclenche la registration de chaque backend.
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
