"""Géométrie CLS — lecture rapide P/T/azi sans charger les données.

Extrait de `arpes_explorer.py`. PyQt-free, testable sans UI.

Le CLS écrit ses positions de manipulateur dans un fichier `*_param.txt`
adjacent au cube de données ; chaque ligne JSON expose ``d.<MOTOR>.position``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_MOTOR_KEYS = (("P", "polar"), ("T", "tilt"))


def manipulator_from_param(path: str | Path) -> dict:
    """Lit P/T depuis le `*_param.txt` adjacent au fichier ou dans le dossier.

    Retourne ``{}`` si aucun fichier param trouvable ou exploitable.
    """
    p = Path(path)
    if p.is_file():
        param_files = [p.parent / f"{p.name}_param.txt"]
    elif p.is_dir():
        param_files = sorted(p.glob("*_param.txt"))
    else:
        return {}
    for param_file in param_files:
        if not param_file.exists():
            continue
        try:
            for line in param_file.read_text(errors="replace").splitlines():
                if not line.strip().startswith("{"):
                    continue
                motors = json.loads(line).get("d", {})
                out: dict = {}
                for motor, key in _MOTOR_KEYS:
                    value = motors.get(motor, {}).get("position")
                    if value is not None:
                        out[key] = float(value)
                if out:
                    return out
        except Exception:
            continue
    return {}


def geometry_for_path(
    path: str | Path,
    *,
    entry_meta=None,
    logbook_record: dict | None = None,
    logbook_mapping: dict | None = None,
    cell_float=None,
) -> dict:
    """Retourne la meilleure géométrie CLS connue pour ``path``.

    Priorité (du plus fiable au plus douteux) :
      1. ``_param.txt`` pour P/T (positions motorisées réelles) ;
      2. champs de l'entrée de session (``entry_meta``) pour P/T/azi/hv ;
      3. ligne de logbook (``logbook_record`` + ``logbook_mapping``) en
         remplissage des champs encore manquants — surtout utile pour ``azi``.

    ``cell_float`` est injecté pour parser les cellules logbook (typiquement
    ``arpes_logbook._cell_float``). Si absent, le fallback logbook est ignoré.
    """
    geom = manipulator_from_param(path)

    if entry_meta is not None:
        if geom.get("polar") is None and getattr(entry_meta, "polar", None) is not None:
            geom["polar"] = float(entry_meta.polar)
        if geom.get("tilt") is None and getattr(entry_meta, "tilt", None) is not None:
            geom["tilt"] = float(entry_meta.tilt)
        if getattr(entry_meta, "azi", None) is not None:
            geom["azi"] = float(entry_meta.azi)
        if getattr(entry_meta, "hv", None):
            geom["hv"] = float(entry_meta.hv)

    if logbook_record is not None and cell_float is not None:
        mapping = logbook_mapping or {}
        for key in ("polar", "tilt", "azi", "hv"):
            col = mapping.get(key, "")
            value = cell_float(logbook_record.get(col)) if col else None
            if value is not None and np.isfinite(value) and geom.get(key) is None:
                geom[key] = float(value)

    return geom
