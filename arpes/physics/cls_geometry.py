"""CLS geometry: quick P/T/azi read without loading data.

Extracted from `arpes_explorer.py`. PyQt-free, testable without UI.

CLS writes its manipulator positions in a `*_param.txt` file adjacent to the
data cube; each JSON line exposes ``d.<MOTOR>.position``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_MOTOR_KEYS = (("P", "polar"), ("T", "tilt"))


def manipulator_from_param(path: str | Path) -> dict:
    """Read P/T from the `*_param.txt` adjacent to the file or in the folder.

    Returns ``{}`` if no parameter file can be found or used.
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
    """Return the best known CLS geometry for ``path``.

    Priority (from most reliable to most doubtful):
      1. ``_param.txt`` for P/T (actual motorized positions);
      2. session entry fields (``entry_meta``) for P/T/azi/hv;
      3. logbook row (``logbook_record`` + ``logbook_mapping``) to fill fields
         still missing, especially useful for ``azi``.

    ``cell_float`` is injected to parse logbook cells (typically
    ``arpes_logbook._cell_float``). If absent, the logbook fallback is ignored.
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
