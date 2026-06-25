"""Stable hashing and JSON-safe normalization for MDC fit state."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
import hashlib
import json

import numpy as np


def sanitize(obj):
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [sanitize(x) for x in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def compute_fit_params_hash(
    fp: Any,
    *,
    ef_offset: float = 0.0,
    view_mode: str = "",
    hv: float | None = None,
    bm_distortion: dict | None = None,
    grid_correction: dict | None = None,
    ef_correction: dict | None = None,
    ensemble_settings: dict | None = None,
) -> str:
    """Hash only state that can change fitted numerical results."""
    if is_dataclass(fp):
        fp_dict = asdict(fp)
    elif isinstance(fp, dict):
        fp_dict = dict(fp)
    else:
        fp_dict = {
            key: getattr(fp, key) for key in dir(fp)
            if not key.startswith("_") and not callable(getattr(fp, key))
        }
    pairs = fp_dict.get("pairs")
    if isinstance(pairs, list):
        fp_dict["pairs"] = [
            {
                key: value for key, value in dict(pair).items()
                if key not in {"label", "results_visible"}
            }
            if isinstance(pair, dict) else pair
            for pair in pairs
        ]
    payload = {
        "fp": sanitize(fp_dict),
        "ef_offset": float(ef_offset or 0.0),
        "view_mode": str(view_mode or ""),
        "hv": float(hv) if hv is not None else None,
        "bm_distortion": sanitize(bm_distortion or {}),
        "grid_correction": sanitize(grid_correction or {}),
        "ef_correction": sanitize(ef_correction or {}),
        "ensemble_settings": sanitize(ensemble_settings or {}),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
