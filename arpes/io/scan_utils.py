"""Filesystem heuristics for ARPES scan layouts. Pure Python, no PyQt.

A *scan dataset directory* is a folder that IS one measurement (e.g. a CLS
Fermi-surface scan stored as many ``*_Cycle_*_Step_*`` slice files next to a
``*_param.txt``). Such folders must never be mistaken for *sample* folders
when a parent directory is organised as one-subfolder-per-sample.
"""
from __future__ import annotations

from pathlib import Path

_DATA_SUFFIXES = {".pxt", ".ibw", ".zip", ".itx", ".pxp"}


def is_scan_dataset_dir(path: Path) -> bool:
    """True if `path` is a directory holding ONE scan (CLS slices layout)."""
    p = Path(path)
    if not p.is_dir():
        return False
    for param_file in p.glob("*_param.txt"):
        prefix = param_file.name.removesuffix("_param.txt")
        if any(p.glob(f"{prefix}_Cycle_*_Step_*.txt")):
            return True
    return False


def is_data_file(path: Path) -> bool:
    """True if `path` looks like a loadable ARPES data file."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.name.endswith("_param.txt"):
        return False
    if p.suffix.lower() in _DATA_SUFFIXES:
        return True
    if p.suffix != "":
        return False
    # CLS BM: extensionless file, normally with a sibling <name>_param.txt file.
    # Older copied folders can be incomplete but still need their sample row in
    # the setup popup.
    return (p.parent / f"{p.name}_param.txt").exists() or p.name.upper().startswith("BM")
