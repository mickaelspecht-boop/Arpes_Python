"""Sample-folder layout detection for the folder-load setup dialog.

Pure Python (no PyQt). Answers: when the user opens a data folder, which
top-level subfolders are *samples* (deserving their own work function /
lattice parameters) and which are merely *scan datasets* (an FS stored as a
folder of slices) that belong to the parent sample?

Rules (validated by the council, redteam case "CLS FS dirs mistaken for
samples"):
- a top-level subfolder that IS a scan dataset dir → not a sample;
- a top-level subfolder containing data files (directly or nested) → sample
  candidate;
- if NO sample candidate remains, the whole folder is one sample (mode
  "single"); otherwise mode "multi" with one entry per candidate.
The user can always override the detected mode in the dialog.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from arpes.io.scan_utils import is_data_file, is_scan_dataset_dir

_MAX_SCANNED_ITEMS = 5000  # keeps huge beamtime trees responsive


@dataclass(frozen=True)
class SampleSubfolder:
    key: str          # top-level subfolder name == sample_configs key
    n_files: int      # loadable items found under it (capped scan)


@dataclass(frozen=True)
class SampleLayout:
    mode: str                                   # "single" | "multi"
    subfolders: tuple[SampleSubfolder, ...] = ()
    n_root_files: int = 0                       # loadable files directly at root


def _count_loadable(folder: Path, budget: int) -> int:
    """Loadable items under `folder` (scan dirs count as ONE item)."""
    n = 0
    scanned = 0
    for p in folder.rglob("*"):
        scanned += 1
        if scanned > budget:
            break
        if p.is_dir():
            if is_scan_dataset_dir(p):
                n += 1
        elif is_data_file(p):
            if any(is_scan_dataset_dir(parent) for parent in p.parents
                   if parent != folder and folder in parent.parents):
                continue  # slice inside a scan dataset, already counted
            n += 1
    return n


def detect_sample_layout(folder: Path | str) -> SampleLayout:
    """Classify the top-level subfolders of `folder` as samples or scans."""
    root = Path(folder)
    if not root.is_dir():
        return SampleLayout(mode="single")
    candidates: list[SampleSubfolder] = []
    n_root = 0
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith("."):
            continue
        if child.is_file():
            if is_data_file(child):
                n_root += 1
            continue
        if is_scan_dataset_dir(child):
            continue  # an FS scan folder, not a sample
        n = _count_loadable(child, _MAX_SCANNED_ITEMS)
        if n > 0:
            candidates.append(SampleSubfolder(key=child.name, n_files=n))
    if not candidates:
        return SampleLayout(mode="single", n_root_files=n_root)
    return SampleLayout(
        mode="multi",
        subfolders=tuple(candidates),
        n_root_files=n_root,
    )


def sample_key_for_entry_key(entry_key: str) -> str:
    """Sample key (top-level subfolder) for a session file key.

    Session keys are folder-relative paths ("BNO/scan_001.zip"). Returns ""
    for files at the root or for keys without a separator (absolute-path
    fallback keys degrade gracefully to the session-wide sample).
    """
    parts = Path(str(entry_key)).parts
    return parts[0] if len(parts) > 1 else ""
