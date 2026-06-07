#!/usr/bin/env python3
"""Audit metadata/logbook/pairing on real Stage_M2 Data."""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Data"
sys.path.insert(0, str(ROOT))

from arpes.core.session import FileEntry, FileMeta  # noqa: E402
from arpes.io.file_pairing import PairingCriteria, group_files_by_fs  # noqa: E402
from arpes.io.loaders.cls import load_cls_txt, _parse_cls_param  # noqa: E402
from arpes.io.logbook import LogbookManager, _cell_float  # noqa: E402
from arpes.io.logbook_io import read_logbook, scan_xlsx_for_scoped_logbooks  # noqa: E402


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(DATA))
    except Exception:
        return str(path)


def _find_cls_candidates(folder: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or path.name.endswith("_param.txt"):
            continue
        if path.is_file() and (path.parent / f"{path.name}_param.txt").exists():
            out.append(path)
        elif path.is_dir() and any(path.glob("*_Cycle_*_Step_*.txt")):
            out.append(path)
    return out


def _scan_kind(path: Path) -> str:
    if path.is_dir() and any(path.glob("*_Cycle_*_Step_*.txt")):
        return "FS"
    if path.is_file() and (path.parent / f"{path.name}_param.txt").exists():
        return "BM"
    if "kz" in path.name.lower():
        return "KZ"
    if "band map" in path.name.lower() or re.search(r"\bbm\d+", path.name.lower()):
        return "BM"
    return "unknown"


def _entry_from_values(values, path: Path) -> FileEntry:
    return FileEntry(
        meta=FileMeta(
            hv=float(values.hv or 0.0),
            temperature=float(values.temperature or 0.0),
            direction=values.direction,
            polarization=values.polarization,
            azi=values.azi,
            polar=values.polar,
            tilt=values.tilt,
            formula=values.formula,
            mp_id=values.mp_id,
            crystal_a_angstrom=float(values.crystal_a_angstrom or 0.0),
            scan_kind=_scan_kind(path),
        )
    )


def audit_ba122_c05_2() -> dict:
    folder = DATA / "Ba122_C05_2"
    logbook = folder / "Measurement CLS 2024 (1).xlsx"
    result = read_logbook(logbook, sheet_selector=lambda sheets: "C05_2" if "C05_2" in sheets else sheets[0])
    manager = LogbookManager(result.records, result.mapping, folder)
    files = {}
    value_rows = []
    for path in _find_cls_candidates(folder):
        values = manager.values_for_path(path)
        if values.has_any():
            files[str(path)] = _entry_from_values(values, path)
        value_rows.append({
            "path": _rel(path),
            "kind": _scan_kind(path),
            "has_logbook": values.has_any(),
            "hv": values.hv,
            "azi": values.azi,
            "polar": values.polar,
            "tilt": values.tilt,
            "pol": values.polarization,
        })
    tree, orphans = group_files_by_fs(files, PairingCriteria(folder_depth=1))

    load_checks = []
    for path in [folder / "BM1", folder / "FS1"]:
        values = manager.values_for_path(path)
        data = load_cls_txt(
            path,
            work_func=4.5,
            a_lattice=3.96,
            hv=values.hv,
            temperature=values.temperature,
            azi=values.azi or 0.0,
            pol=values.polarization,
        )
        md = data.metadata
        parsed = _parse_cls_param(path.parent if path.is_file() else path, path.name if path.is_file() else "FS")
        load_checks.append({
            "path": _rel(path),
            "scan_kind": md.get("scan_kind"),
            "hv": data.hv,
            "azi": md.get("azi"),
            "pol": md.get("pol"),
            "tilt_meta": md.get("tilt_ref"),
            "tilt_param": parsed.get("tilt_ref"),
            "polar_meta": md.get("polar"),
            "polar_raw_motor": md.get("polar_raw_motor"),
            "fs_ky_points": None if md.get("fs_ky") is None else len(md.get("fs_ky")),
        })

    return {
        "logbook": _rel(logbook),
        "mapping": result.mapping,
        "records": len(result.records),
        "values": value_rows,
        "pairing": [
            {"fs": _rel(Path(fs)), "bms": [_rel(Path(m.path)) for m in matches]}
            for fs, _entry, matches in tree
        ],
        "orphans": [_rel(Path(p)) for p, _e in orphans],
        "load_checks": load_checks,
    }


def audit_cls2026() -> dict:
    folder = DATA / "BaNi2As2-CLS2026"
    xlsx = folder / "CLS_2026_Exp_INFO.xlsx"
    subfolders = [str(p.relative_to(folder)) for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")]
    import pandas as pd
    scoped = scan_xlsx_for_scoped_logbooks(pd, xlsx, subfolders)
    out = {"logbook": _rel(xlsx), "scoped_count": len(scoped), "scopes": []}
    for item in scoped:
        rel = item["subfolder_rel"]
        sub = folder / rel
        records = item["df"].where(pd.notnull(item["df"]), None).to_dict(orient="records")
        manager = LogbookManager(records, item["mapping"], sub)
        files = {}
        rows = []
        for path in _find_cls_candidates(sub):
            values = manager.values_for_path(path)
            if values.has_any():
                files[str(path)] = _entry_from_values(values, path)
            rows.append({
                "path": _rel(path),
                "kind": _scan_kind(path),
                "has_logbook": values.has_any(),
                "hv": values.hv,
                "azi": values.azi,
                "polar": values.polar,
                "tilt": values.tilt,
                "pol": values.polarization,
            })
        tree, orphans = group_files_by_fs(files, PairingCriteria(folder_depth=1))
        out["scopes"].append({
            "sheet": item["sheet"],
            "folder_declared": item["folder_declared"],
            "subfolder": rel,
            "mapping": item["mapping"],
            "records": len(records),
            "matched_files": sum(1 for r in rows if r["has_logbook"]),
            "files": rows,
            "pairing": [
                {"fs": _rel(Path(fs)), "bms": [_rel(Path(m.path)) for m in matches]}
                for fs, _entry, matches in tree
            ],
            "orphans": [_rel(Path(p)) for p, _e in orphans],
        })
    return out


def audit_kz() -> dict:
    folder = DATA / "BaNi2As2_" / "kz_scan_1"
    logbook = folder / "___KZ_SCAN_1_FOLDER_LOGBOOK___.csv"
    result = read_logbook(logbook)
    manager = LogbookManager(result.records, result.mapping, folder)
    rows = []
    hv_seen = []
    for path in sorted(folder.glob("*kz_*")):
        if path.name.endswith("_LOGBOOK.txt") or path.name.startswith("___"):
            continue
        if path.suffix.lower() != ".ibw":
            continue
        values = manager.values_for_path(path)
        expected = _cell_float(path.name.split("kz_")[-1].split(".pxt")[0])
        if values.hv is not None:
            hv_seen.append(values.hv)
        rows.append({
            "path": _rel(path),
            "expected_hv_from_name": expected,
            "hv_logbook": values.hv,
            "ok": (
                values.hv is not None
                and expected is not None
                and math.isclose(float(values.hv), float(expected), abs_tol=0.02)
            ),
            "tilt": values.tilt,
            "polar": values.polar,
            "azi": values.azi,
        })
    return {
        "logbook": _rel(logbook),
        "mapping": result.mapping,
        "records": len(result.records),
        "files_checked": len(rows),
        "hv_minmax": [min(hv_seen), max(hv_seen)] if hv_seen else [None, None],
        "bad_hv": [r for r in rows if not r["ok"]],
        "sample": rows[:3] + rows[-3:],
    }


def main() -> int:
    report = {
        "data_root": str(DATA),
        "ba122_c05_2": audit_ba122_c05_2(),
        "cls2026": audit_cls2026(),
        "kz": audit_kz(),
    }
    out = ROOT / "AUDIT_REAL_DATA_METADATA.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "data_root": report["data_root"],
        "ba122_records": report["ba122_c05_2"]["records"],
        "ba122_pairing_groups": len(report["ba122_c05_2"]["pairing"]),
        "cls2026_scoped_count": report["cls2026"]["scoped_count"],
        "kz_files_checked": report["kz"]["files_checked"],
        "kz_bad_hv": len(report["kz"]["bad_hv"]),
        "report": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
