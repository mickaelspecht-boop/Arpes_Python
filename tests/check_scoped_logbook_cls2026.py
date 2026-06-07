"""Manual diagnostic (NOT a pytest test) — checks the scoped CLS2026 logbook.

Reproduces the "logbook scoped by subfolder" flow on real data:
  BNA_S1  <- feuille  CA041_S1  de CLS_2026_Exp_INFO.xlsx
  BNA_S2  <- feuille  CA046_S2

(BaNi2As2 = samples CA041 / CA046 in this logbook.)

For each BMxx file:
  - which logbook record is selected (scoped matching)
  - which hν / T / pol is extracted (with the sheet-specific mapping)
  - then loads the CLS file with this hν and prints the resulting E-EF range
    (+ possible Central Energy guard trigger).

Usage (in the env that has pandas/openpyxl, for example 'peaks'):
    python tests/check_scoped_logbook_cls2026.py [chemin/vers/BaNi2As2-CLS2026]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arpes.io.logbook import LogbookManager
from arpes.io.logbook_io import (
    excel_table_from_header,
    _looks_like_title,
    read_logbook,
)
from arpes.io.loaders.cls import load_cls_txt

DEFAULT_DATA = Path.home() / "Documents/Stage_M2/Code/Data/BaNi2As2-CLS2026"
SCOPES = [("BNA_S1", "CA041_S1"), ("BNA_S2", "CA046_S2")]


def _table_selector_factory():
    """Header-row selector: reuses the score used by the UI."""
    def score_row(raw, row_idx):
        try:
            df, m = excel_table_from_header(raw, row_idx)
        except Exception:
            return -10.0
        s = int(bool(m.get("file"))) * 3 + int(bool(m.get("hv"))) * 3
        s += int(bool(m.get("temperature"))) + int(bool(m.get("polarization")))
        s += int(bool(m.get("direction"))) + int(bool(m.get("azi")))
        s += int(bool(m.get("polar"))) + int(bool(m.get("tilt")))
        if _looks_like_title(m.get("file", "")):
            s -= 5
        if _looks_like_title(m.get("hv", "")):
            s -= 5
        return s

    def selector(raw, candidates):
        if not candidates:
            return None
        best = max(candidates, key=lambda r: score_row(raw, r))
        return excel_table_from_header(raw, best)

    return selector


def main(data_dir: Path) -> int:
    xlsx = data_dir / "CLS_2026_Exp_INFO.xlsx"
    if not xlsx.exists():
        print(f"introuvable : {xlsx}")
        return 1

    all_records: list[dict] = []
    scoped_mappings: dict[str, dict] = {}
    table_selector = _table_selector_factory()

    for subdir, sheet in SCOPES:
        print(f"\n=== {subdir}  <-  feuille '{sheet}' ===")
        res = read_logbook(
            xlsx,
            sheet_selector=lambda names, _s=sheet: _s if _s in names else (names[0] if names else ""),
            table_selector=table_selector,
            mapping_selector=lambda cols, m: m,  # no interaction
        )
        print(f"  sheet read  : {res.sheet_name}")
        print(f"  mapping     : {res.mapping}")
        print(f"  {len(res.records)} rows")
        for r in res.records:
            if isinstance(r, dict):
                r["_subfolder_rel"] = subdir
        all_records.extend(res.records)
        scoped_mappings[subdir] = res.mapping

    mgr = LogbookManager(all_records, {}, data_dir, scoped_mappings=scoped_mappings)

    for subdir, _sheet in SCOPES:
        sub = data_dir / subdir
        bm_files = sorted(p for p in sub.iterdir()
                          if p.is_file() and p.name.startswith("BM") and "_param" not in p.name)
        print(f"\n=== files {subdir} ===")
        for bm in bm_files:
            rec = mgr.find_record_for_path(bm)
            vals = mgr.values_for_path(bm)
            file_col = mgr._mapping_for_record(rec).get("file", "?") if rec else "?"
            file_val = rec.get(file_col) if rec else None
            print(f"\n  {bm.name}")
            print(f"    record found : {'yes' if rec else 'NO'}  (file col='{file_col}', value='{file_val}')")
            print(f"    hv={vals.hv}  T={vals.temperature}  pol={vals.polarization}  "
                  f"azi={vals.azi}  polar={vals.polar}  tilt={vals.tilt}")
            if vals.hv is None:
                print("    !! no hν extracted from the logbook")
                continue
            try:
                ds = load_cls_txt(bm, hv=float(vals.hv))
                e = ds.energy
                md = ds.metadata
                print(f"    loaded : E-EF [{float(e.min()):.3f}, {float(e.max()):.3f}] eV"
                      f"  refE={md.get('energy_reference')}"
                      f"  CentralEnergy={md.get('central_energy')}")
                if md.get("hv_warning"):
                    print(f"    hν guard : {md['hv_warning']}")
                lo, hi = float(e.min()), float(e.max())
                if not (-5.0 < lo < 5.0 and -5.0 < hi < 5.0):
                    print(f"    !! suspicious E-EF range (expected ~[-2, 2] eV for a map near EF)")
            except Exception as exc:
                print(f"    !! load failed : {exc}")
    return 0


if __name__ == "__main__":
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    raise SystemExit(main(data))
