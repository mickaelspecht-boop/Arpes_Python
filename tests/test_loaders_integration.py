"""Integration tests on real ARPES files.

Goal: pin the `load_arpes` output contract for each supported format, using
real files present under `~/Documents/Stage_M2/Code/...`. Each test is skipped
cleanly if the reference file is missing (different machine or directory tree),
so there are no false failures.

To extend: add an entry in `FIXTURES` below. The tested invariants are
intentionally minimal and stable (axes, dims, hv, format); calibration details
(EF, kx) are left to unit tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from arpes.io.loaders import load_arpes, detect_format


# Data root (resolved dynamically, never hard-coded in an assertion).
DATA_ROOT = Path.home() / "Documents" / "Stage_M2" / "Code"


# Each entry: (relative path from DATA_ROOT, expected format, dims, kwargs).
# The kwargs reproduce what the app passes to the loader for this file type.
FIXTURES = [
    {
        "label": "Solaris BM .pxt (BaNi2As2_0015)",
        "path": "BaNi2As2_/BaNi2As2_0015.pxt",
        "format": "solaris_da30",
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
    },
    {
        "label": "Solaris BM .ibw (BaNi2As2_0015 fixed cut)",
        "path": "BaNi2As2_/BaNi2As2_0015fixed cut.ibw",
        "format": "solaris_da30",
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
    },
    {
        "label": "Solaris FS .zip (BaNi2As2_0001)",
        "path": "BaNi2As2_/BaNi2As2_0001.zip",
        "format": "solaris_da30",
        "ndim": 2,  # 2D data (FS volume lives in metadata['fs_data']).
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0},
        "expect_fs": True,
    },
    {
        "label": "BESSY/SES .ibw (Ba1220009w)",
        "path": "Ba122/Ba1220009w_Band Map B122_009.ibw",
        "format": "bessy_ses_ibw",
        "ndim": 2,
        "kwargs": {"work_func": 4.031, "ef_offset": 0.0, "hv": 100.0},
    },
    {
        "label": "ALLS SpecsLab ITX FS (Ba122 alignment grid)",
        "path": "Data/Nouveau loader/ALLS/BaNi2As2/Ba122_FS_alignment10_grid_12_Map1.itx",
        "format": "alls_itx",
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0, "a_lattice": 4.0},
        "expect_fs": True,
    },
    {
        "label": "CLS texte (BM1)",
        "path": "Ba122_C05_2/BM1",
        "format": None,  # CLS format is not in the current registry.
        "ndim": 2,
        "kwargs": {"work_func": 4.5, "ef_offset": 0.0, "hv": 100.0},
        "skip_load": True,  # only detect_format is tested.
    },
]


def _resolve(fix: dict) -> Path | None:
    p = DATA_ROOT / fix["path"]
    return p if p.exists() else None


class TestLoadersIntegration(unittest.TestCase):
    """Loads each reference file and checks the minimum invariants."""

    def _check_common_invariants(self, ds, fix: dict) -> None:
        # Energy axes.
        self.assertEqual(ds.energy.ndim, 1, f"{fix['label']}: energy must be 1D")
        self.assertGreater(len(ds.energy), 1, f"{fix['label']}: energy axis too short")
        self.assertTrue(np.all(np.isfinite(ds.energy)), f"{fix['label']}: energy contains NaN/Inf")
        # Data.
        self.assertEqual(
            ds.data.ndim, fix["ndim"],
            f"{fix['label']}: data.ndim={ds.data.ndim} expected {fix['ndim']}",
        )
        self.assertTrue(np.any(np.isfinite(ds.data)), f"{fix['label']}: data all NaN")
        # Strictly positive hv (anti-regression for the Solaris hv≤0 fix).
        if ds.hv is not None:
            self.assertGreater(
                float(ds.hv), 0,
                f"{fix['label']}: hv={ds.hv} must be > 0 (loader must reject 0)",
            )
        # k axis.
        if ds.kx is not None:
            self.assertEqual(ds.kx.ndim, 1, f"{fix['label']}: kx must be 1D")
            self.assertEqual(
                ds.data.shape[0], len(ds.kx),
                f"{fix['label']}: data.shape[0]={ds.data.shape[0]} != len(kx)={len(ds.kx)}",
            )
        # FS volume if expected.
        if fix.get("expect_fs"):
            fs = ds.metadata.get("fs_data")
            self.assertIsNotNone(fs, f"{fix['label']}: fs_data missing from metadata")
            self.assertEqual(np.asarray(fs).ndim, 3, f"{fix['label']}: fs_data must be 3D")
        # Metadata contract.
        self.assertIn("loader_label", ds.metadata)
        self.assertIn("lab", ds.metadata)
        self.assertIn("energy_axis", ds.metadata)
        self.assertIn("pass_energy_eV", ds.metadata)
        self.assertGreater(
            float(ds.metadata["pass_energy_eV"]), 0,
            f"{fix['label']}: pass_energy_eV must be present and > 0",
        )


def _make_test(fix: dict):
    def test(self):
        path = _resolve(fix)
        if path is None:
            self.skipTest(f"Fixture missing: {fix['path']} (different machine/tree)")
        # 1) detect_format if expected.
        if fix.get("format") is not None:
            fmt = detect_format(path)
            self.assertEqual(
                fmt, fix["format"],
                f"{fix['label']}: detect_format={fmt!r} expected {fix['format']!r}",
            )
        # 2) Loading (skip if the CLS loader is not in the registry).
        if fix.get("skip_load"):
            self.skipTest(f"{fix['label']}: loading not covered (loader outside registry)")
        try:
            ds = load_arpes(str(path), **fix["kwargs"])
        except ImportError as e:
            self.skipTest(f"{fix['label']}: missing dependency ({e})")
        self._check_common_invariants(ds, fix)
    test.__name__ = "test_" + fix["label"].lower().replace(" ", "_").replace("/", "_").replace(".", "_")
    test.__doc__ = f"Loads and checks: {fix['label']}"
    return test


# Dynamically graft tests onto the class.
for _fix in FIXTURES:
    setattr(TestLoadersIntegration, _make_test(_fix).__name__, _make_test(_fix))


if __name__ == "__main__":
    unittest.main()
