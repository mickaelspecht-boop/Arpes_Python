from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from arpes.io.loaders import detect_format, detect_scan_kind, load_arpes
from arpes.io.loaders.alls_itx import _parse_alls_itx_info


def _itx_text(dims=(3, 4, 2), include_z=True, z_label="a.u. (ShiftX)") -> str:
    values = " ".join(str(float(i)) for i in range(int(np.prod(dims))))
    wave_dims = ",".join(str(x) for x in dims)
    z_scale = f'X SetScale/I z, -1, 1, "{z_label}", \'Map1_1\'\n' if include_z else ""
    return (
        "IGOR\n"
        "X //Created by: SpecsLab Prodigy, Version 4.129.2-r127083\n"
        "X //Scan Mode         = Snapshot\n"
        "X //Analysis Mode     = UPS\n"
        "X //Lens Mode         = WideAngleMode_UPS\n"
        "X //Excitation Energy = 21.2182\n"
        "X //Kinetic Energy    = 36.3\n"
        "X //Binding Energy    = -15.0818\n"
        "X //Pass Energy       = 15\n"
        "X //WorkFunction      = 0\n"
        f"WAVES/S/N=({wave_dims}) 'Map1_1'\n"
        "BEGIN\n"
        f"{values}\n"
        "END\n"
        "X SetScale/I x, -2, 2, \"deg (theta_y)\", 'Map1_1'\n"
        "X SetScale/I y, 35, 38, \"eV (Kinetic Energy)\", 'Map1_1'\n"
        f"{z_scale}"
        "X SetScale/I d, 0, 1, \"cps (Intensity)\", 'Map1_1'\n"
    )


class TestAllsItxLoader(unittest.TestCase):
    def test_parse_header_dims_and_scales(self):
        info = _parse_alls_itx_info(_itx_text())
        self.assertEqual(info.dims, (3, 4, 2))
        self.assertEqual(info.wave_name, "Map1_1")
        self.assertEqual(info.header["Excitation Energy"], 21.2182)
        self.assertEqual(info.scales["x"].label, "deg (theta_y)")
        self.assertEqual(info.scales["y"].label, "eV (Kinetic Energy)")

    def test_detect_and_load_3d_as_fs_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.itx"
            path.write_text(_itx_text())

            self.assertEqual(detect_format(path), "alls_itx")
            self.assertEqual(detect_scan_kind(path), "FS")
            ds = load_arpes(path, work_func=4.5, ef_offset=0.0, a_lattice=4.0)

        self.assertEqual(ds.source_format, "alls_itx")
        self.assertEqual(ds.data.shape, (3, 4))
        self.assertEqual(ds.energy.shape, (4,))
        self.assertEqual(ds.kx.shape, (3,))
        self.assertEqual(np.asarray(ds.metadata["fs_data"]).shape, (2, 3, 4))
        self.assertEqual(ds.metadata["loader_label"], "ALLS")
        self.assertEqual(ds.metadata["scan_kind"], "FS")
        self.assertEqual(ds.metadata["igor_dims"], (3, 4, 2))
        self.assertEqual(ds.metadata["work_function_source"], "manual")
        self.assertIn("Third ITX axis", " ".join(ds.metadata["loader_warnings"]))
        self.assertGreater(ds.metadata["pass_energy_eV"], 0)

    def test_load_2d_bandmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.itx"
            path.write_text(_itx_text(dims=(3, 4), include_z=False))

            self.assertEqual(detect_scan_kind(path), "BM")
            ds = load_arpes(path, work_func=4.5, ef_offset=0.0, a_lattice=4.0)

        self.assertEqual(ds.data.shape, (3, 4))
        self.assertNotIn("fs_data", ds.metadata)

    def test_time_scan_without_theta_is_rejected_clearly(self):
        txt = _itx_text(dims=(3, 4, 2), z_label=" (Loop)").replace(
            'X SetScale/I x, -2, 2, "deg (theta_y)"',
            'X SetScale/I x, 763.5, 766.5, "ps (delay)"',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trm.itx"
            path.write_text(txt)

            self.assertEqual(detect_format(path), "alls_itx")
            self.assertEqual(detect_scan_kind(path), "unknown")
            with self.assertRaisesRegex(ValueError, "not analyzer theta"):
                load_arpes(path, work_func=4.5, ef_offset=0.0, a_lattice=4.0)

    def test_malformed_size_has_clear_error(self):
        bad = _itx_text(dims=(3, 4), include_z=False).replace(" 11.0", "")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.itx"
            path.write_text(bad)
            with self.assertRaisesRegex(ValueError, "data size mismatch"):
                load_arpes(path, work_func=4.5, ef_offset=0.0, a_lattice=4.0)


if __name__ == "__main__":
    unittest.main()
