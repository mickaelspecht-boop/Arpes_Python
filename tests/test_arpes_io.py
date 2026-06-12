import unittest
from pathlib import Path
import importlib.util
import os

import numpy as np

from arpes.io.loaders import (
    ARPESData,
    ARPESDataValidationError,
    assert_arpes_data_valid,
    detect_format,
    detect_scan_kind,
    load_bessy_ses_ibw,
    load_arpes,
    register_loader,
    registered_loaders,
)


def _real_data_path(*parts: str) -> Path | None:
    """Return a private test-data path when ARPES_TEST_DATA is configured."""
    root = os.environ.get("ARPES_TEST_DATA")
    if not root:
        return None
    return Path(root).joinpath(*parts)


class TestARPESIOValidation(unittest.TestCase):
    def assert_common_output_convention(self, ds: ARPESData):
        self.assertIs(assert_arpes_data_valid(ds), ds)
        self.assertEqual(ds.data.ndim, 2)
        self.assertEqual(ds.energy.ndim, 1)
        self.assertIsNotNone(ds.kx)
        self.assertEqual(ds.data.shape, (len(ds.kx), len(ds.energy)))
        self.assertTrue(np.any(np.isfinite(ds.data)))
        self.assertIn("loader_label", ds.metadata)
        self.assertIn("lab", ds.metadata)
        self.assertIn("energy_axis", ds.metadata)

        fs_data = ds.metadata.get("fs_data")
        if fs_data is not None:
            self.assertEqual(
                np.asarray(fs_data).shape,
                (
                    len(ds.metadata["fs_ky"]),
                    len(ds.metadata["fs_kx"]),
                    len(ds.metadata["fs_energy"]),
                ),
            )

    def test_valid_bandmap_convention(self):
        ds = ARPESData(
            data=np.ones((4, 6), dtype=float),
            energy=np.linspace(-0.2, 0.1, 6),
            kx=np.linspace(-0.5, 0.5, 4),
            hv=21.2,
            source_format="unit",
            metadata={"x_axis_unit": "pi/a"},
        )

        self.assertIs(assert_arpes_data_valid(ds), ds)

    def test_rejects_transposed_bandmap_shape(self):
        ds = ARPESData(
            data=np.ones((6, 4), dtype=float),
            energy=np.linspace(-0.2, 0.1, 6),
            kx=np.linspace(-0.5, 0.5, 4),
            hv=21.2,
            source_format="unit",
        )

        with self.assertRaises(ARPESDataValidationError):
            assert_arpes_data_valid(ds)

    def test_valid_fs_metadata_convention(self):
        ds = ARPESData(
            data=np.ones((3, 5), dtype=float),
            energy=np.linspace(-0.3, 0.0, 5),
            kx=np.linspace(-0.4, 0.4, 3),
            ky=np.linspace(-0.2, 0.2, 2),
            hv=90.0,
            source_format="unit_fs",
            metadata={
                "fs_data": np.ones((2, 3, 5), dtype=float),
                "fs_ky": np.linspace(-0.2, 0.2, 2),
                "fs_kx": np.linspace(-0.4, 0.4, 3),
                "fs_energy": np.linspace(-0.3, 0.0, 5),
            },
        )

        self.assertIs(assert_arpes_data_valid(ds), ds)

    def test_rejects_transposed_fs_metadata_shape(self):
        ds = ARPESData(
            data=np.ones((3, 5), dtype=float),
            energy=np.linspace(-0.3, 0.0, 5),
            kx=np.linspace(-0.4, 0.4, 3),
            ky=np.linspace(-0.2, 0.2, 2),
            hv=90.0,
            source_format="unit_fs",
            metadata={
                "fs_data": np.ones((3, 2, 5), dtype=float),
                "fs_ky": np.linspace(-0.2, 0.2, 2),
                "fs_kx": np.linspace(-0.4, 0.4, 3),
                "fs_energy": np.linspace(-0.3, 0.0, 5),
            },
        )

        with self.assertRaises(ARPESDataValidationError):
            assert_arpes_data_valid(ds)

    def test_detect_format_uses_registered_loaders(self):
        self.assertEqual(detect_format(Path("sample.ibw")), "solaris_da30")
        self.assertIn("cls_txt", registered_loaders())
        self.assertIn("bessy_ses_ibw", registered_loaders())

    def test_custom_loader_registry_path(self):
        name = "unit_dummy_loader"

        if name not in registered_loaders():
            register_loader(
                name,
                lambda p: p.suffix == ".dummyarpes",
                lambda path, **kwargs: ARPESData(
                    data=np.ones((2, 3), dtype=float),
                    energy=np.array([-0.2, -0.1, 0.0]),
                    kx=np.array([-0.1, 0.1]),
                    hv=kwargs.get("hv") or 21.2,
                    path=Path(path),
                    source_format=name,
                    metadata={},
                ),
                "Unit-test dummy loader",
            )

        path = Path("anything.dummyarpes")
        self.assertEqual(detect_format(path), name)
        ds = load_arpes(path, work_func=4.5, hv=21.2)
        self.assertEqual(ds.source_format, name)
        self.assertEqual(ds.data.shape, (2, 3))

    def test_bessy_fixture_if_available(self):
        path = _real_data_path("Ba122", "Ba1220009w_Band Map B122_009.ibw")
        if path is None or not path.exists():
            self.skipTest("BESSY BM fixture unavailable; set ARPES_TEST_DATA")

        self.assertEqual(detect_format(path), "bessy_ses_ibw")
        self.assertEqual(detect_scan_kind(path), "BM")
        ds = load_bessy_ses_ibw(path, work_func=4.031, hv=47.031)
        self.assert_common_output_convention(ds)
        self.assertEqual(ds.source_format, "bessy_ses_ibw")
        self.assertEqual(ds.metadata["loader_label"], "BESSY")
        self.assertEqual(ds.data.shape, (319, 518))
        self.assertEqual(ds.metadata["scan_kind"], "BM")
        self.assertAlmostEqual(float(ds.energy[0]), -0.4, places=6)
        self.assertEqual(ds.metadata["energy_reference"], "ses_center_energy")
        self.assertEqual(ds.metadata["bessy_energy_reference_mode"], "ses_center_energy")
        self.assertEqual(ds.metadata["bessy_energy_reference_requested"], "auto")

        ds_hv = load_bessy_ses_ibw(
            path,
            work_func=4.031,
            hv=47.031,
            bessy_energy_reference="hv_minus_work_function",
        )
        self.assertEqual(ds_hv.metadata["energy_reference"], "hv_minus_work_function")
        self.assertAlmostEqual(float(ds_hv.energy[0]), 0.0, places=6)
        self.assertEqual(ds_hv.metadata["hv_policy"], "used_for_EF")

    def test_bessy_fs_fixture_if_available(self):
        path = _real_data_path("Ba122", "Ba1220003w_Ba122_003.ibw")
        if path is None or not path.exists():
            self.skipTest("BESSY FS fixture unavailable; set ARPES_TEST_DATA")

        self.assertEqual(detect_format(path), "bessy_ses_ibw")
        self.assertEqual(detect_scan_kind(path), "FS")
        ds = load_bessy_ses_ibw(path, work_func=4.031, hv=64.665)
        self.assert_common_output_convention(ds)
        meta = ds.metadata
        self.assertEqual(ds.source_format, "bessy_ses_ibw")
        self.assertEqual(meta["scan_kind"], "FS")
        self.assertEqual(meta["energy_reference"], "ses_center_energy")
        self.assertEqual(meta["ky_conversion"], "p_axis_scan_minus_scan_center_minus_tilt0")
        self.assertIn("fs_ky_angle_center_deg", meta)
        self.assertEqual(meta["fs_data"].shape, (len(meta["fs_ky"]), len(meta["fs_kx"]), len(meta["fs_energy"])))

    def test_real_loader_outputs_share_internal_convention_if_available(self):
        cases = [
            (
                "bessy_bm",
                _real_data_path("Ba122", "Ba1220009w_Band Map B122_009.ibw"),
                {"work_func": 4.031, "hv": 47.031},
                "bessy_ses_ibw",
                "BM",
            ),
            (
                "bessy_fs",
                _real_data_path("Ba122", "Ba1220003w_Ba122_003.ibw"),
                {"work_func": 4.031, "hv": 64.665},
                "bessy_ses_ibw",
                "FS",
            ),
            (
                "cls_bm",
                _real_data_path("Ba122_C05_2", "BM1"),
                {"work_func": 4.031, "hv": 60.0},
                "cls_txt",
                "BM",
            ),
            (
                "cls_fs",
                _real_data_path("Ba122_C05_2", "FS1"),
                {"work_func": 4.031, "hv": 60.0},
                "cls_txt",
                "FS",
            ),
        ]

        ran = 0
        for name, path, kwargs, fmt, kind in cases:
            if path is None or not path.exists():
                continue
            with self.subTest(name=name):
                self.assertEqual(detect_format(path), fmt)
                self.assertEqual(detect_scan_kind(path), kind)
                ds = load_arpes(path, **kwargs)
                self.assert_common_output_convention(ds)
                self.assertEqual(ds.source_format, fmt)
                self.assertEqual(ds.metadata["scan_kind"], kind)
            ran += 1
        if ran == 0:
            self.skipTest("no real BESSY/CLS fixture available; set ARPES_TEST_DATA")

    def test_solaris_fixture_uses_common_loader_if_erlab_available(self):
        if importlib.util.find_spec("erlab") is None:
            self.skipTest("erlab missing: Solaris/DA30 loader not executable in this environment")
        path = _real_data_path("BaNi2As2_", "BaNi2As2_0012.pxt")
        if path is None or not path.exists():
            self.skipTest("Solaris/DA30 fixture unavailable; set ARPES_TEST_DATA")

        self.assertEqual(detect_format(path), "solaris_da30")
        ds = load_arpes(path, work_func=4.031)
        self.assert_common_output_convention(ds)
        self.assertEqual(ds.source_format, "solaris_da30")


if __name__ == "__main__":
    unittest.main()
