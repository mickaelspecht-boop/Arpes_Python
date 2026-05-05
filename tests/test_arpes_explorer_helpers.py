import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import arpes_explorer
    from arpes import app as arpes_app
    from arpes.ui.widgets.browsers import files as _files_mod
    from arpes_explorer import (
        QApplication,
        FileBrowserPanel,
        FileEntry,
        FileMeta,
        Session,
        _format_direction_label,
        _infer_logbook_mapping,
    )
    ARPES_EXPLORER_HELPERS_AVAILABLE = True
except ImportError:
    ARPES_EXPLORER_HELPERS_AVAILABLE = False


@unittest.skipUnless(ARPES_EXPLORER_HELPERS_AVAILABLE, "arpes_explorer helpers require Qt bindings")
class TestArpesExplorerLogbookHelpers(unittest.TestCase):
    _qt_app = None

    def test_direction_column_aliases_include_gamma_and_zdb(self):
        columns = ["Scan", "Photon Energy", "High symmetry path", "Temp"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["direction"], "High symmetry path")

        columns = ["Num", "Energy", "ZDB", "Pol"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["direction"], "ZDB")

    def test_solaris_logbook_columns_do_not_confuse_temperature_with_angles(self):
        columns = ["Filename", "Photon Energy", "Sample Temperature", "Light Polarization", "High symmetry path"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["file"], "Filename")
        self.assertEqual(mapping["hv"], "Photon Energy")
        self.assertEqual(mapping["temperature"], "Sample Temperature")
        self.assertEqual(mapping["polarization"], "Light Polarization")
        self.assertEqual(mapping["direction"], "High symmetry path")
        self.assertEqual(mapping["polar"], "")
        self.assertEqual(mapping["tilt"], "")

        columns = ["Filename", "Photon Energy", "Polar", "Pol", "Scan path"]
        mapping = _infer_logbook_mapping(columns)
        self.assertEqual(mapping["polarization"], "Pol")
        self.assertEqual(mapping["polar"], "Polar")

    def test_direction_label_formats_gamma_variants(self):
        self.assertEqual(_format_direction_label("G"), "Γ")
        self.assertEqual(_format_direction_label("gamma"), "Γ")
        self.assertEqual(_format_direction_label("Gamma-X"), "Γ-X")
        self.assertEqual(_format_direction_label("G M"), "Γ M")
        self.assertEqual(_format_direction_label("GM"), "ΓM")

    @classmethod
    def _app(cls):
        cls._qt_app = QApplication.instance() or cls._qt_app or QApplication([])
        return cls._qt_app

    def test_file_browser_grouping_modes_and_logbook_priority(self):
        self._app()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm = root / "BM1"
            bm.write_text("dummy")
            (root / "BM1_param.txt").write_text("dummy")
            fs_dir = root / "FS1"
            fs_dir.mkdir()
            (fs_dir / "FS_param.txt").write_text("dummy")
            (fs_dir / "FS_Cycle_0_Step_0.txt").write_text("dummy")
            bessy = root / "scan_bessy.ibw"
            bessy.write_text("dummy")
            solaris = root / "scan_solaris.zip"
            solaris.write_text("dummy")

            session = Session(root)
            session.files[session.key_for_path(bm)] = FileEntry(
                meta=FileMeta(
                    hv=60.0,
                    temperature=18.0,
                    direction="G-M",
                    polarization="LH",
                    azi=12.0,
                    source_format="cls_txt",
                    loader_label="CLS",
                )
            )
            session.logbook_mapping = {
                "file": "file",
                "hv": "hv",
                "temperature": "T",
                "polarization": "Pol",
                "direction": "path",
                "azi": "azi",
            }
            session.logbook_records = [
                {"file": "BM1", "hv": "55", "T": "22", "Pol": "LV", "path": "Gamma-X", "azi": "9"},
                {"file": "scan_bessy.ibw", "hv": "47.0", "T": "12", "Pol": "LH", "path": "G", "azi": "3"},
                {"file": "scan_solaris.zip", "hv": "100", "T": "8", "Pol": "LV", "path": "ZDB", "azi": "0"},
            ]

            def fake_detect_format(path):
                name = Path(path).name
                if name.startswith("scan_bessy"):
                    return "bessy_ses_ibw"
                if name.startswith("scan_solaris"):
                    return "solaris_da30"
                if Path(path).name == "BM1" or Path(path).name == "FS1":
                    return "cls_txt"
                return "unknown"

            def fake_detect_scan_kind(path, format_hint=None):
                name = Path(path).name
                if name == "FS1" or name == "scan_solaris.zip":
                    return "FS"
                if name == "BM1" or name == "scan_bessy.ibw":
                    return "BM"
                return "unknown"

            with mock.patch.object(arpes_app, "detect_format", fake_detect_format), \
                 mock.patch.object(arpes_app, "detect_scan_kind", fake_detect_scan_kind), \
                 mock.patch.object(_files_mod, "detect_format", fake_detect_format), \
                 mock.patch.object(_files_mod, "detect_scan_kind", fake_detect_scan_kind):
                panel = FileBrowserPanel(session)
                panel.set_folder(root)

                discovered = {p.relative_to(root).as_posix() for p in panel._discover_items()}
                self.assertIn("BM1", discovered)
                self.assertIn("FS1", discovered)
                self.assertIn("scan_bessy.ibw", discovered)
                self.assertIn("scan_solaris.zip", discovered)
                self.assertNotIn("FS1/FS_Cycle_0_Step_0.txt", discovered)

                panel._group_fields = ["Type"]
                self.assertEqual(panel._group_key_for_path(bm), "BM")
                self.assertEqual(panel._group_key_for_path(fs_dir), "FS")
                self.assertEqual(panel._group_key_for_path(bessy), "BM")
                self.assertEqual(panel._group_key_for_path(solaris), "FS")

                panel._group_fields = ["Labo"]
                self.assertEqual(panel._group_key_for_path(bm), "CLS")
                self.assertEqual(panel._group_key_for_path(bessy), "BESSY")
                self.assertEqual(panel._group_key_for_path(solaris), "Solaris")

                panel._group_fields = ["hν"]
                self.assertEqual(panel._group_key_for_path(bm), "hν 60.0 eV (session)")
                self.assertEqual(panel._group_key_for_path(bessy), "hν 47.0 eV (logbook)")

                panel._group_fields = ["Chemin"]
                self.assertEqual(panel._group_key_for_path(bm), "Γ-M (session)")
                self.assertEqual(panel._group_key_for_path(bessy), "Γ (logbook)")

                panel._group_fields = ["hν", "Température", "Polarisation"]
                self.assertEqual(
                    panel._group_key_for_path(bm),
                    "hν 60.0 eV (session) / T 18.0 K (session) / Pol LH (session)",
                )
                self.assertEqual(
                    panel._group_key_for_path(bessy),
                    "hν 47.0 eV (logbook) / T 12.0 K (logbook) / Pol LH (logbook)",
                )


if __name__ == "__main__":
    unittest.main()
