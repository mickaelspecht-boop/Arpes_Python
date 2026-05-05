from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arpes_loader_orchestrator import LoaderOrchestrator
from arpes_session import FileEntry


def _label(source_format, metadata=None):
    return {"solaris_da30": "Solaris", "bessy_ses_ibw": "BESSY", "cls_txt": "CLS"}.get(source_format or "", "")


class TestLoaderOrchestrator(unittest.TestCase):
    def test_build_context_uses_entry_metadata(self):
        entry = FileEntry()
        entry.meta.temperature = 20.0
        entry.meta.azi = 12.0
        entry.meta.polarization = "LH"
        orch = LoaderOrchestrator(lambda *a, **k: None, _label)

        ctx = orch.build_context(
            entry,
            hv=48.0,
            angle_offsets={"theta0_deg": 0.1},
            bessy_energy_reference="ses_center_energy",
        )
        self.assertEqual(ctx.hv, 48.0)
        self.assertEqual(ctx.temperature, 20.0)
        self.assertEqual(ctx.azi, 12.0)
        self.assertEqual(ctx.pol, "LH")
        self.assertEqual(ctx.angle_offsets["theta0_deg"], 0.1)
        self.assertEqual(ctx.bessy_energy_reference, "ses_center_energy")

    def test_load_calls_loader_with_context(self):
        calls = []

        def fake_load(path, work_func, ef_offset, **kwargs):
            calls.append((path, work_func, ef_offset, kwargs))
            return {"hv": 50.0, "source_format": "solaris_da30", "metadata": {"temperature": 30.0}}

        entry = FileEntry()
        entry.meta.temperature = 20.0
        orch = LoaderOrchestrator(fake_load, _label)
        result = orch.load(
            "BM1",
            entry,
            work_func=4.5,
            ef_offset=0.0,
            hv=48.0,
            angle_offsets={},
            bessy_energy_reference="auto",
        )

        self.assertEqual(result.data["hv"], 50.0)
        self.assertEqual(calls[0][1], 4.5)
        self.assertEqual(calls[0][3]["hv"], 48.0)
        self.assertEqual(calls[0][3]["temperature"], 20.0)

    def test_apply_loaded_metadata_updates_entry(self):
        entry = FileEntry()
        orch = LoaderOrchestrator(lambda *a, **k: None, _label)
        md = orch.apply_loaded_metadata(
            {"source_format": "solaris_da30", "metadata": {"temperature": 30.0}},
            entry,
        )
        self.assertEqual(md["temperature"], 30.0)
        self.assertEqual(entry.meta.source_format, "solaris_da30")
        self.assertEqual(entry.meta.loader_label, "Solaris")
        self.assertEqual(entry.meta.temperature, 30.0)

    def test_hv_file_has_priority_after_load(self):
        entry = FileEntry()
        orch = LoaderOrchestrator(lambda *a, **k: None, _label)
        src = orch.resolve_hv_after_load(
            {"hv": 51.0},
            entry,
            hv_for_load=48.0,
            hv_from_logbook=True,
        )
        self.assertEqual(src.source, "file")
        self.assertEqual(src.value, 51.0)
        self.assertEqual(entry.meta.hv, 51.0)

    def test_hv_logbook_or_manual_when_file_missing(self):
        orch = LoaderOrchestrator(lambda *a, **k: None, _label)
        entry = FileEntry()
        src = orch.resolve_hv_after_load({}, entry, hv_for_load=48.0, hv_from_logbook=True)
        self.assertEqual(src.source, "logbook")
        self.assertEqual(entry.meta.hv, 48.0)
        entry = FileEntry()
        src = orch.resolve_hv_after_load({}, entry, hv_for_load=49.0, hv_from_logbook=False)
        self.assertEqual(src.source, "manual")
        self.assertEqual(entry.meta.hv, 49.0)

    def test_best_angle_loader_used_for_file_with_offsets(self):
        def fake_load(*args, **kwargs):
            raise AssertionError("regular loader should not be called")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "BM1.ibw"
            path.write_text("dummy")
            entry = FileEntry()
            orch = LoaderOrchestrator(fake_load, _label)

            def best(path_arg, entry_arg, hv_arg, offsets_arg):
                self.assertEqual(Path(path_arg), path)
                self.assertEqual(hv_arg, 48.0)
                return {"hv": 48.0, "metadata": {}}, {"theta0_deg": 0.2}

            result = orch.load(
                str(path),
                entry,
                work_func=4.5,
                ef_offset=0.0,
                hv=48.0,
                angle_offsets={"theta0_deg": 0.1},
                bessy_energy_reference="auto",
                best_angle_load_func=best,
            )
        self.assertEqual(result.angle_offsets["theta0_deg"], 0.2)


if __name__ == "__main__":
    unittest.main()
