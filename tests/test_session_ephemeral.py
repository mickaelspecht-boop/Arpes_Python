from __future__ import annotations

import unittest

from arpes.core.session import (
    FileEntry,
    Session,
    strip_ephemeral_fit_result,
)


def _heavy_fit_result() -> dict:
    # I_smoothed = full smoothed band-map image (the bloat); the rest is small.
    return {
        "n_pairs": 1,
        "e_fitted": [-0.1, -0.05, 0.0],
        "kF_plus": [[0.25, 0.24, 0.23]],
        "kF_minus": [[-0.25, -0.24, -0.23]],
        "I_smoothed": [[float(i + j) for j in range(50)] for i in range(50)],
    }


class TestEphemeralStripping(unittest.TestCase):
    def test_strip_returns_shallow_copy_without_key(self):
        fr = _heavy_fit_result()
        out = strip_ephemeral_fit_result(fr)
        self.assertNotIn("I_smoothed", out)
        self.assertIn("I_smoothed", fr)         # original untouched
        self.assertIs(out["kF_plus"], fr["kF_plus"])  # shallow (no array copy)

    def test_strip_in_place_mutates(self):
        fr = _heavy_fit_result()
        out = strip_ephemeral_fit_result(fr, in_place=True)
        self.assertIs(out, fr)
        self.assertNotIn("I_smoothed", fr)

    def test_to_payload_drops_I_smoothed_but_keeps_live_dict(self):
        s = Session()
        e = FileEntry(fit_result=_heavy_fit_result())
        s.files["f"] = e
        payload = s.to_payload()
        self.assertNotIn("I_smoothed", payload["files"]["f"]["fit_result"])
        # the live in-memory dict still has it for the current display
        self.assertIn("I_smoothed", e.fit_result)
        # real results survive
        self.assertEqual(payload["files"]["f"]["fit_result"]["n_pairs"], 1)

    def test_to_payload_strips_inside_fit_zones(self):
        s = Session()
        e = FileEntry(fit_result={"n_pairs": 1})
        e.fit_zones = [{
            "id": "z1", "label": "Z1", "color_idx": 0, "active": True,
            "fit_result": _heavy_fit_result(),
        }]
        s.files["f"] = e
        payload = s.to_payload()
        self.assertNotIn("I_smoothed",
                         payload["files"]["f"]["fit_zones"][0]["fit_result"])
        self.assertIn("I_smoothed", e.fit_zones[0]["fit_result"])  # live kept

    def test_load_strips_legacy_I_smoothed_from_ram(self):
        s = Session()
        s.files["f"] = FileEntry(fit_result={"n_pairs": 1})
        payload = s.to_payload()
        # simulate a legacy bloated session on disk
        payload["files"]["f"]["fit_result"]["I_smoothed"] = [[1.0, 2.0], [3.0, 4.0]]
        s2 = Session()
        s2.load_from_payload(payload)
        self.assertNotIn("I_smoothed", s2.files["f"].fit_result)
        self.assertEqual(s2.files["f"].fit_result["n_pairs"], 1)

    def test_roundtrip_size_drops(self):
        import json
        s = Session()
        s.files["f"] = FileEntry(fit_result=_heavy_fit_result())
        with_heavy = len(json.dumps(_heavy_fit_result()))
        persisted = len(json.dumps(s.to_payload()["files"]["f"]["fit_result"]))
        self.assertLess(persisted, with_heavy / 5)


class TestSaveScheduler(unittest.TestCase):
    def test_scheduler_defers_write_until_flush(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(folder=Path(tmp))
            s.files["f"] = FileEntry(fit_result={"n_pairs": 1})
            requests = []
            s.set_save_scheduler(lambda: requests.append(1))
            s.save(); s.save(); s.save()
            self.assertEqual(len(requests), 3)             # only requests
            self.assertFalse((Path(tmp) / ".arpes_session.json").exists())
            s.flush_save()                                  # real write
            self.assertTrue((Path(tmp) / ".arpes_session.json").exists())

    def test_no_scheduler_writes_immediately(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            s = Session(folder=Path(tmp))
            s.files["f"] = FileEntry()
            s.save()
            self.assertTrue(s.json_path.exists())

    def test_reset_keeps_scheduler(self):
        s = Session()
        cb = lambda: None
        s.set_save_scheduler(cb)
        s.reset(keep_folder=False)
        self.assertIs(s._save_scheduler, cb)


if __name__ == "__main__":
    unittest.main()
