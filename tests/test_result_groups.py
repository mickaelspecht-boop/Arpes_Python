from __future__ import annotations

import unittest

from arpes.analysis import result_groups as rg
from arpes.core.session import (
    FileEntry,
    FileMeta,
    Session,
    normalize_groups,
)


def _session() -> Session:
    s = Session()
    s.files["Ba122Co_C01/BM1"] = FileEntry(
        fit_result={"n_pairs": 1}, meta=FileMeta(polarization="LH", direction="G-M"))
    s.files["Ba122Co_C01/BM7"] = FileEntry(
        fit_result={"n_pairs": 1}, meta=FileMeta(polarization="LV", direction="G-M"))
    s.files["Ba122Cu_C13/BM2"] = FileEntry(
        fit_result={"n_pairs": 1}, meta=FileMeta(polarization="LH", direction="G-X"))
    return s


class TestManualGroups(unittest.TestCase):
    def test_add_is_idempotent_by_name(self):
        s = Session()
        g1 = rg.add_group(s, "alpha")
        g2 = rg.add_group(s, "ALPHA")
        self.assertIs(g1, g2)
        self.assertEqual(rg.group_names(s), ["alpha"])

    def test_assign_is_exclusive(self):
        s = _session()
        rg.add_group(s, "A")
        rg.add_group(s, "B")
        rg.assign_to_group(s, "A", ["Ba122Co_C01/BM1", "Ba122Co_C01/BM7"])
        rg.assign_to_group(s, "B", ["Ba122Co_C01/BM7"])  # steal from A
        self.assertEqual(rg.find_group(s, "A")["members"], ["Ba122Co_C01/BM1"])
        self.assertEqual(rg.find_group(s, "B")["members"], ["Ba122Co_C01/BM7"])
        self.assertEqual(rg.group_of_file(s, "Ba122Co_C01/BM7")["name"], "B")

    def test_rename_rejects_clash_and_empty(self):
        s = Session()
        rg.add_group(s, "A")
        rg.add_group(s, "B")
        self.assertFalse(rg.rename_group(s, "A", "B"))
        self.assertFalse(rg.rename_group(s, "A", "   "))
        self.assertTrue(rg.rename_group(s, "A", "Alpha"))
        self.assertEqual(rg.group_names(s), ["Alpha", "B"])

    def test_remove_and_unassign(self):
        s = _session()
        rg.add_group(s, "A")
        rg.assign_to_group(s, "A", ["Ba122Cu_C13/BM2"])
        rg.unassign(s, ["Ba122Cu_C13/BM2"])
        self.assertEqual(rg.find_group(s, "A")["members"], [])
        rg.remove_group(s, "A")
        self.assertEqual(rg.group_names(s), [])

    def test_prune_drops_missing_members(self):
        s = _session()
        rg.add_group(s, "A")
        rg.assign_to_group(s, "A", ["Ba122Cu_C13/BM2", "ghost/BM"])
        rg.prune_groups(s, list(s.files))
        self.assertEqual(rg.find_group(s, "A")["members"], ["Ba122Cu_C13/BM2"])


class TestAutoGrouping(unittest.TestCase):
    def test_by_compound(self):
        s = _session()
        grouped = rg.grouped_files(s, list(s.files), rg.GROUP_BY_COMPOUND)
        labels = [g[0] for g in grouped]
        self.assertEqual(labels, ["Co_C01", "Cu_C13"])
        self.assertEqual(len(grouped[0][1]), 2)

    def test_by_polarisation(self):
        s = _session()
        grouped = dict(rg.grouped_files(s, list(s.files), rg.GROUP_BY_POLARISATION))
        self.assertEqual(set(grouped["LH"]),
                         {"Ba122Co_C01/BM1", "Ba122Cu_C13/BM2"})
        self.assertEqual(grouped["LV"], ["Ba122Co_C01/BM7"])

    def test_none_is_flat(self):
        s = _session()
        grouped = rg.grouped_files(s, list(s.files), rg.GROUP_BY_NONE)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0][0], "")
        self.assertEqual(len(grouped[0][1]), 3)

    def test_manual_buckets_plus_ungrouped(self):
        s = _session()
        rg.add_group(s, "A")
        rg.assign_to_group(s, "A", ["Ba122Co_C01/BM1"])
        grouped = rg.grouped_files(s, list(s.files), rg.GROUP_BY_MANUAL)
        self.assertEqual(grouped[0][0], "A")
        self.assertEqual(grouped[0][1], ["Ba122Co_C01/BM1"])
        self.assertEqual(grouped[-1][0], rg.UNGROUPED)
        self.assertEqual(len(grouped[-1][1]), 2)

    def test_color_index_manual_marks_ungrouped_negative(self):
        s = _session()
        rg.add_group(s, "A", color_idx=3)
        rg.assign_to_group(s, "A", ["Ba122Co_C01/BM1"])
        cmap = rg.file_color_index(s, list(s.files), rg.GROUP_BY_MANUAL)
        self.assertEqual(cmap["Ba122Co_C01/BM1"], 3)
        self.assertEqual(cmap["Ba122Cu_C13/BM2"], -1)


class TestGroupsPersistence(unittest.TestCase):
    def test_roundtrip_through_payload(self):
        s = _session()
        rg.add_group(s, "alpha LV", color_idx=2)
        rg.assign_to_group(s, "alpha LV", ["Ba122Co_C01/BM7"])
        payload = s.to_payload()
        self.assertEqual(payload["version"], 6)
        s2 = Session()
        s2.load_from_payload(payload)
        self.assertEqual(rg.group_names(s2), ["alpha LV"])
        self.assertEqual(rg.find_group(s2, "alpha LV")["members"],
                         ["Ba122Co_C01/BM7"])
        self.assertEqual(rg.find_group(s2, "alpha LV")["color_idx"], 2)

    def test_normalize_dedupes_names_and_members(self):
        out = normalize_groups([
            {"name": "A", "members": ["f1", "f1", "f2"], "color_idx": 1},
            {"name": "a", "members": ["f3"]},          # dup name -> dropped
            {"name": "", "members": ["f4"]},           # empty -> dropped
            "garbage",
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["members"], ["f1", "f2"])


if __name__ == "__main__":
    unittest.main()
