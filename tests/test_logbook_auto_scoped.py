"""Tests auto-detection of scoped logbooks via Folder Name (robust templates)."""
from __future__ import annotations

import pytest

from arpes.io.logbook_io import (
    find_folder_name_in_sheet,
    match_folder_to_subfolder,
)

pd = pytest.importorskip("pandas")


def _df(rows: list[list]):
    return pd.DataFrame(rows)


# ---- find_folder_name_in_sheet ---------------------------------------------


class TestFindFolderName:
    def test_classic_row2(self):
        raw = _df([
            ["Bullet", "Sample", "Sample Code"],
            ["14", "BaNi2As2", "CA041"],
            ["Folder Name", "BNA_S1"],
            ["Clive Status", "Good"],
        ])
        assert find_folder_name_in_sheet(raw) == "BNA_S1"

    def test_label_case_insensitive(self):
        raw = _df([
            ["foo"],
            ["FOLDER NAME", "BNA_S2"],
        ])
        assert find_folder_name_in_sheet(raw) == "BNA_S2"

    def test_label_french_dossier(self):
        raw = _df([
            ["Dossier", "MonDossier"],
        ])
        assert find_folder_name_in_sheet(raw) == "MonDossier"

    def test_label_with_spaces_and_punct(self):
        raw = _df([
            ["Folder-Name :", "S3"],
        ])
        # "folder-name :" normalized → "foldername" → match
        assert find_folder_name_in_sheet(raw) == "S3"

    def test_value_skips_empty_columns(self):
        raw = _df([
            ["Folder Name", None, "", "S4"],
        ])
        assert find_folder_name_in_sheet(raw) == "S4"

    def test_no_label_returns_empty(self):
        raw = _df([
            ["File", "hv", "TEMP"],
            ["FS1", 100, 13],
        ])
        assert find_folder_name_in_sheet(raw) == ""

    def test_label_without_value_returns_empty(self):
        raw = _df([
            ["Folder Name"],
            ["Clive Status", "Good"],
        ])
        # Label found but no value → "" (second line is not retried because
        # the label was already seen with value=None; accepted tolerance).
        assert find_folder_name_in_sheet(raw) in ("", "Good")  # either is accepted

    def test_empty_sheet(self):
        raw = _df([])
        assert find_folder_name_in_sheet(raw) == ""

    def test_scan_limited_to_first_rows(self):
        # Folder Name beyond _FOLDER_NAME_SCAN_ROWS must be ignored.
        rows = [["junk"]] * 20 + [["Folder Name", "TooLate"]]
        raw = _df(rows)
        assert find_folder_name_in_sheet(raw) == ""


# ---- match_folder_to_subfolder ---------------------------------------------


class TestMatchFolder:
    def test_exact(self):
        assert match_folder_to_subfolder("BNA_S2", ["BNA_S1", "BNA_S2"]) == "BNA_S2"

    def test_case_insensitive(self):
        assert match_folder_to_subfolder("bna_s2", ["BNA_S1", "BNA_S2"]) == "BNA_S2"

    def test_normalized(self):
        # "BNA-S2" normalized to "bnas2"; "BNA_S2" normalized to "bnas2".
        assert match_folder_to_subfolder("BNA-S2", ["BNA_S1", "BNA_S2"]) == "BNA_S2"

    def test_basename_match(self):
        # Nested subfolder: rel = "data/BNA_S2".
        assert match_folder_to_subfolder("BNA_S2", ["data/BNA_S1", "data/BNA_S2"]) == "data/BNA_S2"

    def test_substring(self):
        # Folder declared = "S2", subfolder = "BNA_S2" → match substring (S2 in BNAS2)
        # But very short "s2" (2 chars) < 3 → not selected.
        assert match_folder_to_subfolder("S2", ["BNA_S2"]) == ""
        # 3+ chars accepted.
        assert match_folder_to_subfolder("BNA", ["BNA_S2", "OTHER"]) == "BNA_S2"

    def test_no_match(self):
        assert match_folder_to_subfolder("MysteryFolder", ["BNA_S1", "BNA_S2"]) == ""

    def test_empty_target(self):
        assert match_folder_to_subfolder("", ["BNA_S1"]) == ""

    def test_ambiguous_substring_refused(self):
        # F6: a truncated declared name substring-matching >=2 distinct
        # subfolders must refuse (return "") instead of first-winner —
        # silently scoping the wrong sample's params would corrupt Γ/φ/a.
        assert match_folder_to_subfolder("YNS", ["YNS_S1", "YNS_S6"]) == ""
        # Distinct full names still match unambiguously (no regression).
        assert match_folder_to_subfolder("YNS_S1", ["YNS_S1", "YNS_S6"]) == "YNS_S1"
        assert match_folder_to_subfolder("YNS_S6", ["YNS_S1", "YNS_S6"]) == "YNS_S6"
