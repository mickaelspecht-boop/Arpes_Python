"""Tests P2.6a — convention signe angle data-driven + registre freezable."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arpes.physics.angle_convention import (
    UNCALIBRATED,
    BeamlineAngleConvention,
    convention_key,
    get_convention,
    freeze_convention,
    filter_candidates,
    select_best_candidate,
    evaluate_confidence,
)
from arpes.core.session import Session


def _cands():
    return [
        {"candidate": "theta0", "theta0_deg": 2.0},
        {"candidate": "-theta0", "theta0_deg": -2.0},
        {"candidate": "azi_plus", "theta0_deg": 1.5},
    ]


class TestConventionKey(unittest.TestCase):
    def test_buckets_round_close_values_together(self):
        k1 = convention_key("CLS", 100.0, 0.0, 0.0)
        k2 = convention_key("CLS", 102.0, 1.0, 0.5)  # hv±5, azi±5, polar±2
        self.assertEqual(k1, k2)

    def test_distinct_geometry_distinct_key(self):
        k1 = convention_key("CLS", 100.0, 0.0, 0.0)
        k2 = convention_key("CLS", 100.0, 30.0, 0.0)
        self.assertNotEqual(k1, k2)

    def test_beamline_separates_keys(self):
        self.assertNotEqual(
            convention_key("CLS", 100.0, 0.0, 0.0),
            convention_key("BESSY", 100.0, 0.0, 0.0),
        )


class TestRegistry(unittest.TestCase):
    def test_empty_registry_is_uncalibrated(self):
        self.assertIsNone(get_convention({}, convention_key("CLS", 100, 0, 0)))

    def test_data_driven_entry_not_frozen(self):
        reg = {}
        key = convention_key("CLS", 100, 0, 0)
        reg[key] = BeamlineAngleConvention("CLS", source="data_driven").to_dict()
        self.assertIsNone(get_convention(reg, key))

    def test_freeze_requires_frozen_source(self):
        with self.assertRaises(ValueError):
            freeze_convention({}, "k", BeamlineAngleConvention("CLS", source="data_driven"))

    def test_freeze_and_retrieve(self):
        reg = {}
        key = convention_key("CLS", 100, 0, 0)
        conv = BeamlineAngleConvention("CLS", theta_sign=-1, source="manual")
        freeze_convention(reg, key, conv)
        got = get_convention(reg, key)
        self.assertIsNotNone(got)
        self.assertEqual(got.theta_sign, -1)


class TestFilterCandidates(unittest.TestCase):
    def test_uncalibrated_keeps_all(self):
        out = filter_candidates(_cands(), {}, beamline="CLS", hv=100, azi=0, polar=0)
        self.assertEqual(len(out), 3)

    def test_frozen_negative_keeps_only_negative_sign(self):
        reg = {}
        key = convention_key("CLS", 100, 0, 0)
        freeze_convention(reg, key, BeamlineAngleConvention("CLS", theta_sign=-1, source="manual"))
        out = filter_candidates(_cands(), reg, beamline="CLS", hv=100, azi=0, polar=0)
        labels = [c["candidate"] for c in out]
        self.assertEqual(labels, ["-theta0"])


class TestSelectBestCandidate(unittest.TestCase):
    def test_returns_min_score(self):
        scores = {"theta0": 0.5, "-theta0": 0.1, "azi_plus": 0.9}
        sel = select_best_candidate(_cands(), lambda c: scores[c["candidate"]])
        self.assertEqual(sel["best"]["candidate"], "-theta0")
        self.assertAlmostEqual(sel["best_score"], 0.1)

    def test_confidence_high_when_clear_winner(self):
        scores = {"theta0": 0.5, "-theta0": 0.1, "azi_plus": 0.9}
        sel = select_best_candidate(_cands(), lambda c: scores[c["candidate"]])
        self.assertGreater(sel["confidence"], 0.20)
        self.assertFalse(sel["tie"])

    def test_tie_detected_for_opposite_sign_near_equal(self):
        scores = {"theta0": 0.2314, "-theta0": 0.2317, "azi_plus": 0.9}
        sel = select_best_candidate(_cands(), lambda c: scores[c["candidate"]])
        self.assertTrue(sel["tie"])
        self.assertLess(sel["confidence"], 0.05)

    def test_single_candidate_no_ambiguity(self):
        sel = select_best_candidate([{"candidate": "theta0"}], lambda c: 0.3)
        self.assertEqual(sel["confidence"], float("inf"))
        self.assertFalse(sel["tie"])


class TestEvaluateConfidence(unittest.TestCase):
    def test_clear_winner_not_ambiguous(self):
        v = evaluate_confidence(confidence=0.4, gamma_best=0.3, mad_best=0.02,
                                gamma_residual_after=0.01, tie=False)
        self.assertFalse(v["ambiguous"])
        self.assertFalse(v["refuse"])

    def test_off_gamma_residual_flags_ambiguous(self):
        # redteam CASE1: residual Γ ≠ 0 after offset → off-Γ pocket.
        v = evaluate_confidence(confidence=0.4, gamma_best=0.3, mad_best=0.02,
                                gamma_residual_after=-0.15, tie=False)
        self.assertTrue(v["ambiguous"])
        self.assertTrue(any("residual" in r for r in v["reasons"]))

    def test_low_confidence_large_gamma_refuses(self):
        v = evaluate_confidence(confidence=0.01, gamma_best=0.3, mad_best=0.02,
                                gamma_residual_after=0.0, tie=True)
        self.assertTrue(v["refuse"])

    def test_native_gamma_flags_ambiguous(self):
        v = evaluate_confidence(confidence=0.4, gamma_best=0.02, mad_best=0.01,
                                gamma_residual_after=0.0, tie=False)
        self.assertTrue(v["ambiguous"])


class TestSessionRegistryRoundTrip(unittest.TestCase):
    def test_convention_registry_persists_and_bumps_version(self):
        self.assertEqual(Session.VERSION, 6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s = Session(root)
            key = convention_key("CLS", 100, 0, 0)
            freeze_convention(
                s.convention_registry, key,
                BeamlineAngleConvention("CLS", theta_sign=-1, source="manual"),
            )
            s.save()
            restored = Session(root)
            restored.load(root / ".arpes_session.json")
            got = get_convention(restored.convention_registry, key)
            self.assertIsNotNone(got)
            self.assertEqual(got.theta_sign, -1)

    def test_legacy_v2_session_loads_empty_registry(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / ".arpes_session.json"
            path.write_text(json.dumps({"version": 2, "files": {}}))
            s = Session(root)
            s.load(path)
            self.assertEqual(s.convention_registry, {})


if __name__ == "__main__":
    unittest.main()
