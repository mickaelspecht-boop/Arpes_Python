from __future__ import annotations

import numpy as np
import pytest

from arpes.theory.alignment import (
    alignment_warnings,
    apply_energy_transform,
    effective_mu_shift,
    effective_z_scale,
)
from arpes.theory.models import TheoryOverlayConfig


def test_apply_energy_transform_option_a():
    cfg = TheoryOverlayConfig(mu_shift=0.10, z_scale=0.5)
    out = apply_energy_transform([-0.3, 0.1, 0.5], cfg)
    np.testing.assert_allclose(out, [-0.20, 0.0, 0.20])


def test_legacy_energy_shift_migrates_to_mu_shift():
    cfg = TheoryOverlayConfig(energy_shift=0.2)
    assert effective_mu_shift(cfg) == pytest.approx(-0.2)
    np.testing.assert_allclose(apply_energy_transform([0.0, 1.0], cfg), [0.2, 1.2])


def test_config_roundtrip_writes_mu_and_legacy_shift():
    cfg = TheoryOverlayConfig(mu_shift=0.12, z_scale=0.7)
    data = cfg.to_dict()
    assert data["mu_shift"] == pytest.approx(0.12)
    assert data["energy_shift"] == pytest.approx(-0.12)
    assert data["z_scale"] == pytest.approx(0.7)
    again = TheoryOverlayConfig.from_dict(data)
    assert effective_mu_shift(again) == pytest.approx(0.12)
    assert effective_z_scale(again) == pytest.approx(0.7)


def test_alignment_warnings_for_large_values():
    warnings = alignment_warnings(0.35, 0.1)
    assert any("mu" in item for item in warnings)
    assert any("Z" in item for item in warnings)
