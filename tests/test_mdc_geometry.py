import numpy as np
import pytest
from types import SimpleNamespace

from arpes.physics.fit import MdcFitter
from arpes.physics.mdc_geometry import (
    energy_window_plan,
    per_pair_values,
    relative_k0_guesses,
    symmetric_k0_ceiling,
    symmetric_peak_positions,
)
from arpes.ui.widgets.plots.mdc_fit import fit_mdc_peak_pairs


def test_peak_positions_follow_manual_center():
    assert symmetric_peak_positions(-0.17, 0.23) == pytest.approx((-0.40, 0.06))


def test_auto_peak_guesses_are_distances_not_absolute_positions():
    peaks = np.array([-0.40, 0.06, 0.31])
    assert relative_k0_guesses(peaks, -0.17, 2) == pytest.approx([0.23, 0.48])


def test_offcenter_ceiling_uses_nearest_window_edge():
    assert symmetric_k0_ceiling(-0.8, 0.8, -0.5) == pytest.approx(0.285)
    assert symmetric_k0_ceiling(-0.2, 0.8, -0.5) == 0.0


def test_per_pair_widths_preserved_and_extended():
    assert per_pair_values([0.03, 0.08], 3, 0.05) == pytest.approx(
        [0.03, 0.08, 0.08]
    )


@pytest.mark.parametrize("axis", [
    np.linspace(-0.3, 0.1, 81),
    np.linspace(0.1, -0.3, 81),
])
def test_energy_window_plan_matches_naive_masks(axis):
    targets = np.array([-0.221, -0.047, 0.083])
    for target, (index, selector) in zip(
        targets, energy_window_plan(axis, targets, 0.04)
    ):
        expected_index = int(np.argmin(np.abs(axis - target)))
        expected = np.abs(axis - axis[expected_index]) <= 0.02
        actual = np.zeros(axis.size, dtype=bool)
        actual[selector] = True
        assert index == expected_index
        assert np.array_equal(actual, expected)


def test_offcenter_fit_recovers_center_relative_k0():
    k = np.linspace(-0.65, 0.55, 241)
    e = np.array([-0.02])
    center, k0, gamma = -0.12, 0.24, 0.035
    left, right = symmetric_peak_positions(center, k0)
    mdc = (
        gamma**2 / ((k - left) ** 2 + gamma**2)
        + 0.8 * gamma**2 / ((k - right) ** 2 + gamma**2)
        + 0.03
    )
    result = fit_mdc_peak_pairs(
        mdc[:, None],
        k,
        e,
        n_pairs=1,
        ev_start=-0.02,
        ev_end=-0.02,
        smooth_fit=0.0,
        smooth_detect=0.0,
        gamma_init=[0.04],
        gamma_max=[0.12],
        kF_init=None,
        center_init=center,
        xg_range=0.04,
        min_amplitude=0.01,
        max_jump=0.2,
        k_min=-0.60,
        k_max=0.45,
    )
    assert result["k0"][0][0] == pytest.approx(k0, abs=0.01)
    assert result["xg"][0] == pytest.approx(center, abs=0.01)


def test_full_fit_receives_each_pairs_width_parameters():
    fp = SimpleNamespace(
        n_pairs=2, ev_start=-0.2, ev_end=-0.01,
        smooth_fit=1.0, smooth_detect=2.0,
        gamma_init=0.03, gamma_max=0.2,
        pairs=[
            {"kF_init": 0.2, "gamma_init": 0.025, "gamma_max": 0.10},
            {"kF_init": 0.5, "gamma_init": 0.080, "gamma_max": 0.30},
        ],
        center_init=-0.1, xg_range=0.05,
        min_amplitude=0.02, max_jump=0.1,
        mdc_energy_window=0.02, mdc_energy_step=0.005,
        scan_direction="down", width_mode="symmetric",
        k_min=-0.7, k_max=0.7, k0_max=None,
        dE_meV=15.0, dk_inv_a=0.005,
        shape="lorentzian", hold_center=False, hold_gamma=False,
    )
    kwargs = MdcFitter.fit_kwargs(fp)
    assert kwargs["gamma_init"] == pytest.approx([0.025, 0.080])
    assert kwargs["gamma_max"] == pytest.approx([0.10, 0.30])
    assert kwargs["kF_init"] == pytest.approx([0.2, 0.5])
