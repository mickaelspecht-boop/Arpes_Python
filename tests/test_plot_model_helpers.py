import numpy as np

from arpes.ui.controllers.plot_model_helpers import build_model_pairs


def test_preview_uses_pair_initials_instead_of_detected_peaks():
    k = np.linspace(-1.0, 1.0, 401)
    # Data peaks intentionally disagree with requested initials.
    mdc = np.exp(-((k - 0.72) / 0.04) ** 2) + np.exp(-((k + 0.72) / 0.04) ** 2)
    pairs, _ = build_model_pairs(
        k, mdc, n_pairs=1, gamma_init=0.08,
        k_min=-0.9, k_max=0.9, center_init=0.10, smooth_sigma=2.0,
        pair_params=[{"kF_init": 0.25, "gamma_init": 0.06, "gamma_max": 0.2}],
    )
    _, km, kp, _, _ = pairs[0]
    assert km == 0.10 - 0.25
    assert kp == 0.10 + 0.25


def test_preview_uses_each_pairs_own_gamma():
    k = np.linspace(-1.0, 1.0, 801)
    mdc = np.ones_like(k)
    pairs, _ = build_model_pairs(
        k, mdc, n_pairs=2, gamma_init=0.08,
        k_min=-0.9, k_max=0.9, center_init=0.0, smooth_sigma=2.0,
        pair_params=[
            {"kF_init": 0.20, "gamma_init": 0.03, "gamma_max": 0.1},
            {"kF_init": 0.55, "gamma_init": 0.15, "gamma_max": 0.3},
        ],
    )
    curve_narrow = pairs[0][0]
    curve_wide = pairs[1][0]
    # Wide pair retains more intensity one HWHM-scale away from its peak.
    i1 = int(np.argmin(abs(k - (0.20 + 0.10))))
    i2 = int(np.argmin(abs(k - (0.55 + 0.10))))
    assert curve_wide[i2] > curve_narrow[i1]
