"""Phase-2 FS axis window: crop + renormalize + cache-key stability."""
from dataclasses import replace

import numpy as np

from arpes.physics.fs import FSParams, _fs_cache_key, extract_fs_map


def _raw(ny=8, nx=12, ne=5):
    rng = np.random.default_rng(1)
    fs_data = rng.random((ny, nx, ne))
    meta = {
        "fs_data": fs_data,
        "fs_kx": np.linspace(-2.0, 2.0, nx),
        "fs_ky": np.linspace(-2.0, 2.0, ny),
        "fs_energy": np.linspace(-0.05, 0.05, ne),
    }
    return {"metadata": meta}


def _base_params():
    return FSParams(normalize_profile=False, smooth_sigma=0.0)


def test_window_crops_both_axes():
    raw = _raw()
    p = _base_params()
    kx, ky, fs, _ = extract_fs_map(raw, p)
    pw = replace(p, kx_min=-1.0, kx_max=1.0, ky_min=-1.0, ky_max=1.0)
    kx2, ky2, fs2, _ = extract_fs_map(raw, pw)
    assert kx2.size < kx.size and ky2.size < ky.size
    assert kx2.min() >= -1.0 - 1e-9 and kx2.max() <= 1.0 + 1e-9
    assert ky2.min() >= -1.0 - 1e-9 and ky2.max() <= 1.0 + 1e-9
    assert fs2.shape == (ky2.size, kx2.size)


def test_window_renormalizes_on_crop():
    raw = _raw()
    pw = replace(_base_params(), kx_min=-1.0, kx_max=1.0)
    _, _, fs2, _ = extract_fs_map(raw, pw)
    finite = fs2[np.isfinite(fs2)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0 + 1e-9


def test_window_respects_center():
    raw = _raw()
    # window [-1,1] around center +1.0 selects native kx in [0, 2].
    pw = replace(_base_params(), kx_center=1.0, kx_min=-1.0, kx_max=1.0)
    kx2, _, _, _ = extract_fs_map(raw, pw)
    assert kx2.min() >= -1e-9 and kx2.max() <= 2.0 + 1e-9


def test_unset_window_is_noop_and_cache_key_stable():
    raw = _raw()
    p = _base_params()  # all window bounds NaN
    kx0, ky0, _, _ = extract_fs_map(raw, p)
    assert kx0.size == 12 and ky0.size == 8
    # NaN window must not break cache-key self-equality.
    assert _fs_cache_key(raw, p) == _fs_cache_key(raw, p)


def test_window_changes_cache_key():
    raw = _raw()
    p = _base_params()
    pw = replace(p, kx_min=-1.0, kx_max=1.0)
    assert _fs_cache_key(raw, p) != _fs_cache_key(raw, pw)


def test_degenerate_window_does_not_collapse():
    raw = _raw()
    # window far outside the data -> fewer than 2 points survive -> no crop.
    pw = replace(_base_params(), kx_min=100.0, kx_max=200.0)
    kx2, _, _, _ = extract_fs_map(raw, pw)
    assert kx2.size == 12
