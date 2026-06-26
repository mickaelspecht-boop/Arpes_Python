"""Headless tests for the print-style band-map export figure builder."""
import numpy as np

from arpes.ui.widgets.bm_export import build_bm_export_figure


def _disp(nk=30, ne=40):
    k = np.linspace(-1.0, 1.0, nk)
    e = np.linspace(-0.3, 0.05, ne)
    disp = np.random.default_rng(0).random((nk, ne))
    return disp, k, e


def test_bm_export_transparent_black_with_ef_line():
    disp, k, e = _disp()
    fig = build_bm_export_figure(
        disp, k, e, cmap="inferno", color_kwargs={"vmin": 0, "vmax": 1},
        title="BM test [Raw]",
    )
    assert fig.patch.get_alpha() == 0.0
    ax = fig.axes[0]
    assert ax.get_title() == "BM test [Raw]"
    assert ax.xaxis.label.get_color() in ("k", "black")
    # an EF line was added at E = 0
    ys = [round(float(ln.get_ydata()[0]), 6) for ln in ax.get_lines()]
    assert 0.0 in ys


def test_bm_export_with_gamma_norm():
    disp, k, e = _disp()
    fig = build_bm_export_figure(
        disp, k, e, cmap="RdBu_r", color_kwargs={"vmin": -1, "vmax": 1},
        gamma=0.5, title="",
    )
    assert fig.axes
