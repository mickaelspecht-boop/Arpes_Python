"""Headless tests for the print-style FS export figure builder."""
import numpy as np

from arpes.physics.fs import FSParams
from arpes.ui.widgets.fs_export import build_export_figure


def _ctx():
    x = np.linspace(-1.0, 1.0, 10)
    y = np.linspace(-1.0, 1.0, 12)
    xx, yy = np.meshgrid(x, y)
    fs = np.random.default_rng(0).random((12, 10))
    return {
        "x_plot": xx, "y_plot": yy, "fs": fs,
        "params": FSParams(), "fs_kind": "kxky", "title": "FS raw",
    }


def _identity(points):
    return np.asarray(points, dtype=float)


def test_export_is_transparent_with_black_axes_and_title():
    fig = build_export_figure(
        _ctx(), add_hsym=True, title="FS : C05_FS3.txt", to_plot=_identity
    )
    assert fig.patch.get_alpha() == 0.0
    ax = fig.axes[0]
    assert ax.get_title() == "FS : C05_FS3.txt"
    assert ax.xaxis.label.get_color() in ("k", "black")
    assert ax.get_aspect() in (1.0, "equal")


def test_export_without_hsym_still_builds():
    fig = build_export_figure(
        _ctx(), add_hsym=False, title="FS", to_plot=_identity
    )
    assert fig.axes


def test_export_non_kxky_uses_auto_aspect():
    ctx = _ctx()
    ctx["fs_kind"] = "tilt"
    fig = build_export_figure(ctx, add_hsym=True, title="FS", to_plot=_identity)
    # high-symmetry overlay is skipped when axes are not kx/ky in pi/a
    assert fig.axes[0].get_aspect() == "auto"
