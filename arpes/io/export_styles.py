"""Matplotlib presets for figure exports.

P4.1: force a light background for exports (screen canvases are dark).
P4.3: Nature/Science presets (normalized column widths, 7 pt sans-serif,
vector PDF by default) plus matplotlib ``metadata`` passthrough (PDF/EXIF).
"""
from __future__ import annotations

from contextlib import contextmanager
import shutil
from typing import Iterator

import matplotlib.pyplot as plt


# Normalized column widths (mm) - P4.3.
NATURE_WIDTHS_MM = {"single": 89.0, "double": 183.0}
SCIENCE_WIDTHS_MM = {"single": 55.0, "double": 121.0}

# Presets that force a light background (publication-ready figures).
LIGHT_BG_PRESETS = frozenset({
    "publication_npj", "publication_prb", "publication_nature", "publication_science",
})


def figure_size_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Convert dimensions from mm to inches for ``figsize``."""
    return (float(width_mm) / 25.4, float(height_mm) / 25.4)


PRESETS: dict[str, dict] = {
    "default": {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "text.usetex": False,
    },
    "publication_npj": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
    "publication_prb": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.75,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 0.9,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "text.usetex": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
    # Nature: sans-serif 7 pt, vector PDF by default (P4.3).
    "publication_nature": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.5,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.75,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.transparent": False,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
    "publication_science": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.5,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.75,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.transparent": False,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
}


@contextmanager
def apply_preset(name: str) -> Iterator[None]:
    """Apply a matplotlib preset inside an rcParams context."""
    preset = dict(PRESETS.get(name, PRESETS["default"]))
    if preset.get("text.usetex") and shutil.which("latex") is None:
        preset["text.usetex"] = False
    with plt.rc_context(preset):
        yield


def _recolor_light(fig) -> dict:
    """Recolor a dark figure onto a light background for export (P4.1).

    Return a snapshot of the original colors for restoration after save, so the
    on-screen canvas is not modified permanently.
    """
    snap: dict = {"fig": fig.get_facecolor(), "axes": []}
    fig.set_facecolor("white")
    for ax in fig.get_axes():
        ax_snap = {
            "face": ax.get_facecolor(),
            "title": ax.title.get_color(),
            "xlabel": ax.xaxis.label.get_color(),
            "ylabel": ax.yaxis.label.get_color(),
            "spines": {k: s.get_edgecolor() for k, s in ax.spines.items()},
            "tick_x": [t.get_color() for t in ax.get_xticklabels()],
            "tick_y": [t.get_color() for t in ax.get_yticklabels()],
        }
        snap["axes"].append((ax, ax_snap))
        ax.set_facecolor("white")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        for s in ax.spines.values():
            s.set_edgecolor("black")
        ax.tick_params(colors="black")
    return snap


def _restore(fig, snap: dict) -> None:
    """Restore the original colors after a light-background export."""
    fig.set_facecolor(snap["fig"])
    for ax, ax_snap in snap["axes"]:
        ax.set_facecolor(ax_snap["face"])
        ax.title.set_color(ax_snap["title"])
        ax.xaxis.label.set_color(ax_snap["xlabel"])
        ax.yaxis.label.set_color(ax_snap["ylabel"])
        for k, col in ax_snap["spines"].items():
            ax.spines[k].set_edgecolor(col)
        for t, col in zip(ax.get_xticklabels(), ax_snap["tick_x"]):
            t.set_color(col)
        for t, col in zip(ax.get_yticklabels(), ax_snap["tick_y"]):
            t.set_color(col)


def savefig_with_preset(
    fig,
    path: str,
    preset_name: str,
    *,
    light_background: bool | None = None,
    metadata: dict | None = None,
    **kwargs,
) -> None:
    """Save a figure with a preset, light background, and PDF/EXIF metadata.

    Args:
        light_background: force the light background. ``None`` enables it for
            publication presets (``LIGHT_BG_PRESETS``).
        metadata: dict passed to ``fig.savefig(metadata=...)`` (PDF/PNG).
    """
    if light_background is None:
        light_background = preset_name in LIGHT_BG_PRESETS
    if metadata is not None:
        kwargs["metadata"] = metadata

    def _save():
        snap = _recolor_light(fig) if light_background else None
        try:
            fig.savefig(path, dpi=300, **kwargs)
        finally:
            if snap is not None:
                _restore(fig, snap)

    with apply_preset(preset_name):
        try:
            _save()
        except RuntimeError:
            if not PRESETS.get(preset_name, {}).get("text.usetex"):
                raise
            safe = dict(PRESETS[preset_name])
            safe["text.usetex"] = False
            with plt.rc_context(safe):
                _save()
