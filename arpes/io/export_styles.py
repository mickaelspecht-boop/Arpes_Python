"""Presets matplotlib pour exports de figures."""
from __future__ import annotations

from contextlib import contextmanager
import shutil
from typing import Iterator

import matplotlib.pyplot as plt


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
}


@contextmanager
def apply_preset(name: str) -> Iterator[None]:
    """Applique un preset matplotlib dans un contexte rcParams."""
    preset = dict(PRESETS.get(name, PRESETS["default"]))
    if preset.get("text.usetex") and shutil.which("latex") is None:
        preset["text.usetex"] = False
    with plt.rc_context(preset):
        yield


def savefig_with_preset(fig, path: str, preset_name: str, **kwargs) -> None:
    """Sauve une figure avec preset, fallback sans LaTeX si rendu impossible."""
    with apply_preset(preset_name):
        try:
            fig.savefig(path, dpi=300, **kwargs)
        except RuntimeError:
            if not PRESETS.get(preset_name, {}).get("text.usetex"):
                raise
            safe = dict(PRESETS[preset_name])
            safe["text.usetex"] = False
            with plt.rc_context(safe):
                fig.savefig(path, dpi=300, **kwargs)
