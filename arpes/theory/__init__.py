"""Optional theoretical-band overlays for ARPES Explorer.

This package is intentionally isolated: no experimental loader imports it, and
Materials Project dependencies are loaded only by ``materials_project``.
"""

from .models import (
    TheoryBandData,
    TheoryOverlayConfig,
    compare_fit_to_theory,
    filter_bands_for_view,
    normalize_direction_label,
    segment_from_direction,
)

__all__ = [
    "TheoryBandData",
    "TheoryOverlayConfig",
    "compare_fit_to_theory",
    "filter_bands_for_view",
    "normalize_direction_label",
    "segment_from_direction",
]
