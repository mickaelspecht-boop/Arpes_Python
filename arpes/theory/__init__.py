"""Optional theoretical-band overlays for ARPES Explorer.

This package is intentionally isolated: no experimental loader imports it, and
Materials Project dependencies are loaded only by ``materials_project``.
"""

from .band_select import (
    aggregate_projection_character,
    bands_crossing_ef,
    compute_band_meta,
    format_band_indices,
)
from .models import (
    TheoryBandData,
    TheoryOverlayConfig,
    compare_fit_to_theory,
    filter_bands_for_view,
    normalize_direction_label,
    segment_from_direction,
    select_bands_for_view,
)

__all__ = [
    "TheoryBandData",
    "TheoryOverlayConfig",
    "compare_fit_to_theory",
    "filter_bands_for_view",
    "select_bands_for_view",
    "normalize_direction_label",
    "segment_from_direction",
    "compute_band_meta",
    "bands_crossing_ef",
    "format_band_indices",
    "aggregate_projection_character",
]
