"""Compatibility facade for DFT band overlay models/helpers.

The implementation lives in focused pure modules:
``data``, ``labels``, ``selection``, ``comparison`` and ``conversion``.
Existing imports from ``arpes.theory.models`` stay valid.
"""
from __future__ import annotations

from arpes.theory.comparison import compare_fit_to_theory, fit_mu_shift
from arpes.theory.conversion import (
    _branches_from_bandstructure,
    _k_distance_abs_from_bandstructure,
    _k_distance_from_bandstructure,
    _labels_from_bandstructure,
    _nearest_kpoint_index,
    _scaled_k_axis,
    _structure_elements,
    bandstructure_to_theory_data,
)
from arpes.theory.data import TheoryBandData, TheoryOverlayConfig, _finite_float
from arpes.theory.labels import (
    _branch_index_for_segment,
    _clean_label,
    _clean_segment_name,
    available_segments,
    branch_display_names,
    normalize_direction_label,
    segment_from_direction,
)
from arpes.theory.selection import (
    _branch_local_k,
    _segment_mask,
    displayed_k_axis,
    filter_bands_for_view,
    parse_band_indices,
    select_bands_for_view,
    selected_segment_mask,
)

__all__ = [
    "TheoryBandData",
    "TheoryOverlayConfig",
    "available_segments",
    "bandstructure_to_theory_data",
    "branch_display_names",
    "compare_fit_to_theory",
    "fit_mu_shift",
    "displayed_k_axis",
    "filter_bands_for_view",
    "normalize_direction_label",
    "parse_band_indices",
    "segment_from_direction",
    "select_bands_for_view",
    "selected_segment_mask",
]
