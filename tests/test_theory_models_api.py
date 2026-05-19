from __future__ import annotations


def test_models_reexports_public_api():
    from arpes.theory import models

    for name in (
        "TheoryBandData",
        "TheoryOverlayConfig",
        "available_segments",
        "bandstructure_to_theory_data",
        "branch_display_names",
        "compare_fit_to_theory",
        "displayed_k_axis",
        "filter_bands_for_view",
        "normalize_direction_label",
        "parse_band_indices",
        "segment_from_direction",
        "select_bands_for_view",
        "selected_segment_mask",
    ):
        assert hasattr(models, name), name


def test_models_keeps_legacy_private_exports():
    from arpes.theory import models

    for name in (
        "_branch_index_for_segment",
        "_branch_local_k",
        "_clean_label",
        "_segment_mask",
    ):
        assert hasattr(models, name), name
