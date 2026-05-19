"""Tests P1-P3 : sélection / caractère bandes DFT + rétrocompat schéma."""
import numpy as np
import pytest

from arpes.theory.band_select import (
    aggregate_projection_character,
    bands_crossing_ef,
    compute_band_meta,
    format_band_indices,
)
from arpes.theory.models import (
    TheoryBandData,
    TheoryOverlayConfig,
    parse_band_indices,
    select_bands_for_view,
)


class TestComputeBandMeta:
    def test_min_max_crosses(self):
        bands = [[-2.0, -1.0], [-0.3, 0.4], [1.0, 2.0]]
        meta = compute_band_meta(bands)
        assert meta[0]["crosses_ef"] is False
        assert meta[1]["crosses_ef"] is True
        assert meta[2]["crosses_ef"] is False
        assert meta[1]["e_min"] == pytest.approx(-0.3)
        assert meta[2]["e_max"] == pytest.approx(2.0)

    def test_window_widens_crossing(self):
        bands = [[0.5, 0.8]]  # ne touche pas 0 sans fenêtre
        assert compute_band_meta(bands)[0]["crosses_ef"] is False
        assert compute_band_meta(bands, ef_window=1.0)[0]["crosses_ef"] is True

    def test_all_nan_band_safe(self):
        meta = compute_band_meta([[np.nan, np.nan]])
        assert meta[0]["crosses_ef"] is False
        assert np.isnan(meta[0]["e_min"])


class TestBandsCrossingEf:
    def test_recompute_from_meta(self):
        meta = compute_band_meta([[-2, -1], [-0.1, 0.1], [0.5, 0.7]])
        assert bands_crossing_ef(meta, 0.0) == [1]
        assert bands_crossing_ef(meta, 0.6) == [1, 2]


class TestFormatBandIndices:
    def test_compress_runs(self):
        assert format_band_indices([1, 3, 5, 6, 7, 8]) == "1,3,5-8"

    def test_empty(self):
        assert format_band_indices([]) == ""

    def test_dedup_sort(self):
        assert format_band_indices([4, 2, 2, 3]) == "2-4"

    def test_roundtrip_with_parser(self):
        spec = "0,2,4-7,10"
        idx = parse_band_indices(spec, 50)
        assert format_band_indices(idx) == "0,2,4-7,10"
        assert parse_band_indices(format_band_indices(idx), 50) == idx


class TestAggregateProjectionCharacter:
    def test_empty_graceful(self):
        assert aggregate_projection_character(None) == []
        assert aggregate_projection_character({}) == []
        assert aggregate_projection_character(np.zeros((0,))) == []

    def test_ndarray_dominant_channel(self):
        # (n_band=2, n_k=3, n_orb=4, n_ion=2)
        arr = np.zeros((2, 3, 4, 2))
        arr[0, :, 2, 1] = 1.0  # bande0 -> orb d, ion1
        arr[1, :, 0, 0] = 1.0  # bande1 -> orb s, ion0
        out = aggregate_projection_character(arr, ["Ti", "O"])
        assert out[0] == "O-d"
        assert out[1] == "Ti-s"

    def test_spin_dict_summed(self):
        a = np.zeros((1, 2, 4, 1))
        a[0, :, 1, 0] = 1.0
        out = aggregate_projection_character({"up": a, "down": a}, ["Fe"])
        assert out[0] == "Fe-p"


class TestSelectBandsForView:
    def _data(self):
        return TheoryBandData(
            source="materials_project",
            material_id="mp-1",
            k_distance=[-1.0, 0.0, 1.0],
            bands=[[-2, -2, -2], [-0.2, 0.0, 0.2], [1.5, 1.5, 1.5]],
        )

    def test_returns_index(self):
        cfg = TheoryOverlayConfig(enabled=True, band_indices="1")
        curves = select_bands_for_view(self._data(), cfg, xlim=(-2, 2), ylim=(-1, 1))
        assert len(curves) == 1
        idx, k, band = curves[0]
        assert idx == 1
        assert np.asarray(k).shape == (3,)

    def test_ef_window_filters(self):
        cfg = TheoryOverlayConfig(enabled=True, band_indices="0,1,2", ef_window=0.5)
        curves = select_bands_for_view(self._data(), cfg, xlim=(-2, 2), ylim=(-3, 3))
        assert [c[0] for c in curves] == [1]

    def test_mirror_after_selection(self):
        cfg = TheoryOverlayConfig(
            enabled=True, band_indices="1", mirror_gamma=True
        )
        curves = select_bands_for_view(self._data(), cfg, xlim=(-2, 2), ylim=(-1, 1))
        assert [c[0] for c in curves] == [1, 1]
        # second = miroir k -> -k
        assert np.allclose(curves[1][1], -np.asarray(curves[0][1]))


class TestSchemaRetrocompat:
    def test_legacy_dict_without_new_keys(self):
        legacy = {
            "source": "materials_project",
            "material_id": "mp-9",
            "k_distance": [0.0, 1.0],
            "bands": [[-1.0, 0.0]],
            "labels": [],
        }
        d = TheoryBandData.from_dict(legacy)
        assert d.band_meta == []
        assert d.band_character == []
        assert d.schema_version == 1  # cache ancien

    def test_roundtrip_preserves_new_fields(self):
        d = TheoryBandData(
            source="x", material_id="mp-2",
            bands=[[0.0]], k_distance=[0.0],
            band_meta=[{"idx": 0, "crosses_ef": True}],
            band_character=["Ti-d"],
        )
        again = TheoryBandData.from_dict(d.to_dict())
        assert again.band_character == ["Ti-d"]
        assert again.band_meta[0]["crosses_ef"] is True
        assert again.schema_version == 2

    def test_config_new_keys_default(self):
        c = TheoryOverlayConfig.from_dict({})
        assert c.ef_window == 0.0
        assert c.color_by_band is True
        c2 = TheoryOverlayConfig.from_dict({"ef_window": -3.0})
        assert c2.ef_window == 0.0  # clampé >=0
