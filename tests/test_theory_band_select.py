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
    available_segments,
    branch_display_names,
    parse_band_indices,
    segment_from_direction,
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


class TestRealMpBranches:
    BR = [
        {"name": "\\Gamma-X", "start": 0, "end": 2},
        {"name": "X-M", "start": 3, "end": 5},
        {"name": "M-\\Gamma", "start": 6, "end": 8},
        {"name": "\\Gamma-X", "start": 9, "end": 11},  # repasse
    ]

    def test_display_names_disambiguate(self):
        assert branch_display_names(self.BR) == [
            "Γ-X", "X-M", "M-Γ", "Γ-X (2)",
        ]

    def test_available_segments_only_real_path(self):
        # branches présentes → on ne propose QUE le chemin réel
        segs = available_segments([{"label": "Γ", "k": 0.0}], self.BR)
        assert segs == ["Γ-X", "X-M", "M-Γ", "Γ-X (2)"]

    def test_available_segments_fallback_without_branches(self):
        labels = [{"label": "Γ", "k": 0.0}, {"label": "X", "k": 1.0}]
        assert "Γ-X" in available_segments(labels, None)

    def test_segment_from_direction_prefers_branches(self):
        assert segment_from_direction("Gamma-X", [], self.BR) == "Γ-X"
        assert segment_from_direction("M-Gamma", [], self.BR) == "M-Γ"
        assert segment_from_direction("K-W", [], self.BR) == ""

    def test_segment_mask_uses_branch_indices(self):
        n = 12
        bands = [[float(i) - 6 for i in range(n)]]  # rampe -6..5
        data = TheoryBandData(
            source="materials_project", material_id="mp-x",
            k_distance=[float(i) for i in range(n)],
            bands=bands, branches=self.BR,
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="X-M",
                                  band_indices="0")
        curves = select_bands_for_view(data, cfg, xlim=(-99, 99),
                                       ylim=(-99, 99))
        _idx, _k, band = curves[0]
        finite = np.where(np.isfinite(band))[0]
        # X-M = indices 3..5 uniquement
        assert finite.min() == 3 and finite.max() == 5

    def test_second_occurrence_distinct_window(self):
        n = 12
        data = TheoryBandData(
            source="materials_project", material_id="mp-x",
            k_distance=[float(i) for i in range(n)],
            bands=[[1.0] * n], branches=self.BR,
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X (2)",
                                  band_indices="0")
        _i, _k, band = select_bands_for_view(
            data, cfg, xlim=(-99, 99), ylim=(-99, 99))[0]
        finite = np.where(np.isfinite(band))[0]
        assert finite.min() == 9 and finite.max() == 11


class TestBranchLocalK:
    def _data(self, branches):
        n = 12
        return TheoryBandData(
            source="materials_project", material_id="mp-x",
            k_distance=[float(i) for i in range(n)],  # 0..11 global
            bands=[[1.0] * n], branches=branches,
        )

    def test_gamma_x_local_0_to_1(self):
        data = self._data([{"name": "\\Gamma-X", "start": 0, "end": 5}])
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X",
                                  band_indices="0")
        _i, k, band = select_bands_for_view(
            data, cfg, xlim=(-9, 9), ylim=(-9, 9))[0]
        kk = np.asarray(k)[np.isfinite(band)]
        assert kk.min() == pytest.approx(0.0)  # Γ
        assert kk.max() == pytest.approx(1.0)  # bord zone

    def test_x_gamma_inverted_gamma_at_zero(self):
        data = self._data([{"name": "X-\\Gamma", "start": 0, "end": 5}])
        cfg = TheoryOverlayConfig(enabled=True, segment="X-Γ",
                                  band_indices="0")
        _i, k, band = select_bands_for_view(
            data, cfg, xlim=(-9, 9), ylim=(-9, 9))[0]
        kk = np.asarray(k)
        # Γ (extrémité finale start..end) doit être ramené à k=0
        assert kk[5] == pytest.approx(0.0)
        assert kk[0] == pytest.approx(1.0)

    def test_no_branches_keeps_global_axis(self):
        n = 4
        data = TheoryBandData(
            source="local", material_id="x",
            k_distance=[0.0, 1.0, 2.0, 3.0], bands=[[1.0] * n],
        )
        cfg = TheoryOverlayConfig(enabled=True, band_indices="0")
        _i, k, _b = select_bands_for_view(
            data, cfg, xlim=(-9, 9), ylim=(-9, 9))[0]
        assert np.allclose(k, [0.0, 1.0, 2.0, 3.0])


class TestPhysicalKAlignment:
    def test_abs_distance_converted_a_over_pi(self):
        # branche Γ-X, distances absolues Å⁻¹ = 0,0.5,1.0,1.5 ; a=4.0
        n = 4
        data = TheoryBandData(
            source="materials_project", material_id="mp-x",
            k_distance=[-1.0, -0.33, 0.33, 1.0],  # normalisé (ignoré)
            k_distance_abs=[0.0, 0.5, 1.0, 1.5],
            bands=[[1.0] * n],
            branches=[{"name": "\\Gamma-X", "start": 0, "end": 3}],
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X",
                                  band_indices="0", crystal_a=4.0)
        _i, k, band = select_bands_for_view(
            data, cfg, xlim=(-9, 9), ylim=(-9, 9))[0]
        kk = np.asarray(k)[np.isfinite(band)]
        # k[π/a] = dist · a/π : Γ=0, X = 1.5·4/π
        assert kk[0] == pytest.approx(0.0)
        assert kk[-1] == pytest.approx(1.5 * 4.0 / np.pi)

    def test_fallback_normalized_when_no_a(self):
        n = 4
        data = TheoryBandData(
            source="materials_project", material_id="mp-x",
            k_distance=[0, 1, 2, 3],
            k_distance_abs=[0.0, 0.5, 1.0, 1.5],
            bands=[[1.0] * n],
            branches=[{"name": "\\Gamma-X", "start": 0, "end": 3}],
        )
        cfg = TheoryOverlayConfig(enabled=True, segment="Γ-X",
                                  band_indices="0", crystal_a=0.0)
        _i, k, band = select_bands_for_view(
            data, cfg, xlim=(-9, 9), ylim=(-9, 9))[0]
        kk = np.asarray(k)[np.isfinite(band)]
        assert kk.min() == pytest.approx(0.0)
        assert kk.max() == pytest.approx(1.0)  # [0,1] legacy

    def test_roundtrip_abs_and_crystal_a(self):
        d = TheoryBandData(source="x", material_id="m",
                            bands=[[0.0]], k_distance=[0.0],
                            k_distance_abs=[0.0, 1.0])
        assert TheoryBandData.from_dict(d.to_dict()).k_distance_abs == [0.0, 1.0]
        c = TheoryOverlayConfig.from_dict({"crystal_a": 4.143})
        assert c.crystal_a == pytest.approx(4.143)
        assert TheoryOverlayConfig.from_dict({}).crystal_a == 0.0


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

    def test_crystal_system_legacy_default_and_roundtrip(self):
        assert TheoryBandData.from_dict({"material_id": "mp-1"}).crystal_system == ""
        d = TheoryBandData(source="materials_project", material_id="mp-1",
                            crystal_system="Tetragonal")
        assert TheoryBandData.from_dict(d.to_dict()).crystal_system == "Tetragonal"

    def test_config_new_keys_default(self):
        c = TheoryOverlayConfig.from_dict({})
        assert c.ef_window == 0.0
        assert c.color_by_band is True
        c2 = TheoryOverlayConfig.from_dict({"ef_window": -3.0})
        assert c2.ef_window == 0.0  # clampé >=0
