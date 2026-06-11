"""Tests for the BESSY SES/R8000 IBW loader using synthetic IBW v5 files.

`_write_ibw5` builds a minimal-but-valid Igor Binary Wave v5 file from
scratch (header layout per the reader's offsets, which follow the public
IBW v5 spec). This doubles as a harness for loader tests: no instrument
files are needed in the repo.
"""
import struct
from pathlib import Path

import numpy as np
import pytest

from arpes.io.loaders.bessy import (
    _is_bessy_ses_ibw,
    _parse_ses_note,
    _read_ibw5_info,
    _read_ibw5_note,
    load_bessy_ses_ibw,
)

_BIN_HEADER = 64
_WAVE_HEADER = 320


def _write_ibw5(
    path: Path,
    data: np.ndarray,
    sf_a: tuple[float, ...],
    sf_b: tuple[float, ...],
    note: str,
    *,
    version: int = 5,
    wave_type: int = 0x02,  # float32
    name: str = "wave0",
) -> Path:
    """Write a synthetic IBW v5: bin header, wave header, data (F order), note."""
    arr = np.asarray(data, dtype="<f4")
    payload = arr.flatten(order="F").tobytes()
    note_bytes = note.encode("latin1")
    wfm_size = _WAVE_HEADER + len(payload)

    bin_header = bytearray(_BIN_HEADER)
    struct.pack_into("<H", bin_header, 0, version)
    struct.pack_into("<I", bin_header, 4, wfm_size)
    struct.pack_into("<I", bin_header, 12, len(note_bytes))

    wave_header = bytearray(_WAVE_HEADER)
    struct.pack_into("<I", wave_header, 12, arr.size)
    struct.pack_into("<H", wave_header, 16, wave_type)
    wave_header[28:28 + len(name)] = name.encode("latin1")
    dims = list(arr.shape) + [0] * (4 - arr.ndim)
    struct.pack_into("<4I", wave_header, 68, *dims)
    sfa = list(sf_a) + [0.0] * (4 - len(sf_a))
    sfb = list(sf_b) + [0.0] * (4 - len(sf_b))
    struct.pack_into("<4d", wave_header, 84, *sfa)
    struct.pack_into("<4d", wave_header, 116, *sfb)

    path.write_bytes(bytes(bin_header) + bytes(wave_header) + payload + note_bytes)
    return path


def _ses_note(extra: str = "", *, scale: str = "Kinetic", center: str = "16.80",
              instrument: str = "R8000-8ES202") -> str:
    lines = [
        "[SES]",
        f"Instrument={instrument}",
        f"Energy Scale={scale}",
        "Lens Mode=Angular30",
        "Pass Energy=10",
        "Number of Sweeps=4",
        "Excitation Energy=0",
        f"Center Energy={center}" if center else "",
        "P-Axis=0.5",
        "Sample=Ba122",
        "Region Name=cut1",
    ]
    note = "\r".join(line for line in lines if line)
    if extra:
        note += "\r" + extra
    return note


N_E, N_TH = 40, 30
SF_A_E, SF_B_E = 0.01, 16.60   # E_kin: 16.60 .. 16.99, center 16.795
SF_A_TH, SF_B_TH = 0.5, -7.25  # theta: -7.25 .. +7.25 deg


def _bm_file(tmp_path: Path, note: str | None = None, **kw) -> Path:
    rng = np.random.default_rng(0)
    data = rng.uniform(10, 20, (N_E, N_TH))  # (E, theta), Igor convention
    return _write_ibw5(
        tmp_path / "cut1.ibw", data,
        sf_a=(SF_A_E, SF_A_TH), sf_b=(SF_B_E, SF_B_TH),
        note=note if note is not None else _ses_note(), **kw,
    )


def _fs_file(tmp_path: Path, *, with_scan_table: bool = True,
             p_values=None) -> Path:
    n_p = 6
    if p_values is None:
        p_values = np.linspace(-2.5, 2.5, n_p)
    rng = np.random.default_rng(1)
    data = rng.uniform(10, 20, (N_E, N_TH, n_p))
    extra = ""
    if with_scan_table:
        extra = "\r".join(f"{i}\t{p:.3f}\t0.0" for i, p in enumerate(p_values))
    return _write_ibw5(
        tmp_path / "fs1.ibw", data,
        sf_a=(SF_A_E, SF_A_TH, 1.0), sf_b=(SF_B_E, SF_B_TH, float(p_values[0])),
        note=_ses_note(extra),
    )


# -------------------------------------------------------------- low level

class TestIbw5Reader:
    def test_roundtrip_info(self, tmp_path):
        path = _bm_file(tmp_path)
        info = _read_ibw5_info(path)
        assert info.dims == (N_E, N_TH)
        assert info.npnts == N_E * N_TH
        assert info.dtype == "<f4"
        assert info.wave_name == "wave0"
        assert info.sf_a[0] == pytest.approx(SF_A_E)
        assert info.sf_b[1] == pytest.approx(SF_B_TH)

    def test_note_roundtrip(self, tmp_path):
        path = _bm_file(tmp_path)
        info = _read_ibw5_info(path)
        note = _read_ibw5_note(path, info)
        assert "[SES]" in note
        assert "Instrument=R8000-8ES202" in note

    def test_rejects_non_v5(self, tmp_path):
        path = _bm_file(tmp_path, version=2)
        with pytest.raises(ValueError, match="v5"):
            _read_ibw5_info(path)

    def test_rejects_unknown_dtype(self, tmp_path):
        path = _bm_file(tmp_path, wave_type=0x3FF)
        with pytest.raises(ValueError, match="type"):
            _read_ibw5_info(path)

    def test_rejects_truncated(self, tmp_path):
        path = tmp_path / "short.ibw"
        path.write_bytes(b"\x05\x00" + b"\x00" * 10)
        with pytest.raises(ValueError, match="short"):
            _read_ibw5_info(path)

    def test_parse_ses_note_values(self):
        ses = _parse_ses_note(_ses_note())
        assert ses["Center Energy"] == pytest.approx(16.80)
        assert ses["Instrument"] == "R8000-8ES202"
        assert ses["Pass Energy"] == pytest.approx(10.0)

    def test_parse_ses_note_scan_table(self):
        note = _ses_note("0\t-2.0\t0.1\r1\t-1.0\t0.2\r2\t0.0\t0.3")
        ses = _parse_ses_note(note)
        np.testing.assert_allclose(ses["P-Axis scan"], [-2.0, -1.0, 0.0])
        np.testing.assert_allclose(ses["R-Axis scan"], [0.1, 0.2, 0.3])


# -------------------------------------------------------------- detection

class TestDetection:
    def test_accepts_synthetic_r8000(self, tmp_path):
        assert _is_bessy_ses_ibw(_bm_file(tmp_path))

    def test_rejects_wrong_suffix(self, tmp_path):
        path = _bm_file(tmp_path)
        renamed = path.with_suffix(".dat")
        path.rename(renamed)
        assert not _is_bessy_ses_ibw(renamed)

    def test_rejects_da30_like_note(self, tmp_path):
        # DA30 exports also contain [SES] but a different instrument.
        path = _bm_file(tmp_path, note=_ses_note(instrument="DA30L"))
        assert not _is_bessy_ses_ibw(path)

    def test_rejects_garbage_file(self, tmp_path):
        path = tmp_path / "junk.ibw"
        path.write_bytes(b"not an ibw at all")
        assert not _is_bessy_ses_ibw(path)


# ------------------------------------------------------------------ BM 2D

class TestLoadBandmap:
    def test_axes_and_shape(self, tmp_path):
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96)
        # Igor (E, theta) -> app (k, E)
        assert ds.data.shape == (N_TH, N_E)
        assert len(ds.energy) == N_E
        assert len(ds.kx) == N_TH
        assert ds.metadata["scan_kind"] == "BM"

    def test_energy_referenced_to_center_energy(self, tmp_path):
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96)
        # E-EF = E_kin - Center Energy; raw axis 16.60..16.99, center 16.80.
        assert ds.energy[0] == pytest.approx(16.60 - 16.80, abs=1e-9)
        assert ds.metadata["energy_reference"] == "ses_center_energy"
        assert ds.metadata["ef_kinetic_nominal"] == pytest.approx(16.80)

    def test_data_orientation_preserved(self, tmp_path):
        # The (i_E, i_theta) sample written must land at data[i_theta, i_E].
        arr = np.zeros((N_E, N_TH), dtype=np.float32)
        arr[7, 3] = 99.0
        path = _write_ibw5(tmp_path / "o.ibw", arr, sf_a=(SF_A_E, SF_A_TH),
                           sf_b=(SF_B_E, SF_B_TH), note=_ses_note())
        ds = load_bessy_ses_ibw(path, a_lattice=3.96)
        assert ds.data[3, 7] == pytest.approx(99.0)

    def test_kx_monotonic_and_centered(self, tmp_path):
        # theta symmetric, P-Axis=0.5: kx strictly increasing, near-centered.
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96)
        assert np.all(np.diff(ds.kx) > 0)
        span = ds.kx.max() - ds.kx.min()
        assert abs(ds.metadata["kx_axis_midpoint"]) < 0.2 * span

    def test_hv_mode_requires_hv(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96,
                               bessy_energy_reference="hv")

    def test_hv_mode_places_ef_from_hv(self, tmp_path):
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96,
                                hv=21.2, work_func=4.4,
                                bessy_energy_reference="hv")
        assert ds.metadata["energy_reference"] == "hv_minus_work_function"
        assert ds.metadata["ef_kinetic_nominal"] == pytest.approx(21.2 - 4.4)
        assert ds.energy[0] == pytest.approx(16.60 - (21.2 - 4.4), abs=1e-9)

    def test_center_vs_hv_discrepancy_warned(self, tmp_path):
        # hv-phi (16.8) far from Center Energy would NOT warn; force a gap >1 eV.
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96,
                                hv=30.0, work_func=4.4)
        warns = ds.metadata["loader_warnings"]
        assert any("differs from Center Energy" in w for w in warns)

    def test_center_energy_fallback(self, tmp_path):
        path = _bm_file(tmp_path, note=_ses_note(center=""))
        ds = load_bessy_ses_ibw(path, a_lattice=3.96)
        assert ds.metadata["center_energy_from_fallback"] is True
        # Fallback = mean of axis endpoints.
        expected = 0.5 * (16.60 + (16.60 + (N_E - 1) * SF_A_E))
        assert ds.metadata["ef_kinetic_nominal"] == pytest.approx(expected)

    def test_binding_axis_sign_flip(self, tmp_path):
        # SES Binding scale: axis 0.05..0.44 eV binding -> E-EF = -axis.
        path = _write_ibw5(
            tmp_path / "b.ibw",
            np.ones((N_E, N_TH), dtype=np.float32),
            sf_a=(SF_A_E, SF_A_TH), sf_b=(0.05, SF_B_TH),
            note=_ses_note(scale="Binding"),
        )
        ds = load_bessy_ses_ibw(path, a_lattice=3.96)
        assert ds.metadata["energy_reference"] == "ses_binding_axis"
        assert ds.energy[0] == pytest.approx(-0.05)
        assert (ds.energy <= 0).all()

    def test_ef_offset_applied(self, tmp_path):
        ds0 = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96)
        ds1 = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96, ef_offset=0.02)
        np.testing.assert_allclose(ds1.energy, ds0.energy + 0.02, atol=1e-9)

    def test_metadata_passthrough(self, tmp_path):
        ds = load_bessy_ses_ibw(_bm_file(tmp_path), a_lattice=3.96,
                                temperature=12.0, pol="LH")
        m = ds.metadata
        assert m["lab"] == "BESSY"
        assert m["sample"] == "Ba122"
        assert m["pass_energy"] == pytest.approx(10.0)
        assert m["temperature"] == pytest.approx(12.0)
        assert m["pol"] == "LH"

    def test_unsupported_ndim_raises(self, tmp_path):
        path = _write_ibw5(tmp_path / "d4.ibw",
                           np.ones((4, 4, 4, 4), dtype=np.float32),
                           sf_a=(1, 1, 1, 1), sf_b=(0, 0, 0, 0),
                           note=_ses_note())
        with pytest.raises(ValueError, match="dimension"):
            load_bessy_ses_ibw(path, a_lattice=3.96)


# ------------------------------------------------------------------ FS 3D

class TestLoadFermiSurface:
    def test_fs_volume_axes(self, tmp_path):
        ds = load_bessy_ses_ibw(_fs_file(tmp_path), a_lattice=3.96)
        m = ds.metadata
        assert m["scan_kind"] == "FS"
        assert m["fs_kind"] == "kxky"
        fs = m["fs_data"]
        assert fs.shape == (6, N_TH, N_E)  # (P, theta/kx, E)
        assert len(m["fs_ky"]) == 6
        assert m["fs_ky_angle_from_note"] is True
        # 2D view = mean over the scan axis.
        np.testing.assert_allclose(ds.data, np.nanmean(fs, axis=0), rtol=1e-6)

    def test_ky_centered_on_scan_midpoint(self, tmp_path):
        ds = load_bessy_ses_ibw(_fs_file(tmp_path), a_lattice=3.96)
        ky = ds.metadata["fs_ky"]
        # Scan -2.5..+2.5 recentered: ky must straddle zero symmetrically.
        assert ky.min() < 0 < ky.max()
        assert abs(ky.min() + ky.max()) < 0.05 * (ky.max() - ky.min())

    def test_ky_rebuilt_without_scan_table(self, tmp_path):
        ds = load_bessy_ses_ibw(_fs_file(tmp_path, with_scan_table=False),
                                a_lattice=3.96)
        m = ds.metadata
        assert m["fs_ky_angle_from_note"] is False
        warns = m["loader_warnings"]
        assert any("Igor scale" in w for w in warns)
        assert len(m["fs_ky"]) == 6

    def test_offcenter_scan_flagged(self, tmp_path):
        p = np.linspace(8.0, 13.0, 6)  # midpoint 10.5 >> span/4
        ds = load_bessy_ses_ibw(_fs_file(tmp_path, p_values=p), a_lattice=3.96)
        flag = ds.metadata["fs_p_axis_offcenter"]
        assert flag is not None
        assert flag["midpoint_deg"] == pytest.approx(10.5)
        assert flag["span_deg"] == pytest.approx(5.0)
