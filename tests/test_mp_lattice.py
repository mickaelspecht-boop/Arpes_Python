"""Tests load_lattice: disk cache + timeout + pymatgen fallback.

mp_api is mocked to avoid network and API-key dependencies.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest

from arpes.physics.bz import Lattice3D
from arpes.theory.materials_project import (
    DEFAULT_MP_TIMEOUT_S,
    MaterialsProjectUnavailable,
    _lattice_cache_path,
    _lattice_from_dict,
    _structure_to_dict,
    load_lattice,
)


# ---- helpers: mock MPRester / structure ------------------------------------


class _FakeLattice:
    def __init__(self, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class _FakeStructure:
    def __init__(self, lat):
        self.lattice = lat


class _FakeSymmetry:
    def __init__(self, symbol="I4/mmm", number=139):
        self.symbol = symbol
        self.number = number


class _FakeDoc:
    def __init__(self, sym=None, structure=None):
        self.symmetry = sym
        self.structure = structure


def _install_fake_mp(monkeypatch, *, structure, crystal_system="Tetragonal",
                     sg_symbol="I4/mmm", sg_number=139, sleep_s=0.0,
                     fail=False):
    """Installs a fake mp_api.client.MPRester into sys.modules."""
    class _FakeMPRester:
        def __init__(self, *a, **kw):
            self._materials = types.SimpleNamespace(
                summary=types.SimpleNamespace(
                    search=lambda **kw: [
                        _FakeDoc(sym=_FakeSymmetry(sg_symbol, sg_number),
                                structure=structure)
                    ]
                )
            )

        def __enter__(self):
            if sleep_s > 0:
                time.sleep(sleep_s)
            if fail:
                raise RuntimeError("mp boom")
            return self

        def __exit__(self, *a):
            return False

        def get_structure_by_material_id(self, mpid):
            return structure

        @property
        def materials(self):
            return self._materials

    fake_module = types.ModuleType("mp_api.client")
    fake_module.MPRester = _FakeMPRester
    parent = types.ModuleType("mp_api")
    parent.client = fake_module
    monkeypatch.setitem(sys.modules, "mp_api", parent)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_module)


# ---- _structure_to_dict ----------------------------------------------------


class TestStructureToDict:
    def test_tetragonal_basics(self):
        s = _FakeStructure(_FakeLattice(3.96, 3.96, 11.6))
        d = _structure_to_dict(s, "mp-1", "Tetragonal", "I4/mmm (139)")
        assert d["a"] == 3.96
        assert d["c"] == 11.6
        assert d["bravais"] == "tetragonal"
        assert d["mp_id"] == "mp-1"
        assert d["schema_version"] == 1

    def test_hexagonal_mapping(self):
        s = _FakeStructure(_FakeLattice(3.0, 3.0, 10.0, gamma=120.0))
        d = _structure_to_dict(s, "mp-2", "Hexagonal", "P6/mmm")
        assert d["bravais"] == "hexagonal"
        assert d["gamma_deg"] == 120.0

    def test_unknown_crystal_system_fallback(self):
        s = _FakeStructure(_FakeLattice(1, 1, 1))
        d = _structure_to_dict(s, "mp-3", "Quasicrystal", "")
        assert d["bravais"] == "tetragonal"  # reasonable fallback


# ---- _lattice_from_dict ----------------------------------------------------


class TestLatticeFromDict:
    def test_roundtrip(self):
        s = _FakeStructure(_FakeLattice(3.96, 3.96, 11.6))
        d = _structure_to_dict(s, "mp-1", "Tetragonal", "I4/mmm")
        lat = _lattice_from_dict(d, Lattice3D)
        assert isinstance(lat, Lattice3D)
        assert lat.a == 3.96
        assert lat.bravais == "tetragonal"
        assert lat.mp_id == "mp-1"

    def test_missing_fields_have_defaults(self):
        lat = _lattice_from_dict({"a": 4.0}, Lattice3D)
        assert lat.a == 4.0
        assert lat.b == 4.0  # default = a
        assert lat.c == 1.0
        assert lat.bravais == "tetragonal"


# ---- load_lattice : cache + MP -------------------------------------------


class TestLoadLatticeCache:
    def test_load_from_cache_disk(self, tmp_path: Path):
        # Writes pre-existing cache and verifies that no MP request is made
        # (MPRester is not installed at all; should pass on cache hit).
        mpid = "mp-cached"
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        d = {
            "mp_id": mpid, "a": 3.96, "b": 3.96, "c": 11.6,
            "alpha_deg": 90.0, "beta_deg": 90.0, "gamma_deg": 90.0,
            "bravais": "tetragonal", "space_group": "I4/mmm",
            "schema_version": 1,
        }
        path = _lattice_cache_path(cache_dir, mpid)
        path.write_text(json.dumps(d))
        lat = load_lattice(mpid, cache_dir=cache_dir)
        assert lat.a == 3.96
        assert lat.mp_id == mpid

    def test_fetch_writes_cache(self, tmp_path: Path, monkeypatch):
        s = _FakeStructure(_FakeLattice(3.5, 3.5, 8.0))
        _install_fake_mp(monkeypatch, structure=s, crystal_system="Tetragonal")
        cache_dir = tmp_path / "c"
        lat = load_lattice("mp-new", cache_dir=cache_dir, api_key="x")
        assert lat.a == 3.5
        cache_file = _lattice_cache_path(cache_dir, "mp-new")
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert cached["mp_id"] == "mp-new"

    def test_force_refresh_bypasses_cache(self, tmp_path: Path, monkeypatch):
        mpid = "mp-1"
        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        # Stale cache (a=999); force_refresh must fetch 3.5 from fake MP.
        path = _lattice_cache_path(cache_dir, mpid)
        path.write_text(json.dumps({"mp_id": mpid, "a": 999.0, "b": 1, "c": 1,
                                    "bravais": "tetragonal"}))
        s = _FakeStructure(_FakeLattice(3.5, 3.5, 8.0))
        _install_fake_mp(monkeypatch, structure=s)
        lat = load_lattice(mpid, cache_dir=cache_dir, force_refresh=True)
        assert lat.a == 3.5


class TestLoadLatticeFailures:
    def test_empty_mpid_raises(self):
        with pytest.raises(ValueError):
            load_lattice("")

    def test_timeout_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        mpid = "mp-slow"
        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        cached = {"mp_id": mpid, "a": 1.23, "b": 1.23, "c": 4.56,
                  "bravais": "tetragonal"}
        _lattice_cache_path(cache_dir, mpid).write_text(json.dumps(cached))
        # Bust cache via mtime trick: NO, keep it and use force_refresh.
        s = _FakeStructure(_FakeLattice(9.0, 9.0, 9.0))
        _install_fake_mp(monkeypatch, structure=s, sleep_s=2.0)
        # Very short timeout → cache fallback → a=1.23 (not 9.0).
        lat = load_lattice(mpid, cache_dir=cache_dir, timeout_s=0.2,
                           force_refresh=True)
        assert lat.a == 1.23

    def test_timeout_no_cache_raises(self, tmp_path: Path, monkeypatch):
        s = _FakeStructure(_FakeLattice(1, 1, 1))
        _install_fake_mp(monkeypatch, structure=s, sleep_s=2.0)
        with pytest.raises(MaterialsProjectUnavailable):
            load_lattice("mp-x", cache_dir=tmp_path / "c", timeout_s=0.2)

    def test_mp_error_falls_back_to_cache(self, tmp_path: Path, monkeypatch):
        mpid = "mp-err"
        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        cached = {"mp_id": mpid, "a": 5.55, "b": 5.55, "c": 1.0,
                  "bravais": "cubic"}
        _lattice_cache_path(cache_dir, mpid).write_text(json.dumps(cached))
        s = _FakeStructure(_FakeLattice(9, 9, 9))
        _install_fake_mp(monkeypatch, structure=s, fail=True)
        lat = load_lattice(mpid, cache_dir=cache_dir, force_refresh=True)
        assert lat.a == 5.55  # fallback cache

    def test_mp_module_missing_falls_back_to_cache(
        self, tmp_path: Path, monkeypatch
    ):
        mpid = "mp-1"
        cache_dir = tmp_path / "c"
        cache_dir.mkdir()
        cached = {"mp_id": mpid, "a": 7.7, "b": 7.7, "c": 7.7,
                  "bravais": "cubic"}
        _lattice_cache_path(cache_dir, mpid).write_text(json.dumps(cached))
        # No fake MP install; also remove it if it happens to be present.
        monkeypatch.delitem(sys.modules, "mp_api.client", raising=False)
        monkeypatch.delitem(sys.modules, "mp_api", raising=False)
        # Cache hit before MP import → returns directly.
        lat = load_lattice(mpid, cache_dir=cache_dir)
        assert lat.a == 7.7


def test_default_timeout_is_10s():
    assert DEFAULT_MP_TIMEOUT_S == 10.0
