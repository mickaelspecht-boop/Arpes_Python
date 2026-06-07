"""Tests for BandAnalysisController dispersion extraction + guards (no Qt)."""
from __future__ import annotations

import numpy as np
import pytest


class _StubPanel:
    """Captures show_*/options calls from controller."""
    def __init__(self):
        self.last_tb = None
        self.last_kink = None
        self.last_gap = None
        self.tb_opts = {}
        self.kink_opts = {}
        self.gap_opts = {}

    def tb_options(self): return self.tb_opts
    def kink_options(self): return self.kink_opts
    def gap_options(self): return self.gap_opts
    def show_tb_result(self, tb, **kw): self.last_tb = tb
    def show_kink_result(self, k): self.last_kink = k
    def show_gap_result(self, g): self.last_gap = g
    def restore(self, ba): pass


class _StubEntry:
    def __init__(self, fit_result=None):
        self.fit_result = fit_result
        self.band_analysis = None
        self.meta = type("Meta", (), {
            "formula": "",
            "mp_id": "",
            "crystal_a_angstrom": 0.0,
            "crystal_c_angstrom": 0.0,
            "work_function_eV": 0.0,
            "space_group": "",
            "lattice_source": "",
            "sample_config": {},
        })()


class _StubSession:
    def __init__(self, entry):
        self._entry = entry
        self.current_sample = {}
    def key_for_path(self, p): return "key"
    def get_or_create(self, k): return self._entry


class _StubParent:
    def __init__(self, entry, raw_data=None):
        self._session = _StubSession(entry)
        self._current_path = "fake.h5"
        self._raw_data = raw_data or {}
        self._band_panel = _StubPanel()


from arpes.ui.controllers.band_analysis_controller import BandAnalysisController


class TestGuards:
    def test_no_file_warns(self, monkeypatch):
        parent = _StubParent(None)
        parent._current_path = None
        c = BandAnalysisController(parent)
        warned = []
        monkeypatch.setattr(c, "_warn", lambda m: warned.append(m))
        c._run_tb_fit()
        assert warned and "file" in warned[0].lower()

    def test_no_fit_result_warns(self, monkeypatch):
        entry = _StubEntry(fit_result=None)
        parent = _StubParent(entry)
        c = BandAnalysisController(parent)
        warned = []
        monkeypatch.setattr(c, "_warn", lambda m: warned.append(m))
        c._run_kink_analysis()
        assert warned and "mdc" in warned[0].lower()


class TestExtractDispersion:
    def test_crystal_a_unknown_stays_zero(self):
        entry = _StubEntry()
        parent = _StubParent(entry)
        c = BandAnalysisController(parent)
        assert c._crystal_a() == 0.0

    def test_crystal_a_uses_entry_meta_before_raw_meta(self):
        entry = _StubEntry()
        entry.meta.crystal_a_angstrom = 4.2
        parent = _StubParent(entry, raw_data={"meta": {"crystal_a_angstrom": 3.9}})
        c = BandAnalysisController(parent)
        assert c._crystal_a() == 4.2

    def test_converts_pi_a_to_inv_A(self):
        entry = _StubEntry(fit_result={
            "e_fitted": [-0.20, -0.10, 0.0],
            "kF_minus": [[0.5, 0.6, 0.7]],     # π/a units
            "gamma_corrige": [[0.02, 0.025, 0.03]],
        })
        parent = _StubParent(entry)
        c = BandAnalysisController(parent)
        a = 4.0
        E, k, g = c._extract_dispersion(entry.fit_result, a)
        pi_a = np.pi / a
        np.testing.assert_allclose(k, np.array([0.5, 0.6, 0.7]) * pi_a)
        np.testing.assert_allclose(g, np.array([0.02, 0.025, 0.03]) * pi_a)
        np.testing.assert_allclose(E, [-0.20, -0.10, 0.0])

    def test_missing_branch_returns_empty(self):
        entry = _StubEntry(fit_result={"e_fitted": [-0.1, 0.0]})
        parent = _StubParent(entry)
        c = BandAnalysisController(parent)
        E, k, g = c._extract_dispersion(entry.fit_result, 4.0)
        assert len(E) == 0 and len(k) == 0 and g is None


class TestEDCExtraction:
    def test_picks_nearest_k(self):
        E = np.linspace(-0.05, 0.05, 11)
        k = np.linspace(-1.0, 1.0, 21)
        Z = np.outer(np.exp(-E ** 2 / 0.0002), 1.0 / (1.0 + k ** 2 * 100))
        parent = _StubParent(_StubEntry(), raw_data={"E": E, "k": k, "Z": Z})
        c = BandAnalysisController(parent)
        out = c._extract_edc_at_kf(0.0)
        assert out is not None
        Eout, Iout = out
        assert Eout.shape == E.shape
        assert Iout.shape == E.shape
        # k=0 column has the highest values (peak around E=0)
        assert Iout.argmax() == 5  # E=0

    def test_invalid_raw_returns_none(self):
        parent = _StubParent(_StubEntry(), raw_data={"E": [], "k": [], "Z": [[]]})
        c = BandAnalysisController(parent)
        assert c._extract_edc_at_kf(0.0) is None
