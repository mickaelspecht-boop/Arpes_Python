"""Controller orchestrating TB / kink / gap analyses on the current file.

Reads:
- entry.fit_result (MDC fit: e_fitted, kF_minus/plus in π/a, gamma in π/a)
- parent._raw_data (intensity I(E, k) for EDC extraction at k_F)
- crystal_a (from raw meta or fit_params)

Writes:
- entry.band_analysis = {"tb": {...}, "kink": {...}, "gap": {...}}

Surface signals back to UI via parent._band_panel.show_*() methods.
"""
from __future__ import annotations

import traceback
import numpy as np

from arpes.physics import tb_fit, kink_analysis, gap_extraction


DEFAULT_CRYSTAL_A_ANGSTROM = 4.143  # BaNi2As2-like fallback


class BandAnalysisController:
    """Orchestrate TB fit / kink / gap pipelines on current entry."""

    def __init__(self, parent):
        self._parent = parent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _session(self):
        return self._parent._session

    def _current_entry(self):
        p = self._parent
        path = getattr(p, "_current_path", None)
        if not path:
            return None
        return self._session.get_or_create(self._session.key_for_path(path))

    def _crystal_a(self) -> float:
        p = self._parent
        a = 0.0
        try:
            meta = p._raw_data.get("meta", {}) if p._raw_data else {}
            a = float(meta.get("crystal_a_angstrom", 0.0) or 0.0)
        except Exception:
            a = 0.0
        if a <= 0:
            a = DEFAULT_CRYSTAL_A_ANGSTROM
        return a

    def _extract_dispersion(
        self,
        fit_result: dict,
        crystal_a: float,
        branch: str = "kF_minus",
        pair_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (E, k_inv_A, gamma_inv_A) from fit_result.

        fit_result stores k in π/a → convert to Å⁻¹.
        """
        e_raw = fit_result.get("e_fitted")
        if e_raw is None:
            return np.array([]), np.array([]), None
        E = np.asarray(e_raw, float)
        branches = fit_result.get(branch)
        if branches is None or not (0 <= pair_index < len(branches)):
            return np.array([]), np.array([]), None
        k_pi_a = np.asarray(branches[pair_index], float)
        pi_over_a = np.pi / float(crystal_a)
        k_inv_A = k_pi_a * pi_over_a
        g = fit_result.get("gamma_corrige") or fit_result.get("gamma")
        g_inv_A = None
        if g is not None and pair_index < len(g):
            g_pi_a = np.asarray(g[pair_index], float)
            if g_pi_a.size == E.size:
                g_inv_A = g_pi_a * pi_over_a
        mask = np.isfinite(E) & np.isfinite(k_inv_A)
        E = E[mask]
        k_inv_A = k_inv_A[mask]
        if g_inv_A is not None:
            g_inv_A = g_inv_A[mask] if g_inv_A.size == mask.size else None
        return E, k_inv_A, g_inv_A

    def _ensure_band_analysis(self, entry) -> dict:
        ba = getattr(entry, "band_analysis", None) or {}
        if not isinstance(ba, dict):
            ba = {}
        entry.band_analysis = ba
        return ba

    def _guard_fit_result(self):
        entry = self._current_entry()
        if entry is None:
            self._warn("Aucun fichier sélectionné.")
            return None, None
        fr = getattr(entry, "fit_result", None)
        if not fr:
            self._warn("Lance d'abord un fit MDC sur le fichier courant.")
            return None, None
        return entry, fr

    def _warn(self, msg: str):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self._parent, "Analyse de bandes", msg)
        except Exception:
            print(f"[band_analysis] {msg}")

    def _info(self, msg: str):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self._parent, "Analyse de bandes", msg)
        except Exception:
            print(f"[band_analysis] {msg}")

    # ------------------------------------------------------------------
    # TB fit
    # ------------------------------------------------------------------

    def _run_tb_fit(self):
        """Triggered by panel "Run TB fit" button."""
        entry, fr = self._guard_fit_result()
        if entry is None:
            return
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        opts = panel.tb_options()  # dict: lattice_type, a, b, branch, pair
        crystal_a = float(opts.get("a") or self._crystal_a())
        branch = opts.get("branch", "kF_minus")
        pair = int(opts.get("pair", 0))
        E, k, _g = self._extract_dispersion(fr, crystal_a, branch, pair)
        if len(E) < 3:
            self._warn(f"Branche {branch}/pair {pair} : <3 points utilisables.")
            return
        try:
            lt = opts.get("lattice_type", "chain")
            if lt == "chain":
                res = tb_fit.fit_dispersion_1d(k, E, crystal_a)
            else:
                # For 2D from a 1D MDC cut, set ky=0 (cut along Γ-X assumed).
                ky = np.zeros_like(k)
                res = tb_fit.fit_dispersion_2d(
                    k, ky, E, crystal_a,
                    lattice_type=lt,
                    b=opts.get("b"),
                )
        except Exception as exc:
            self._warn(f"Échec fit TB : {exc}\n{traceback.format_exc()}")
            return
        ba = self._ensure_band_analysis(entry)
        ba["tb"] = {
            "model": res.model,
            "lattice_type": res.lattice_type,
            "params": res.params,
            "perr": res.perr,
            "a": res.a,
            "b": res.b,
            "m_eff_over_me": res.m_eff_over_me,
            "bandwidth_eV": res.bandwidth_eV,
            "chi2_red": res.chi2_red,
            "n_points": res.n_points,
            "notes": res.notes,
            "branch": branch,
            "pair_index": pair,
        }
        panel.show_tb_result(ba["tb"], k=k, E=E,
                             E_fit=tb_fit.evaluate_tb_model(res, k,
                                  ky=np.zeros_like(k) if res.lattice_type != "chain" else None))

    # ------------------------------------------------------------------
    # Kink
    # ------------------------------------------------------------------

    def _run_kink_analysis(self):
        entry, fr = self._guard_fit_result()
        if entry is None:
            return
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        opts = panel.kink_options()  # {branch, pair, bare, window_lo, window_hi, lambda_window, E_F}
        crystal_a = self._crystal_a()
        E, k, g = self._extract_dispersion(
            fr, crystal_a, opts.get("branch", "kF_minus"),
            int(opts.get("pair", 0))
        )
        if len(E) < 8:
            self._warn("Kink: <8 points dans la dispersion.")
            return
        try:
            res = kink_analysis.run_kink_analysis(
                E, k,
                E_F=float(opts.get("E_F", 0.0)),
                bare=opts.get("bare", "parabolic"),
                bare_window_eV=(
                    float(opts.get("window_lo", -0.3)),
                    float(opts.get("window_hi", -0.08)),
                ),
                gamma_mdc=g,
                lambda_window_eV=float(opts.get("lambda_window", 0.05)),
            )
        except Exception as exc:
            self._warn(f"Échec kink : {exc}")
            return
        ba = self._ensure_band_analysis(entry)
        ba["kink"] = {
            "lambda": res.lambda_coupling,
            "lambda_err": res.lambda_err,
            "v_bare": res.v_bare,
            "bare_model": res.bare_model,
            "bare_params": res.bare_params,
            "E_exp": res.E_exp.tolist(),
            "k_exp": res.k_exp.tolist(),
            "re_sigma": res.re_sigma.tolist(),
            "im_sigma": res.im_sigma.tolist() if res.im_sigma is not None else None,
            "notes": res.notes,
        }
        panel.show_kink_result(ba["kink"])

    # ------------------------------------------------------------------
    # Gap
    # ------------------------------------------------------------------

    def _extract_edc_at_kf(self, k_F_inv_A: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract EDC I(E) at given k_F from raw_data."""
        p = self._parent
        raw = getattr(p, "_raw_data", None)
        if not raw:
            return None
        try:
            E = np.asarray(raw["E"], float)
            k = np.asarray(raw["k"], float)
            Z = np.asarray(raw["Z"], float)
        except (KeyError, ValueError, TypeError):
            return None
        if Z.shape != (E.size, k.size):
            return None
        idx = int(np.argmin(np.abs(k - k_F_inv_A)))
        return E, Z[:, idx]

    def _run_gap_fit(self):
        entry, fr = self._guard_fit_result()
        if entry is None:
            return
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        opts = panel.gap_options()  # {branch, pair, n_gaps, resolution_meV, omega_max_meV, E_F}
        crystal_a = self._crystal_a()
        E_disp, k_disp, _ = self._extract_dispersion(
            fr, crystal_a, opts.get("branch", "kF_minus"),
            int(opts.get("pair", 0))
        )
        if len(E_disp) == 0:
            self._warn("Pas de dispersion → impossible de localiser k_F.")
            return
        # k_F at E closest to E_F=0
        idx_F = int(np.argmin(np.abs(E_disp - float(opts.get("E_F", 0.0)))))
        k_F = float(k_disp[idx_F])
        edc = self._extract_edc_at_kf(k_F)
        if edc is None:
            self._warn("Impossible d'extraire l'EDC à k_F (raw_data manquant ou incohérent).")
            return
        E_axis, I_edc = edc
        try:
            omega, sym = gap_extraction.symmetrize_edc(
                E_axis, I_edc,
                E_F=float(opts.get("E_F", 0.0)),
                omega_max_meV=float(opts.get("omega_max_meV", 30.0)),
            )
            if int(opts.get("n_gaps", 1)) == 2:
                res = gap_extraction.fit_dynes_two_gap(
                    omega, sym,
                    resolution_meV=float(opts.get("resolution_meV", 0.0)),
                )
            else:
                res = gap_extraction.fit_dynes_single(
                    omega, sym,
                    resolution_meV=float(opts.get("resolution_meV", 0.0)),
                )
        except Exception as exc:
            self._warn(f"Échec fit gap : {exc}")
            return
        ba = self._ensure_band_analysis(entry)
        ba["gap"] = {
            "k_F_inv_A": k_F,
            "deltas_meV": res.deltas_meV,
            "delta_err_meV": res.delta_err_meV,
            "gammas_meV": res.gammas_meV,
            "weights": res.weights,
            "n_gaps": res.n_gaps,
            "resolution_meV": res.resolution_meV,
            "chi2_red": res.chi2_red,
            "notes": res.notes,
            "omega_meV": res.omega_meV.tolist(),
            "I_sym": res.I_sym.tolist(),
            "I_fit": res.I_fit.tolist(),
        }
        panel.show_gap_result(ba["gap"])

    # ------------------------------------------------------------------
    # Restore on file switch
    # ------------------------------------------------------------------

    def _refresh_band_analysis_panel(self):
        """Repopulate panel from entry.band_analysis when current file changes."""
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        entry = self._current_entry()
        ba = getattr(entry, "band_analysis", None) if entry else None
        panel.restore(ba or {})
