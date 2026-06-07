"""Controller orchestrating TB / kink / gap analyses on the current file.

Reads:
- entry.fit_result (MDC fit: e_fitted, kF_minus/plus in π/a, gamma in π/a)
- parent._raw_data (intensity I(E, k) for EDC extraction at k_F)
- crystal_a (from raw meta or fit_params)

Writes:
- entry.band_analysis = {"tb": {...}, "kink": {...}, "gap": {...}}

Surfaces signals back to UI via parent._band_panel.show_*() methods.
"""
from __future__ import annotations

import traceback
import numpy as np

from arpes.core.sample import sample_for_entry
from arpes.physics import tb_fit, kink_analysis, gap_extraction


class BandAnalysisController:
    """Orchestrates TB fit / kink / gap pipelines on the current entry."""

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
        entry = self._current_entry()
        if entry is not None:
            try:
                sample = sample_for_entry(self._session, entry)
                a = sample.a_angstrom if sample.has_lattice_a else 0.0
            except Exception:
                a = 0.0
        if a > 0:
            return float(a)
        try:
            meta = p._raw_data.get("meta", {}) if p._raw_data else {}
            a = float(meta.get("crystal_a_angstrom", 0.0) or 0.0)
        except Exception:
            a = 0.0
        return float(a) if a > 0 else 0.0

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

    def _after_run_refresh(self, entry) -> None:
        """Persist band_analysis and refresh status row + summary."""
        try:
            self._session.save()
        except Exception:
            pass
        try:
            self._refresh_band_analysis_panel()
        except Exception:
            pass

    def _ensure_band_analysis(self, entry) -> dict:
        ba = getattr(entry, "band_analysis", None) or {}
        if not isinstance(ba, dict):
            ba = {}
        entry.band_analysis = ba
        return ba

    def _guard_fit_result(self):
        entry = self._current_entry()
        if entry is None:
            self._warn("No file selected.")
            return None, None
        # Multi-zone: if the user has selected a zone in the strip combo
        # different from active_zone_id, sync it before running so band
        # analysis always operates on the visually-selected zone.
        self._sync_active_zone_with_strip(entry)
        fr = getattr(entry, "fit_result", None)
        if not fr:
            self._warn("Run an MDC fit on the current file first.")
            return None, None
        return entry, fr

    def _sync_active_zone_with_strip(self, entry) -> None:
        p = self._parent
        strip = getattr(getattr(p, "_params", None), "zones_strip", None)
        if strip is None:
            return
        zid_strip = strip.current_zone_id() if hasattr(strip, "current_zone_id") else None
        if not zid_strip or zid_strip == entry.active_zone_id:
            return
        zctrl = getattr(p, "_fit_zones_ctrl", None)
        if zctrl is None:
            return
        try:
            zctrl.fit_zone_action("set_active", {"zone_id": zid_strip})
        except Exception:
            pass

    def _warn(self, msg: str):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self._parent, "Band analysis", msg)
        except Exception:
            print(f"[band_analysis] {msg}")

    def _info(self, msg: str):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self._parent, "Band analysis", msg)
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
            self._warn(f"Branch {branch}/pair {pair}: fewer than 3 usable points.")
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
            self._warn(f"TB fit failed: {exc}\n{traceback.format_exc()}")
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
        self._after_run_refresh(entry)

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
            self._warn("Kink: fewer than 8 points in the dispersion.")
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
            self._warn(f"Kink analysis failed: {exc}")
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
        self._after_run_refresh(entry)

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
            self._warn("No dispersion — cannot locate k_F.")
            return
        # k_F at E closest to E_F=0
        idx_F = int(np.argmin(np.abs(E_disp - float(opts.get("E_F", 0.0)))))
        k_F = float(k_disp[idx_F])
        edc = self._extract_edc_at_kf(k_F)
        if edc is None:
            self._warn("Cannot extract EDC at k_F (raw_data missing or inconsistent).")
            return
        E_axis, I_edc = edc
        try:
            omega, sym = gap_extraction.symmetrize_edc(
                E_axis, I_edc,
                E_F=float(opts.get("E_F", 0.0)),
                omega_max_meV=float(opts.get("omega_max_meV", 30.0)),
            )
            # P2.5 — EDC ARPES symétrisé → fonction spectrale Norman (1998),
            # pas la DOS Dynes (tunnel). Dynes reste dispo pour comparaison STS.
            if int(opts.get("n_gaps", 1)) == 2:
                res = gap_extraction.fit_norman_two_gap(
                    omega, sym,
                    resolution_meV=float(opts.get("resolution_meV", 0.0)),
                )
            else:
                res = gap_extraction.fit_norman_single(
                    omega, sym,
                    resolution_meV=float(opts.get("resolution_meV", 0.0)),
                )
        except Exception as exc:
            self._warn(f"Gap fit failed: {exc}")
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
        self._after_run_refresh(entry)

    # ------------------------------------------------------------------
    # Restore on file switch
    # ------------------------------------------------------------------

    def _refresh_band_analysis_panel(self):
        """Repopulate panel from entry.band_analysis when current file changes."""
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        entry = self._current_entry()
        ba = (getattr(entry, "band_analysis", None) if entry else None) or {}
        panel.restore(ba)
        fr = getattr(entry, "fit_result", None) if entry else None
        n_pairs = 1
        n_pts = 0
        if fr:
            e_fit = fr.get("e_fitted")
            n_pts = 0 if e_fit is None else len(e_fit)
            try:
                n_pairs = max(
                    len(fr.get("kF_minus") or []),
                    len(fr.get("kF_plus") or []),
                    1,
                )
            except Exception:
                n_pairs = 1
        try:
            panel.update_prerequisites(
                has_fit=bool(fr), n_pairs=n_pairs, n_points=n_pts,
            )
            panel.update_stage_row(
                ba, has_fit=bool(fr), n_points=n_pts, n_pairs=n_pairs,
            )
            panel.update_summary(
                ba, has_fit=bool(fr), n_points=n_pts, n_pairs=n_pairs,
            )
        except Exception:
            pass

    @staticmethod
    def build_csv_rows(entry, ba: dict) -> list[tuple[str, str, str, str, str]]:
        """Build the (source, metric, value, error, unit) row list for CSV export.

        Pure function (no Qt) so it's directly testable.
        """
        rows: list[tuple[str, str, str, str, str]] = [
            ("source", "metric", "value", "error", "unit"),
        ]
        fr = getattr(entry, "fit_result", None) or {}
        if fr.get("e_fitted") is not None:
            rows.append(("MDC", "n_points", str(len(fr["e_fitted"])), "", ""))
        tb = ba.get("tb") or {}
        for name, v in (tb.get("params") or {}).items():
            err = (tb.get("perr") or {}).get(name, 0.0)
            rows.append(("TB", name, f"{v:.6f}", f"{err:.6f}", "eV"))
        if tb.get("m_eff_over_me") is not None:
            rows.append(("TB", "m_eff_over_me",
                         f"{tb['m_eff_over_me']:.4f}", "", ""))
        if tb.get("bandwidth_eV") is not None:
            rows.append(("TB", "W", f"{tb['bandwidth_eV']:.4f}", "", "eV"))
        if tb.get("chi2_red") is not None:
            rows.append(("TB", "chi2_red", f"{tb['chi2_red']:.4e}", "", ""))
        kink = ba.get("kink") or {}
        if kink.get("lambda") is not None:
            err = kink.get("lambda_err") or 0.0
            rows.append(("Kink", "lambda",
                         f"{kink['lambda']:.4f}", f"{err:.4f}", ""))
        if kink.get("v_bare") is not None:
            rows.append(("Kink", "v_bare",
                         f"{kink['v_bare']:.4f}", "", "eV.A"))
        gap = ba.get("gap") or {}
        for i, D in enumerate(gap.get("deltas_meV") or []):
            errs = gap.get("delta_err_meV") or []
            e = errs[i] if i < len(errs) else 0.0
            rows.append(("Gap", f"Delta_{i+1}",
                         f"{D:.4f}", f"{e:.4f}", "meV"))
        for i, G in enumerate(gap.get("gammas_meV") or []):
            rows.append(("Gap", f"Gamma_{i+1}",
                         f"{G:.4f}", "", "meV"))
        if gap.get("k_F_inv_A") is not None:
            rows.append(("Gap", "k_F",
                         f"{gap['k_F_inv_A']:.6f}", "", "A^-1"))
        if gap.get("chi2_red") is not None:
            rows.append(("Gap", "chi2_red",
                         f"{gap['chi2_red']:.4e}", "", ""))
        return rows

    def _export_band_analysis_csv(self) -> None:
        """Write all metrics from entry.band_analysis to a user-chosen CSV."""
        entry = self._current_entry()
        if entry is None:
            self._warn("No file selected.")
            return
        ba = getattr(entry, "band_analysis", None) or {}
        if not ba:
            self._warn("No analysis to export. Run TB / Kink / Gap first.")
            return
        try:
            from PyQt6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self._parent, "Export band analysis", "", "CSV (*.csv)",
            )
        except Exception:
            path = ""
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path = path + ".csv"
        rows = self.build_csv_rows(entry, ba)
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            self._info(f"Export OK: {path}")
        except Exception as exc:
            self._warn(f"CSV export failed: {exc}")

    # ------------------------------------------------------------------
    # Auto-fill defaults from current context
    # ------------------------------------------------------------------
    @staticmethod
    def compute_autofill_defaults(target: str, entry, *, ef_offset: float = 0.0) -> dict:
        """Pure function returning the autofill dict for the chosen tab.

        Decoupled from the panel + spinboxes so it's directly testable.
        """
        defaults: dict = {}
        fr = getattr(entry, "fit_result", None) or {}
        chosen_branch = "kF_minus"
        for b in ("kF_minus", "kF_plus"):
            arrs = fr.get(b) or []
            if arrs and any(
                np.any(np.isfinite(np.asarray(arr, float))) for arr in arrs
            ):
                chosen_branch = b
                break
        defaults["branch"] = chosen_branch
        defaults["E_F"] = float(ef_offset)
        if target == "tb":
            try:
                a = float(getattr(entry.meta, "crystal_a_angstrom", 0.0) or 0.0)
            except Exception:
                a = 0.0
            if a > 0:
                defaults["a"] = a
        elif target == "kink":
            e_fit = np.asarray(fr.get("e_fitted") or [], float)
            if e_fit.size >= 4:
                e_min = float(np.nanmin(e_fit))
                e_max = float(np.nanmax(e_fit))
                span = e_max - e_min
                defaults["window_lo"] = e_min
                defaults["window_hi"] = e_min + 0.6 * span
        elif target == "gap":
            defaults["omega_max_meV"] = 30.0
        return defaults

    def _autofill_band_analysis(self, target: str) -> None:
        panel = getattr(self._parent, "_band_panel", None)
        if panel is None:
            return
        entry = self._current_entry()
        if entry is None:
            return
        try:
            ef = float(self._parent._params.sp_ef.value())
        except Exception:
            ef = 0.0
        defaults = self.compute_autofill_defaults(target, entry, ef_offset=ef)
        panel.apply_autofill(target, defaults)
