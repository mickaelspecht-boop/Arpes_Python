"""Qt controller for the KZ tab."""
from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from arpes.core.sample import sample_for_entry, work_function_for_entry
from arpes.io.kz_dataset import KzDataset, dataset_summary, load_kz_stack
from arpes.physics.kz import (
    KzParams,
    compute_hv_k_map,
    compute_kz_map,
    convert_kz_unit,
    fit_inner_potential,
    kz_coverage_summary,
    kz_high_symmetry_planes,
    kz_profile_at_normal_emission,
    kz_unit_to_inv_a,
    scan_from_legacy_dict,
)


class KzController:
    def __init__(self, parent):
        self._parent = parent
        self._dataset: KzDataset | None = None
        self._last_folder: str = ""

    @property
    def _params(self):
        return self._parent._params

    def _status(self, msg: str) -> None:
        self._parent._status(msg)

    def _current_entry(self):
        path = getattr(self._parent, "_current_path", None)
        session = getattr(self._parent, "_session", None)
        if not path or session is None:
            return None
        return session.get_or_create(session.key_for_path(path))

    def _work_func(self) -> float:
        fallback = float(getattr(self._parent._session, "work_func", 0.0) or 0.0)
        try:
            fallback = float(self._params.sp_phi.value())
        except Exception:
            pass
        return work_function_for_entry(
            self._parent._session,
            self._current_entry(),
            fallback=fallback,
        )

    def _lattice_a(self) -> float:
        """Lattice a (Å) for the angle→k// conversion at load time.

        Prefer the value typed in the KZ tab, else the sample's. Without it the
        CLS photon-scan loader leaves k// at zero (degenerate kz map).
        """
        try:
            typed = float(self._parent._kz_controls.sp_a.value())
        except Exception:
            typed = 0.0
        if typed > 0:
            return typed
        try:
            return float(sample_for_entry(self._parent._session, self._current_entry()).a_angstrom)
        except Exception:
            return 0.0

    def _current_supports_kz(self) -> bool:
        d = self._parent._raw_data
        if d is None:
            return False
        try:
            scan_from_legacy_dict(d)
        except Exception:
            return False
        return True

    def _open_kz_folder(self):
        start = self._last_folder
        if not start and self._parent._session.folder:
            start = str(self._parent._session.folder)
        folder = QFileDialog.getExistingDirectory(self._parent, "Choose KZ folder", start or str(Path.home()))
        if not folder:
            return
        self._last_folder = folder
        self._refresh_kz_dataset()

    def _open_kz_logbook(self):
        start = self._last_folder or str(self._parent._session.folder or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Logbook KZ",
            start,
            "All files (*);;Logbook (*.xlsx *.xls *.xlsm *.csv *.tsv *.txt)",
        )
        if not path:
            return
        try:
            records, mapping, sheet_name = self._parent._logbook_ctrl.read(Path(path))
            session = self._parent._session
            session.kz_logbook_path = path
            session.kz_logbook_sheet = sheet_name
            session.kz_logbook_mapping = mapping
            session.kz_logbook_records = records
            session.save()
            used = ", ".join(f"{k}={v or '—'}" for k, v in mapping.items())
            sheet_txt = f" [{sheet_name}]" if sheet_name else ""
            self._status(f"KZ logbook loaded: {Path(path).name}{sheet_txt} | {len(records)} rows | {used}")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"KZ logbook loaded ({len(records)} rows)")
            QMessageBox.information(
                self._parent,
                "KZ Logbook Loaded",
                f"{Path(path).name}{sheet_txt}\n{len(records)} rows read.\n\nDetected columns:\n{used}",
            )
            if self._last_folder:
                self._refresh_kz_dataset()
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook KZ", str(exc))
            self._status(f"Warning: Logbook KZ : {exc}")

    @staticmethod
    def _format_hv_sources(sources: dict) -> str:
        if not sources:
            return "hν source: unknown"
        bits = [f"{key}={value}" for key, value in sorted(sources.items())]
        return "hν source: " + ", ".join(bits)

    def _refresh_kz_dataset(self):
        if not self._last_folder:
            self._status("Warning: KZ: choose a folder")
            return
        try:
            ds = load_kz_stack(
                self._last_folder,
                work_func=self._work_func(),
                ef_offset=float(self._params.sp_ef.value()),
                a_lattice=self._lattice_a(),
                hv_fallback=float(self._params.sp_hv.value()) if self._params.sp_hv.value() > 0 else None,
                kz_logbook_records=self._parent._session.kz_logbook_records,
                kz_logbook_mapping=self._parent._session.kz_logbook_mapping,
                main_logbook_records=self._parent._session.logbook_records,
                main_logbook_mapping=self._parent._session.logbook_mapping,
                session_folder=self._parent._session.folder,
            )
            self._dataset = ds
            summary = dataset_summary(ds)
            n_ignored = len(summary["warnings"])
            self._parent._kz_controls.set_info(
                f"{summary['n_scans']} scans kept | {n_ignored} ignored | "
                f"hν={summary['hv_min']:.1f}→{summary['hv_max']:.1f} eV | "
                f"{self._format_hv_sources(summary['hv_sources'])}"
            )
            self._status(f"OK KZ: {summary['n_scans']} scans loaded")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"KZ loaded ({summary['n_scans']} scans)")
            self._draw_kz_tab()
        except Exception as exc:
            self._dataset = None
            self._parent._kz_controls.set_info(f"KZ error: {exc}")
            self._status(f"Warning: KZ: {exc}")
            traceback.print_exc()

    def _on_kz_params_changed(self, _=None):
        self._draw_kz_tab()

    def _draw_kz_tab(self):
        if not hasattr(self._parent, "_kz_canvas"):
            return
        fig = self._parent._kz_canvas.map.fig
        while len(fig.axes) > 1:
            fig.delaxes(fig.axes[-1])
        ax = self._parent._kz_canvas.map.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        if self._dataset is None:
            self._kz_placeholder(ax, "Choose a KZ folder\n(variable-hν band-map series)")
            return

        controls = self._parent._kz_controls
        self._autofill_lattice(controls)
        ui = controls.params()

        view = ui.view
        note = ""
        if view == "kz" and (ui.a_lattice <= 0 or ui.c_lattice <= 0):
            view = "raw"
            controls.force_raw_view()
            note = "set lattice a & c (sample) for the kz view — showing Raw"

        params = KzParams(
            work_func=self._work_func(),
            inner_potential=ui.inner_potential,
            a_lattice=ui.a_lattice,
            c_lattice=ui.c_lattice,
            energy_center=ui.energy_center,
            energy_window=ui.energy_window,
            k_bins=ui.k_bins,
            kz_bins=ui.kz_bins,
            kz_unit=ui.kz_unit,
            normalize=ui.normalize,
        )
        try:
            if view == "raw":
                info = self._draw_raw_view(ax, params, ui)
            else:
                info = self._draw_kz_view(ax, params, ui)
        except Exception as exc:
            self._kz_placeholder(ax, str(exc), color="tomato")
            self._status(f"Warning: KZ : {exc}")
            return

        if note:
            info = f"{note} | {info}"
        self._style_axes(ax)
        controls.set_info(info)
        fig.tight_layout(pad=0.8)
        self._parent._kz_canvas.map.redraw()

    # --- draw helpers -----------------------------------------------------

    def _kz_placeholder(self, ax, text: str, *, color: str = "white") -> None:
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", color=color, fontsize=11)
        self._parent._kz_canvas.map.redraw()

    @staticmethod
    def _style_axes(ax) -> None:
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

    @staticmethod
    def _vmax(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.nanpercentile(finite, 99)) if finite.size else 1.0

    def _colorbar(self, artist, ax) -> None:
        try:
            self._parent._kz_canvas.map.fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            pass

    def _autofill_lattice(self, controls) -> None:
        try:
            sample = sample_for_entry(self._parent._session, self._current_entry())
        except Exception:
            return
        controls.autofill_lattice(sample.a_angstrom, sample.c_angstrom)

    def _draw_raw_view(self, ax, params: KzParams, ui) -> str:
        result = compute_hv_k_map(self._dataset.scans, params)
        artist = ax.pcolormesh(
            result.k_grid, result.hv_grid, result.image,
            shading="auto", cmap="inferno", vmin=0.0, vmax=max(self._vmax(result.image), 1e-12),
        )
        ax.set_xlabel("k// (π/a)", color="white")
        ax.set_ylabel("hν (eV)", color="white")
        ax.set_title(
            f"Raw hν map  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV",
            color="white", fontsize=10,
        )
        self._colorbar(artist, ax)
        diag = result.diagnostics
        summary = dataset_summary(self._dataset)
        return (
            f"{diag['n_scans']} scans | hν={diag['hv_min']:.1f}→{diag['hv_max']:.1f} eV | "
            f"raw (no kz conversion) | {self._format_hv_sources(summary['hv_sources'])}"
        )

    def _draw_kz_view(self, ax, params: KzParams, ui) -> str:
        result = compute_kz_map(self._dataset.scans, params)
        diag = result.diagnostics
        artist = ax.pcolormesh(
            result.k_grid, result.kz_grid, result.image,
            shading="auto", cmap="inferno", vmin=0.0, vmax=max(self._vmax(result.image), 1e-12),
        )
        if ui.show_points:
            ax.scatter(diag["point_k"], diag["point_kz"], c="#66ccff",
                       s=3, alpha=0.35, linewidths=0)
        coverage = self._draw_kz_planes(ax, result, params) if ui.show_planes else None
        profile = self._draw_kz_profile(ax, result, params) if ui.show_profile else None
        ax.set_xlabel("k// (π/a)", color="white")
        ax.set_ylabel(f"kz ({ui.kz_unit})", color="white")
        ax.set_title(
            f"KZ  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV  V0={ui.inner_potential:.1f} eV",
            color="white", fontsize=10,
        )
        self._colorbar(artist, ax)
        summary = dataset_summary(self._dataset)
        info = (
            f"{diag['n_scans']} scans | hν={summary['hv_min']:.1f}→{summary['hv_max']:.1f} eV | "
            f"points={diag['n_points']} | bins={diag['n_bins_filled']} | "
            f"{diag['interpolation_backend']}"
        )
        if coverage:
            info += f" | {coverage}"
        if profile:
            info += f" | {profile}"
        return info + f" | {self._format_hv_sources(summary['hv_sources'])}"

    def _draw_kz_profile(self, ax, result, params: KzParams) -> str:
        """Overlay the normal-emission I(kz) curve on the left margin of the map."""
        try:
            prof = kz_profile_at_normal_emission(self._dataset.scans, params)
        except Exception:
            return ""
        kz = prof["kz"]
        inten = prof["intensity"]
        if kz.size < 2:
            return ""
        y = convert_kz_unit(kz, unit=params.kz_unit, c_lattice=params.c_lattice)
        k0 = float(result.k_grid[0])
        kspan = float(result.k_grid[-1] - result.k_grid[0]) or 1.0
        lo, hi = float(np.nanmin(inten)), float(np.nanmax(inten))
        rng = (hi - lo) or 1.0
        x = k0 + ((inten - lo) / rng) * 0.3 * kspan
        ax.plot(x, y, color="white", lw=1.2, alpha=0.9)
        c_imp = prof.get("c_implied", float("nan"))
        return f"I(kz)@k//0 → c≈{c_imp:.1f} Å" if np.isfinite(c_imp) else "I(kz)@k//0"

    def _fit_kz_v0(self):
        if self._dataset is None:
            self._status("Warning: KZ: load a folder first")
            return
        ui = self._parent._kz_controls.params()
        params = KzParams(
            work_func=self._work_func(),
            inner_potential=ui.inner_potential,
            a_lattice=ui.a_lattice,
            c_lattice=ui.c_lattice,
            energy_center=ui.energy_center,
            energy_window=ui.energy_window,
            kz_unit=ui.kz_unit,
            normalize=ui.normalize,
        )
        if params.c_lattice <= 0:
            self._status("Warning: KZ: set lattice c before fitting V0")
            return
        try:
            res = fit_inner_potential(self._dataset.scans, params)
        except Exception as exc:
            self._status(f"Warning: KZ fit V0: {exc}")
            return
        if res["confidence"] == "low":
            # Don't overwrite V0 with an unconstrained value: tell the user why.
            reason = "rails to range edge" if res["boundary"] else "no clear periodicity"
            self._status(
                f"Warning: KZ: V0 not determined ({reason}; power={res['power']:.2f}, "
                f"{res['n_zones']:.1f} zones). Need a wider hν range or set V0 manually."
            )
            return
        sp = self._parent._kz_controls.sp_v0
        sp.blockSignals(True)
        sp.setValue(float(res["v0_best"]))
        sp.blockSignals(False)
        sig = res["v0_sigma"]
        sig_txt = f"±{sig:.2f}" if np.isfinite(sig) else "±?"
        phase = res["cluster_phase"]
        plane = "Γ" if (phase < 0.25 or phase > 0.75) else "Z"
        self._status(
            f"V0 = {res['v0_best']:.2f} {sig_txt} eV (power={res['power']:.2f}, "
            f"{res['n_zones']:.1f} zones, maxima@{plane})"
        )
        self._draw_kz_tab()

    def _draw_kz_planes(self, ax, result, params: KzParams) -> str:
        kz_lo = float(kz_unit_to_inv_a(
            float(np.nanmin(result.kz_grid)), unit=params.kz_unit, c_lattice=params.c_lattice))
        kz_hi = float(kz_unit_to_inv_a(
            float(np.nanmax(result.kz_grid)), unit=params.kz_unit, c_lattice=params.c_lattice))
        for plane in kz_high_symmetry_planes(kz_lo, kz_hi, params.c_lattice, unit=params.kz_unit):
            is_gamma = plane["label"] == "Γ"
            ax.axhline(plane["kz"], color="#dddddd", lw=1.0,
                       ls="-" if is_gamma else "--", alpha=0.7)
            ax.text(result.k_grid[-1], plane["kz"], f" {plane['label']}",
                    color="#dddddd", fontsize=8, va="center", ha="left")
        cov = kz_coverage_summary(
            kz_lo, kz_hi, params.c_lattice,
            work_func=params.work_func,
            inner_potential=params.inner_potential,
            energy=params.energy_center,
        )
        g = ",".join(f"{h:.0f}" for h in cov["gamma_hv"]) or "—"
        z = ",".join(f"{h:.0f}" for h in cov["z_hv"]) or "—"
        return f"{cov['n_zones']:.1f} zones | Γ@hν≈{g} · Z@hν≈{z} eV"

    def _save_kz_session(self):
        # MVP: only UI parameters remain in the widgets.
        self._status("KZ: nothing to save for now")
