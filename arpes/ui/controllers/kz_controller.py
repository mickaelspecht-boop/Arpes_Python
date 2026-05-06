"""Controller Qt pour l'onglet KZ."""
from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from arpes.io.kz_dataset import KzDataset, dataset_summary, load_kz_stack
from arpes.physics.kz import (
    KzParams,
    compute_hv_k_map,
    compute_kz_map,
    compute_mdc_waterfall,
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
        folder = QFileDialog.getExistingDirectory(self._parent, "Choisir dossier KZ", start or str(Path.home()))
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
            "Logbook (*.xlsx *.xls *.csv *.tsv);;Tous les fichiers (*)",
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
            self._status(f"Logbook KZ chargé : {Path(path).name}{sheet_txt} | {len(records)} lignes | {used}")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"logbook KZ chargé ({len(records)} lignes)")
            QMessageBox.information(
                self._parent,
                "Logbook KZ chargé",
                f"{Path(path).name}{sheet_txt}\n{len(records)} lignes lues.\n\nColonnes détectées :\n{used}",
            )
            if self._last_folder:
                self._refresh_kz_dataset()
        except Exception as exc:
            QMessageBox.warning(self._parent, "Logbook KZ", str(exc))
            self._status(f"Attention: Logbook KZ : {exc}")

    @staticmethod
    def _format_hv_sources(sources: dict) -> str:
        if not sources:
            return "hν source: unknown"
        bits = [f"{key}={value}" for key, value in sorted(sources.items())]
        return "hν source: " + ", ".join(bits)

    def _refresh_kz_dataset(self):
        if not self._last_folder:
            self._status("Attention: KZ : choisir un dossier")
            return
        try:
            ds = load_kz_stack(
                self._last_folder,
                work_func=float(self._params.sp_phi.value()),
                ef_offset=float(self._params.sp_ef.value()),
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
                f"{summary['n_scans']} scans retenus | {n_ignored} ignorés | "
                f"hν={summary['hv_min']:.1f}→{summary['hv_max']:.1f} eV | "
                f"{self._format_hv_sources(summary['hv_sources'])}"
            )
            self._status(f"OK KZ : {summary['n_scans']} scans chargés")
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"KZ chargé ({summary['n_scans']} scans)")
            self._draw_kz_tab()
        except Exception as exc:
            self._dataset = None
            self._parent._kz_controls.set_info(f"Erreur KZ : {exc}")
            self._status(f"Attention: KZ : {exc}")
            traceback.print_exc()

    def _on_kz_params_changed(self, _=None):
        self._draw_kz_tab()

    def _draw_kz_tab(self):
        if not hasattr(self._parent, "_kz_canvas"):
            return
        while len(self._parent._kz_canvas.map.fig.axes) > 1:
            self._parent._kz_canvas.map.fig.delaxes(self._parent._kz_canvas.map.fig.axes[-1])
        ax = self._parent._kz_canvas.map.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        if self._dataset is None:
            ax.text(
                0.5, 0.5,
                "Choisir un dossier KZ\n(série de band maps à hν variable)",
                transform=ax.transAxes,
                ha="center", va="center",
                color="white", fontsize=11,
            )
            self._parent._kz_canvas.map.redraw()
            return

        ui = self._parent._kz_controls.params()
        params = KzParams(
            work_func=float(self._params.sp_phi.value()),
            inner_potential=ui.inner_potential,
            a_lattice=ui.a_lattice,
            c_lattice=ui.c_lattice,
            energy_center=ui.energy_center,
            energy_window=ui.energy_window,
            k_bins=ui.k_bins,
            kz_bins=ui.kz_bins,
            kz_unit=ui.kz_unit,
            normalize=ui.normalize,
            display_mode=ui.display_mode,
        )
        try:
            if ui.display_mode == "hv map":
                result = compute_hv_k_map(self._dataset.scans, params)
            elif ui.display_mode == "MDC waterfall":
                result = compute_mdc_waterfall(self._dataset.scans, params)
            else:
                result = compute_kz_map(self._dataset.scans, params)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes,
                    ha="center", va="center", color="tomato", fontsize=10)
            self._parent._kz_canvas.map.redraw()
            self._status(f"Attention: KZ : {exc}")
            return

        diag = result.diagnostics
        if ui.display_mode == "MDC waterfall":
            curves = result.curves
            finite = curves[np.isfinite(curves)]
            vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        else:
            img = result.image
            finite = img[np.isfinite(img)]
            vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        if ui.display_mode == "hv map":
            artist = ax.pcolormesh(
                result.k_grid, result.hv_grid, img,
                shading="auto", cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-12),
            )
            ax.set_xlabel("k// (π/a)", color="white")
            ax.set_ylabel("hν (eV)", color="white")
            ax.set_title(
                f"Raw hν map  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV",
                color="white", fontsize=10,
            )
        elif ui.display_mode == "MDC waterfall":
            artist = None
            for idx, (hv, offset) in enumerate(zip(result.hv_grid, result.offsets)):
                line, = ax.plot(result.k_grid, result.curves[idx] + offset, lw=1.0)
                artist = line
                ax.text(
                    result.k_grid[-1], offset, f"{hv:.0f} eV",
                    color=line.get_color(), fontsize=7, va="bottom", ha="left",
                )
            ax.set_xlabel("k// (π/a)", color="white")
            ax.set_ylabel("MDC intensity + offset", color="white")
            ax.set_title(
                f"MDC waterfall  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV",
                color="white", fontsize=10,
            )
        elif ui.display_mode == "points":
            vals = np.asarray(diag["point_i"], dtype=float)
            vmax = float(np.nanpercentile(vals[np.isfinite(vals)], 99)) if np.isfinite(vals).any() else 1.0
            artist = ax.scatter(
                diag["point_k"], diag["point_kz"], c=vals,
                s=5, cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-12), linewidths=0,
            )
            ax.set_xlabel("k// (π/a)", color="white")
            ax.set_ylabel(f"kz ({ui.kz_unit})", color="white")
            ax.set_title(
                f"KZ points  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV  "
                f"V0={ui.inner_potential:.1f} eV",
                color="white", fontsize=10,
            )
        else:
            artist = ax.pcolormesh(
                result.k_grid, result.kz_grid, img,
                shading="auto", cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-12),
            )
            ax.set_xlabel("k// (π/a)", color="white")
            ax.set_ylabel(f"kz ({ui.kz_unit})", color="white")
            ax.set_title(
                f"KZ  E={ui.energy_center:+.3f}±{ui.energy_window:.3f} eV  "
                f"V0={ui.inner_potential:.1f} eV",
                color="white", fontsize=10,
            )
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")
        try:
            if artist is not None and ui.display_mode != "MDC waterfall":
                self._parent._kz_canvas.map.fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            pass
        summary = dataset_summary(self._dataset)
        hv_source_txt = self._format_hv_sources(summary["hv_sources"])
        if ui.display_mode == "hv map":
            info = (
                f"{diag['n_scans']} scans retenus | {len(summary['warnings'])} ignorés | "
                f"hν={diag['hv_min']:.1f}→{diag['hv_max']:.1f} eV | "
                f"points={diag['n_points']} | raw hν map (no kz conversion) | {hv_source_txt}"
            )
        elif ui.display_mode == "MDC waterfall":
            info = (
                f"{diag['n_scans']} scans retenus | {len(summary['warnings'])} ignorés | "
                f"hν={diag['hv_min']:.1f}→{diag['hv_max']:.1f} eV | "
                f"points={diag['n_points']} | MDC waterfall (no fits) | {hv_source_txt}"
            )
        else:
            info = (
                f"{diag['n_scans']} scans retenus | {len(summary['warnings'])} ignorés | "
                f"hν={summary['hv_min']:.1f}→{summary['hv_max']:.1f} eV | "
                f"points={diag['n_points']} | bins={diag['n_bins_filled']} | "
                f"{diag['display_mode']} ({diag['interpolation_backend']}) | {hv_source_txt}"
            )
        self._parent._kz_controls.set_info(info)
        self._parent._kz_canvas.map.fig.tight_layout(pad=0.8)
        self._parent._kz_canvas.map.redraw()

    def _save_kz_session(self):
        # MVP: seuls les paramètres UI restent dans les widgets.
        self._status("KZ : rien à sauvegarder pour l'instant")
