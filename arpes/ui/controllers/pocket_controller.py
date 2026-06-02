"""FS pocket characterization controller."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QFileDialog

from arpes.physics.bz import bz_high_symmetry_points, bz_polygon
from arpes.physics.fs import extract_fs_map
from arpes.io.dft_grid import load_dft_grid_npz
from arpes.physics.dft_slice import (
    isocontour_at_energy,
    kz_from_hv,
    slice_grid_at_kz,
)
from arpes.physics.pocket import (
    characterize_pocket,
    characterize_pocket_bootstrap,
    extract_fs_contour,
    simplify_closed_contour,
    smooth_closed_contour,
    smooth_fs_image,
)
from arpes.physics.pocket_compare import compare_pocket_contours
from arpes.ui.widgets.dialogs.pocket_result import PocketResultDialog


class PocketController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_preview_seed_raw", None)
        object.__setattr__(self, "_preview_seed_plot", None)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _pocket_action(self, verb: str, payload: dict | None = None):
        payload = payload or {}
        if verb == "characterize":
            return self._characterize_at(payload)
        if verb == "preview_start":
            return self._preview_start(payload)
        if verb == "preview_update":
            return self._preview_update(payload)
        if verb == "preview_validate":
            return self._preview_validate()
        if verb == "preview_cancel":
            return self._preview_cancel()
        if verb == "show":
            return self._show_pocket(payload)
        if verb == "delete":
            return self._delete_pocket(payload)
        if verb == "clear":
            return self._clear_pockets()
        if verb == "export_csv":
            return self._export_csv(payload)
        if verb == "load_dft":
            return self._load_dft(payload)
        if verb == "clear_dft":
            return self._clear_dft()
        raise ValueError(f"pocket action inconnue: {verb}")

    def _load_dft(self, payload: dict):
        entry = self._current_entry()
        if entry is None:
            self._status("DFT : aucun fichier courant.")
            return None
        path = payload.get("path")
        if not path:
            path, _filter = QFileDialog.getOpenFileName(
                self._parent,
                "Charger grille DFT 3D (npz)",
                str(getattr(self._session, "folder", "") or ""),
                "DFT 3D (*.npz)",
            )
        if not path:
            return None
        try:
            params = self._fs_controls.params()
            grid = load_dft_grid_npz(path, a_lattice_fallback=float(params.a_lattice))
        except Exception as exc:
            self._status(f"DFT : {exc}")
            return None
        entry.dft_grid_path = str(path)
        self._session.save()
        if hasattr(self._fs_controls, "set_dft_status"):
            self._fs_controls.set_dft_status(Path(path).name)
        self._status(
            f"DFT chargé : {Path(path).name} | "
            f"a={grid.a_lattice:.3f} Å, grid={grid.energies.shape}."
        )
        return str(path)

    def _clear_dft(self):
        entry = self._current_entry()
        if entry is None:
            return None
        entry.dft_grid_path = ""
        self._session.save()
        if hasattr(self._fs_controls, "set_dft_status"):
            self._fs_controls.set_dft_status("")
        self._status("DFT : grille oubliée.")
        return None

    def _attach_dft_compare(self, pocket: dict, entry, params) -> None:
        path = str(getattr(entry, "dft_grid_path", "") or "")
        if not path or not Path(path).exists():
            return
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        hv = meta.get("photon_energy") or meta.get("hv")
        if hv is None:
            pocket["dft_compare_error"] = "DFT : pas de hν dans metadata."
            return
        try:
            grid = load_dft_grid_npz(path, a_lattice_fallback=float(params.a_lattice))
            kz = kz_from_hv(float(hv), float(params.v0_eV))
            slice_ = slice_grid_at_kz(grid.kx, grid.ky, grid.kz, grid.energies, kz)
            seed_1_per_ang = (
                float(pocket["centroid_kx"]) * float(np.pi) / float(params.a_lattice),
                float(pocket["centroid_ky"]) * float(np.pi) / float(params.a_lattice),
            )
            contour_dft_1_per_ang = isocontour_at_energy(
                slice_, energy_eV=0.0, seed_point_1_per_ang=seed_1_per_ang,
            )
            scale = float(params.a_lattice) / float(np.pi)
            contour_dft_pi_a = contour_dft_1_per_ang * scale
            contour_exp = np.asarray(pocket.get("contour") or [], dtype=float)
            res = compare_pocket_contours(contour_exp, contour_dft_pi_a)
            d = res.asdict()
            d["kz_used_1_per_ang"] = float(slice_.kz_used)
            d["dft_path"] = path
            pocket["dft_compare"] = d
        except Exception as exc:
            pocket["dft_compare_error"] = f"DFT : {exc}"

    def _preview_start(self, payload: dict):
        if self._raw_data is None or not self._current_is_fs():
            self._status("Aperçu poche : charge une carte FS d'abord.")
            return None
        try:
            params = self._fs_controls.params()
            settings = self._pocket_settings()
            kx, ky, fs, _ = extract_fs_map(self._raw_data, params)
            seed_plot = (float(payload["kx"]), float(payload["ky"]))
            seed_raw = (
                seed_plot[0] + float(params.kx_center),
                seed_plot[1] + float(params.ky_center),
            )
            fs_pocket = smooth_fs_image(
                fs,
                sigma=(settings["smooth_sigma_y"], settings["smooth_sigma_x"]),
            )
            level = self._auto_level(fs_pocket, seed_raw, kx, ky)
            self._fs_controls.sp_pocket_level.blockSignals(True)
            self._fs_controls.sp_pocket_level.setValue(float(level))
            self._fs_controls.sp_pocket_level.blockSignals(False)
            self._fs_controls.chk_pocket_level_manual.setChecked(True)
            object.__setattr__(self, "_preview_seed_raw", seed_raw)
            object.__setattr__(self, "_preview_seed_plot", seed_plot)
            self._draw_preview_at(level)
            self._status(
                f"Aperçu poche : ajuste le slider Level (auto={level:.3f}). "
                "Clic droit → Valider ou Annuler."
            )
        except Exception as exc:
            self._status(f"Aperçu poche : {exc}")
        return None

    def _preview_update(self, payload: dict):
        if self._preview_seed_raw is None:
            return None
        try:
            level = float(payload.get("level", self._fs_controls.sp_pocket_level.value()))
            self._draw_preview_at(level)
        except Exception as exc:
            self._status(f"Aperçu poche : {exc}")
        return None

    def _preview_cancel(self):
        object.__setattr__(self, "_preview_seed_raw", None)
        object.__setattr__(self, "_preview_seed_plot", None)
        if hasattr(self._fs_canvas, "clear_pocket_preview"):
            self._fs_canvas.clear_pocket_preview()
        self._status("Aperçu poche annulé.")
        return None

    def _preview_validate(self):
        if self._preview_seed_plot is None:
            return None
        seed = self._preview_seed_plot
        level = float(self._fs_controls.sp_pocket_level.value())
        if hasattr(self._fs_canvas, "clear_pocket_preview"):
            self._fs_canvas.clear_pocket_preview()
        object.__setattr__(self, "_preview_seed_raw", None)
        object.__setattr__(self, "_preview_seed_plot", None)
        return self._characterize_at({"kx": seed[0], "ky": seed[1], "level": level})

    def _draw_preview_at(self, level: float) -> None:
        if self._preview_seed_raw is None or not hasattr(self._fs_canvas, "draw_pocket_preview"):
            return
        params = self._fs_controls.params()
        settings = self._pocket_settings()
        kx, ky, fs, _ = extract_fs_map(self._raw_data, params)
        fs_pocket = smooth_fs_image(
            fs,
            sigma=(settings["smooth_sigma_y"], settings["smooth_sigma_x"]),
        )
        try:
            raw_contour = extract_fs_contour(
                fs_pocket, kx, ky, level, seed_point=self._preview_seed_raw,
            )
        except ValueError:
            self._fs_canvas.clear_pocket_preview()
            return
        contour = simplify_closed_contour(
            smooth_closed_contour(raw_contour, window=int(settings["contour_window"])),
            min_step=float(settings["simplify_step"]),
        )
        shifted = np.asarray(contour, dtype=float).copy()
        shifted[:, 0] -= float(params.kx_center)
        shifted[:, 1] -= float(params.ky_center)
        self._fs_canvas.draw_pocket_preview(shifted)

    def _characterize_at(self, payload: dict):
        if self._raw_data is None or not self._current_is_fs():
            self._status("Poche FS : charge une carte FS d'abord.")
            return None
        if not hasattr(self, "_fs_controls"):
            return None
        entry = self._current_entry()
        if entry is None:
            return None
        try:
            seed_plot = (
                float(payload["kx"]),
                float(payload["ky"]),
            )
            params = self._fs_controls.params()
            settings = self._pocket_settings()
            seed_raw = (
                seed_plot[0] + float(params.kx_center),
                seed_plot[1] + float(params.ky_center),
            )
            kx, ky, fs, _ = extract_fs_map(self._raw_data, params)
            sigma = (settings["smooth_sigma_y"], settings["smooth_sigma_x"])
            fs_pocket = smooth_fs_image(fs, sigma=sigma)
            level_source = payload.get("level", None)
            if level_source is None:
                level_source = settings.get("level", None)
            level = float(level_source if level_source is not None else self._auto_level(fs_pocket, seed_raw, kx, ky))
            bz_raw = self._bz_polygon_raw(params)
            hs_raw = self._hs_points_raw(params)
            char_kwargs = dict(
                contour_window=int(settings["contour_window"]),
                n_bands=int(settings.get("n_bands", 1)),
                spin=int(settings.get("spin", 2)),
                hs_dir_x_deg=float(settings.get("hs_dir_x_deg", 0.0)),
                hs_dir_m_deg=float(settings.get("hs_dir_m_deg", 45.0)),
                hs_dir_tol_deg=float(settings.get("hs_dir_tol_deg", 10.0)),
            )
            if bool(settings.get("bootstrap", False)):
                bs = characterize_pocket_bootstrap(
                    fs, kx, ky,
                    seed_point=seed_raw,
                    level=level,
                    bz_polygon=bz_raw,
                    hs_points=hs_raw,
                    smooth_sigma=sigma,
                    n_bootstrap=int(settings.get("bootstrap_n", 20)),
                    **char_kwargs,
                )
                pocket = bs.asdict()
            else:
                props = characterize_pocket(
                    fs_pocket, kx, ky,
                    seed_point=seed_raw,
                    level=level,
                    bz_polygon=bz_raw,
                    hs_points=hs_raw,
                    **char_kwargs,
                )
                pocket = props.asdict()
            pocket["level"] = level
            pocket["contour"] = self._contour_for_storage(fs_pocket, kx, ky, level, seed_raw, params)
            pocket["processing"] = {
                "quality": settings.get("quality", "Standard"),
                "smooth_sigma_yx": [settings["smooth_sigma_y"], settings["smooth_sigma_x"]],
                "contour_window": int(settings["contour_window"]),
                "simplify_step": settings["simplify_step"],
                "min_area_pct_bz": settings["min_area_pct_bz"],
            }
            mp_label = self._mp_label_for(pocket)
            if mp_label:
                pocket["mp_label"] = mp_label
            self._attach_dft_compare(pocket, entry, params)
            if float(pocket.get("area_pct_bz", 0.0) or 0.0) < settings["min_area_pct_bz"]:
                self._status(
                    f"Poche FS rejetée : aire {pocket['area_pct_bz']:.2f}% BZ "
                    f"< min {settings['min_area_pct_bz']:.2f}%."
                )
                return None
            entry.fs_pockets = list(getattr(entry, "fs_pockets", []) or []) + [pocket]
            self._session.save()
            self._draw_fs_tab()
            idx = len(entry.fs_pockets) - 1
            dialog = PocketResultDialog(self._parent, pocket, allow_delete=True)
            dialog.exec()
            if getattr(dialog, "delete_requested", False):
                self._delete_pocket({"index": idx})
                return None
            self._status(
                f"Poche FS : {pocket['hs_label_nearest'] or '?'} "
                f"{pocket['area_pct_bz']:.2f}% BZ, {pocket['topology']}."
            )
            return pocket
        except Exception as exc:
            self._status(f"Poche FS : {exc}")
            return None

    def _show_pocket(self, payload: dict):
        entry = self._current_entry()
        if entry is None:
            return None
        pockets = list(getattr(entry, "fs_pockets", []) or [])
        idx = int(payload.get("index", -1))
        if idx < 0 or idx >= len(pockets):
            return None
        pocket = pockets[idx]
        dialog = PocketResultDialog(self._parent, pocket, allow_delete=True)
        dialog.exec()
        if getattr(dialog, "delete_requested", False):
            self._delete_pocket({"index": idx})
            return None
        return pocket

    def _delete_pocket(self, payload: dict):
        entry = self._current_entry()
        if entry is None:
            return None
        pockets = list(getattr(entry, "fs_pockets", []) or [])
        idx = int(payload.get("index", -1))
        if idx < 0 or idx >= len(pockets):
            return None
        removed = pockets.pop(idx)
        entry.fs_pockets = pockets
        self._session.save()
        self._draw_fs_tab()
        self._status(f"Poche FS supprimée : {removed.get('hs_label_nearest') or idx + 1}.")
        return removed

    def _clear_pockets(self):
        entry = self._current_entry()
        if entry is None:
            return
        entry.fs_pockets = []
        self._session.save()
        self._draw_fs_tab()
        self._status("Poches FS effacées pour ce fichier.")

    def _export_csv(self, payload: dict | None = None):
        payload = payload or {}
        entry = self._current_entry()
        if entry is None:
            return None
        pockets = list(getattr(entry, "fs_pockets", []) or [])
        if not pockets:
            self._status("Export poches FS : aucune poche.")
            return None
        out_path = payload.get("path")
        if not out_path:
            default = self._default_export_path()
            out_path, _filter = QFileDialog.getSaveFileName(
                self._parent,
                "Exporter poches FS",
                str(default),
                "CSV (*.csv)",
            )
        if not out_path:
            return None
        path = Path(out_path)
        rows = [self._pocket_export_row(i, p) for i, p in enumerate(pockets, start=1)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self._status(f"Poches FS exportées : {path}")
        return path

    def _default_export_path(self) -> Path:
        folder = getattr(self._session, "folder", None) or Path.cwd()
        stem = "fs"
        if getattr(self, "_current_path", None):
            stem = Path(self._current_path).stem or "fs"
        return Path(folder) / f"{stem}_pockets.csv"

    def _pocket_export_row(self, index: int, pocket: dict) -> dict:
        keys = [
            "centroid_kx", "centroid_ky", "area_inv_a2", "area_pct_bz",
            "kF_mean", "kF_a", "kF_b", "ellipse_angle_deg",
            "kF_gamma_x", "kF_gamma_m", "aspect_ratio", "eccentricity",
            "curvature_mean", "curvature_var", "n_carriers_2D",
            "topology", "topology_confidence", "topology_rays_used",
            "hs_label_nearest", "hs_distance", "level", "mp_label",
        ]
        row = {"index": index}
        row.update({k: pocket.get(k, "") for k in keys})
        return row

    def _mp_label_for(self, pocket: dict) -> str:
        entry = self._current_entry()
        lat = getattr(entry, "fs_lattice", {}) if entry is not None else {}
        mp_id = str((lat or {}).get("mp_id", "") or "").strip()
        hs = str(pocket.get("hs_label_nearest") or "").strip()
        if not (mp_id and hs):
            return ""
        return f"{mp_id}:{hs}"

    def _auto_level(self, fs: np.ndarray, seed_raw, kx: np.ndarray, ky: np.ndarray) -> float:
        fs = np.asarray(fs, dtype=float)
        finite = fs[np.isfinite(fs)]
        if finite.size == 0:
            raise ValueError("image FS vide.")
        seed_i = self._nearest_value(fs, kx, ky, seed_raw)
        med = float(np.nanmedian(finite))
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if np.isfinite(seed_i) and seed_i > med:
            return float(med + 0.5 * (seed_i - med))
        return float(lo + 0.5 * (med - lo))

    def _pocket_settings(self) -> dict[str, float | int | str | None]:
        defaults = {
            "smooth_sigma_y": 1.0,
            "smooth_sigma_x": 3.0,
            "contour_window": 9,
            "simplify_step": 0.015,
            "min_area_pct_bz": 0.20,
            "quality": "Standard",
            "level": None,
            "n_bands": 1,
            "spin": 2,
            "hs_dir_x_deg": 0.0,
            "hs_dir_m_deg": 45.0,
            "hs_dir_tol_deg": 10.0,
        }
        controls = getattr(self, "_fs_controls", None)
        if controls is None or not hasattr(controls, "pocket_settings"):
            return defaults
        raw = controls.pocket_settings()
        return {
            "smooth_sigma_y": float(raw.get("smooth_sigma_y", defaults["smooth_sigma_y"])),
            "smooth_sigma_x": float(raw.get("smooth_sigma_x", defaults["smooth_sigma_x"])),
            "contour_window": int(raw.get("contour_window", defaults["contour_window"])),
            "simplify_step": float(raw.get("simplify_step", defaults["simplify_step"])),
            "min_area_pct_bz": float(raw.get("min_area_pct_bz", defaults["min_area_pct_bz"])),
            "quality": str(raw.get("quality", defaults["quality"]) or defaults["quality"]),
            "level": (None if raw.get("level", None) is None else float(raw.get("level"))),
            "n_bands": int(raw.get("n_bands", defaults["n_bands"])),
            "spin": int(raw.get("spin", defaults["spin"])),
            "hs_dir_x_deg": float(raw.get("hs_dir_x_deg", defaults["hs_dir_x_deg"])),
            "hs_dir_m_deg": float(raw.get("hs_dir_m_deg", defaults["hs_dir_m_deg"])),
            "hs_dir_tol_deg": float(raw.get("hs_dir_tol_deg", defaults["hs_dir_tol_deg"])),
            "bootstrap": bool(raw.get("bootstrap", False)),
            "bootstrap_n": int(raw.get("bootstrap_n", 20)),
        }

    def _nearest_value(self, fs, kx, ky, point) -> float:
        ix = int(np.argmin(np.abs(np.asarray(kx) - float(point[0]))))
        iy = int(np.argmin(np.abs(np.asarray(ky) - float(point[1]))))
        return float(np.asarray(fs)[iy, ix])

    def _bz_polygon_raw(self, params):
        poly = bz_polygon(
            params.bz_shape,
            float(params.bz_half_x),
            float(params.bz_half_y),
            float(params.bz_angle_deg),
        )
        offset = np.array([float(params.kx_center), float(params.ky_center)])
        return np.asarray(poly, dtype=float) + offset

    def _hs_points_raw(self, params) -> dict[str, tuple[float, float]]:
        offset = np.array([float(params.kx_center), float(params.ky_center)])
        out: dict[str, tuple[float, float]] = {}
        for x, y, name, _color in bz_high_symmetry_points(
            params.bz_shape,
            float(params.bz_half_x),
            float(params.bz_half_y),
            float(params.bz_angle_deg),
        ):
            p = np.array([float(x), float(y)]) + offset
            out[str(name)] = (float(p[0]), float(p[1]))
        return out

    def _contour_for_storage(self, fs, kx, ky, level, seed_raw, params):
        from arpes.physics.pocket import extract_fs_contour

        contour = extract_fs_contour(fs, kx, ky, level, seed_point=seed_raw)
        settings = self._pocket_settings()
        contour = simplify_closed_contour(
            smooth_closed_contour(contour, window=int(settings["contour_window"])),
            min_step=float(settings["simplify_step"]),
        )
        shifted = np.asarray(contour, dtype=float).copy()
        shifted[:, 0] -= float(params.kx_center)
        shifted[:, 1] -= float(params.ky_center)
        return shifted.tolist()
