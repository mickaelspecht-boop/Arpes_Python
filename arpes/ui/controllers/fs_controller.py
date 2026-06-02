"""Fermi-surface UI controller for ArpesExplorer."""
from __future__ import annotations

import numpy as np

from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas, FSControlPanel
from arpes.physics.kz import kz_from_hv_kpar, fold_kz_to_1bz


class FSController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _current_is_fs(self) -> bool:
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        return meta.get("fs_data") is not None

    def _on_fs_params_changed(self):
        self._save_current_fs_center()
        self._draw_fs_tab()

    def _schedule_fs_redraw(self, _=None):
        timer = getattr(self._parent, "_fs_redraw_timer", None)
        if timer is not None:
            timer.start(150)
            return
        self._on_fs_params_changed()

    def _choose_bz_preset(self):
        if not hasattr(self, "_fs_controls"):
            return
        from arpes.ui.widgets.dialogs import BZSelectorDialog
        dialog = BZSelectorDialog(self._parent)
        if dialog.exec():
            self._fs_controls.apply_bz_preset(dialog.selected_key)
            self._draw_fs_tab()
            self._status(f"ZDB appliquée : {dialog.selected_key}")

    def _save_current_fs_center(self):
        if self._raw_data is None or not self._current_path or not self._current_is_fs():
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        entry = self._current_entry()
        if entry is None:
            return
        try:
            p = self._fs_controls.params()
            entry.fs_center_kx = float(p.kx_center)
            entry.fs_center_ky = float(p.ky_center)
            self._session.save()
        except Exception:
            pass

    def _draw_fs_tab(self):
        if not hasattr(self, "_fs_canvas") or FermiSurfaceCanvas is None:
            return
        if not hasattr(self, "_fs_controls") or FSControlPanel is None:
            return
        # Injecte lattice MP de l'entry dans metadata avant draw (canvas le lit).
        self._inject_fs_lattice_into_raw()
        # GF redteam : applique distortion BM au volume FS si opt-in actif.
        propagated = self._apply_distortion_to_fs_volume_if_enabled()
        fs_params = self._fs_controls.params()
        info = self._fs_canvas.draw_fs(self._raw_data, fs_params)
        entry = self._current_entry()
        if entry is not None and hasattr(self._fs_canvas, "draw_pockets"):
            pockets = getattr(entry, "fs_pockets", []) or []
            self._fs_canvas.draw_pockets(pockets)
            if hasattr(self._fs_controls, "set_pocket_count"):
                self._fs_controls.set_pocket_count(len(pockets))
        elif hasattr(self._fs_controls, "set_pocket_count"):
            self._fs_controls.set_pocket_count(0)
        try:
            self._fs_controls.lbl_info.setText(info)
        except Exception:
            pass
        self._update_fs_kz_label()
        # Badge orange visible si propagation FS active.
        self._draw_fs_distortion_badge(propagated)
        # B.4 — overlay BM cuts si toggle actif (after main draw).
        cuts_collected: list = []
        if getattr(self, "_show_bm_cuts", False) and self._current_is_fs():
            try:
                cuts_collected = self._pairing_action("collect_cuts", {
                    "fs_metadata": (self._raw_data or {}).get("metadata", {}),
                    "a_lattice": fs_params.a_lattice,
                }) or []
                self._fs_canvas.draw_bm_cuts(cuts_collected)
            except Exception as exc:
                self._status(f"Attention: BM cuts overlay : {exc}")
        # A.5 — refresh liste « BMs reliées » (toujours, indépendant du toggle).
        if hasattr(self, "_fs_linked_bms"):
            try:
                cuts_for_list = cuts_collected or (
                    self._pairing_action("collect_cuts", {
                        "fs_metadata": (self._raw_data or {}).get("metadata", {}),
                        "a_lattice": fs_params.a_lattice,
                    }) or []
                )
                self._fs_linked_bms.refresh(self._pairing_action("active_fs"),
                                            cuts_for_list)
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  Overlay BZ cristal (Materials Project)
    # ------------------------------------------------------------------

    def _on_bz_crystal_overlay_changed(self):
        if not self._check_bz_crystal_consistency():
            return
        self._save_current_bz_crystal_settings()
        self._draw_fs_tab()

    def _on_mp_lattice_fetch(self):
        from arpes.theory.materials_project import (
            MaterialsProjectUnavailable, load_lattice,
        )
        if not hasattr(self, "_fs_controls"):
            return
        mp_id = self._fs_controls.ed_mp_id.text().strip()
        if not mp_id:
            entry = self._current_entry()
            if entry is not None:
                mp_id = str(getattr(entry.meta, "mp_id", "") or "").strip()
                if mp_id:
                    self._fs_controls.ed_mp_id.setText(mp_id)
        if not mp_id:
            self._status("✗ MP : saisir mp-xxxx ou charger un fichier avec mp_id dans le logbook.")
            return
        cache_dir = None
        try:
            if getattr(self, "_session", None) is not None and self._session.folder:
                cache_dir = self._session.folder / ".arpes_theory_cache"
        except Exception:
            cache_dir = None
        try:
            lat = load_lattice(mp_id, cache_dir=cache_dir)
        except MaterialsProjectUnavailable as exc:
            self._status(f"✗ MP : {exc}")
            return
        except Exception as exc:
            self._status(f"✗ MP : échec {exc}")
            return
        lat_dict = {
            "a": lat.a, "b": lat.b, "c": lat.c,
            "alpha_deg": lat.alpha_deg, "beta_deg": lat.beta_deg,
            "gamma_deg": lat.gamma_deg,
            "bravais": lat.bravais, "space_group": lat.space_group,
            "mp_id": lat.mp_id,
        }
        entry = self._current_entry()
        if entry is not None:
            entry.fs_lattice = lat_dict
            try:
                self._session.save()
            except Exception:
                pass
        # GF3 redteam : avertir si lattice ARPES (FSParams.a) ≠ lattice MP > 2%
        try:
            a_ui = float(self._fs_controls.sp_a.value())
            rel = abs(a_ui - lat.a) / max(lat.a, 1e-6)
            if rel > 0.02:
                if entry is not None:
                    entry.fs_bz_crystal_force_override = False
                self._status(
                    f"⚠ BZ MP désactivée tant que non forcée : a ARPES "
                    f"({a_ui:.3f}) ≠ a MP ({lat.a:.3f}), écart {100*rel:.1f}%."
                )
            else:
                self._status(
                    f"✓ Symétrie MP récupérée : {lat.bravais}, "
                    f"a={lat.a:.3f} Å, c={lat.c:.3f} Å ({lat.space_group or 'sg ?'})"
                )
        except Exception:
            self._status(f"✓ Symétrie MP récupérée pour {mp_id}.")
        self._draw_fs_tab()

    def _save_current_bz_crystal_settings(self):
        if not hasattr(self, "_fs_controls"):
            return
        entry = self._current_entry()
        if entry is None:
            return
        try:
            p = self._fs_controls.params()
            entry.fs_v0 = float(p.v0_eV)
            entry.fs_kz_plane = str(p.kz_plane)
            entry.fs_phi_c_deg = float(p.phi_c_deg)
            entry.fs_bz_crystal_visible = bool(p.overlay_bz_crystal)
            entry.fs_hs_crystal_visible = bool(p.overlay_hs_crystal)
            self._session.save()
        except Exception:
            pass

    def _check_bz_crystal_consistency(self) -> bool:
        """Refuse MP BZ overlay when ARPES and MP lattice constants diverge."""
        if not hasattr(self, "_fs_controls"):
            return True
        entry = self._current_entry()
        if entry is None:
            return True
        p = self._fs_controls.params()
        if not (p.overlay_bz_crystal or p.overlay_hs_crystal):
            return True
        lat = getattr(entry, "fs_lattice", None) or {}
        if not lat:
            return True
        try:
            a_ui = float(p.a_lattice)
            a_mp = float(lat.get("a", 0.0) or 0.0)
        except Exception:
            return True
        if a_mp <= 0:
            return True
        rel = abs(a_ui - a_mp) / max(a_mp, 1e-12)
        if rel <= 0.02 or bool(getattr(entry, "fs_bz_crystal_force_override", False)):
            self._warn_crystal_symmetry_mismatch(p, lat)
            return True

        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox(self._parent)
        box.setWindowTitle("Cohérence BZ MP")
        box.setText(
            f"Refuser overlay BZ MP : a ARPES = {a_ui:.3f} Å, "
            f"a MP = {a_mp:.3f} Å, écart = {100*rel:.1f}% (> 2%).\n\n"
            "Forcer seulement si les unités k sont volontairement calibrées "
            "avec un autre paramètre de maille."
        )
        force_btn = box.addButton("Forcer override", QMessageBox.ButtonRole.AcceptRole)
        disable_btn = box.addButton("Désactiver", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(disable_btn)
        box.exec()
        if box.clickedButton() == force_btn:
            entry.fs_bz_crystal_force_override = True
            self._session.save()
            self._status(f"⚠ BZ MP forcée malgré écart a ARPES/MP {100*rel:.1f}%.")
            self._warn_crystal_symmetry_mismatch(p, lat)
            return True

        self._set_bz_crystal_checks(False)
        entry.fs_bz_crystal_visible = False
        entry.fs_hs_crystal_visible = False
        entry.fs_bz_crystal_force_override = False
        self._session.save()
        self._status(f"BZ MP désactivée : écart a ARPES/MP {100*rel:.1f}% (> 2%).")
        return False

    def _set_bz_crystal_checks(self, checked: bool) -> None:
        c = getattr(self, "_fs_controls", None)
        if c is None:
            return
        for name in ("chk_bz_xtal", "chk_hs_xtal"):
            widget = getattr(c, name, None)
            if widget is None:
                continue
            try:
                old = widget.blockSignals(True)
                widget.setChecked(bool(checked))
                widget.blockSignals(old)
            except Exception:
                pass

    def _warn_crystal_symmetry_mismatch(self, params, lat: dict) -> None:
        bravais = str(lat.get("bravais", "") or "").lower()
        shape = str(getattr(params, "bz_shape", "") or "").lower()
        if "hex" in bravais and shape in {"square", "rectangle", "centered_rect"}:
            self._status(
                "⚠ BZ MP : bravais hexagonal, mais preset FS C4/rectangulaire. "
                "Vérifie le mp_id ou la symétrie visible."
            )
        elif any(k in bravais for k in ("tetra", "ortho", "cubic")) and shape == "hexagon":
            self._status(
                "⚠ BZ MP : bravais C4/orthogonal, mais preset FS hexagonal. "
                "Vérifie le mp_id ou la symétrie visible."
            )

    def _inject_fs_lattice_into_raw(self):
        """Pousse fs_lattice (depuis FileEntry) dans raw_data.metadata."""
        if self._raw_data is None:
            return
        entry = self._current_entry()
        if entry is None:
            return
        lat = getattr(entry, "fs_lattice", None) or {}
        if not lat:
            return
        meta = self._raw_data.setdefault("metadata", {}) or {}
        meta["fs_lattice"] = dict(lat)
        self._raw_data["metadata"] = meta

    def _update_fs_kz_label(self):
        """Met à jour label kz dans le panel FS (GF5 redteam : repliement 1BZ)."""
        if not hasattr(self, "_fs_controls"):
            return
        lbl = getattr(self._fs_controls, "lbl_kz", None)
        if lbl is None:
            return
        try:
            hv = self._raw_data and float((self._raw_data.get("metadata", {}) or {}).get("hv", 0.0))
        except Exception:
            hv = None
        if not hv or hv <= 0:
            lbl.setText("kz : —  |  hν inconnu  |  cristal : "
                        + self._lattice_summary())
            return
        try:
            p = self._fs_controls.params()
            wf = float(getattr(self._session, "work_func", 4.031))
            kz_arr = kz_from_hv_kpar(
                hv, np.array([0.0]),
                work_func=wf, inner_potential=float(p.v0_eV),
                a_lattice=float(p.a_lattice or 3.96),
                energy=0.0,
            )
            kz_val = float(kz_arr[0])
            entry = self._current_entry()
            lat_dict = (entry.fs_lattice if entry else {}) or {}
            c = float(lat_dict.get("c", 0.0) or 0.0)
            if c > 0:
                folded = fold_kz_to_1bz(kz_val, c)
                bord = " ⚠ proche bord" if folded["near_boundary"] else ""
                lbl.setText(
                    f"kz = {folded['kz_reduced_pi_over_c']:.3f} π/c  "
                    f"| plan : {folded['plane']}  (zone n={folded['n_zone']}){bord}  "
                    f"| cristal : {self._lattice_summary()}"
                )
            else:
                lbl.setText(
                    f"kz = {kz_val:.3f} Å⁻¹  |  c inconnu (récup MP)  "
                    f"|  cristal : {self._lattice_summary()}"
                )
        except Exception as exc:
            lbl.setText(f"kz : erreur ({exc})")

    # ------------------------------------------------------------------
    #  Propagation distortion BM → volume FS (opt-in)
    # ------------------------------------------------------------------

    def _on_propagate_distortion_fs_toggled(self):
        """Sync UI → entry + redraw FS."""
        enabled = False
        sender = None
        try:
            sender = self._parent.sender()
        except Exception:
            sender = None
        if sender is not None and hasattr(sender, "isChecked"):
            enabled = bool(sender.isChecked())
        elif hasattr(self, "_fs_controls") and hasattr(self._fs_controls, "chk_distortion_fs"):
            enabled = bool(self._fs_controls.chk_distortion_fs.isChecked())
        elif hasattr(self, "_params") and hasattr(self._params, "chk_distortion_fs_propagate"):
            enabled = bool(self._params.chk_distortion_fs_propagate.isChecked())
        entry = self._current_entry()
        if entry is not None:
            try:
                entry.propagate_distortion_to_fs = enabled
                self._session.save()
            except Exception:
                pass
        self._sync_distortion_fs_toggles(enabled)
        # Invalide cache local pour forcer recalcul
        self._fs_distortion_cache_invalidate()
        self._draw_fs_tab()

    def _fs_distortion_cache_invalidate(self):
        if hasattr(self, "_fs_distortion_cache"):
            self._fs_distortion_cache.clear()

    def _apply_distortion_to_fs_volume_if_enabled(self) -> bool:
        """Si opt-in actif + cfg distortion valide : swap fs_data → corrigé.

        Retourne ``True`` si propagation effective (pour badge).

        Garde-fous :
        - Refus silencieux si pas FS (fs_data absent).
        - Refus + warning si calib_key BM ≠ FS (calib_key_for_meta).
        - Refus + warning si drift_ratio > 15% (calculé dans
          apply_distortion_to_fs_volume).
        """
        from arpes.physics.distortion import (
            apply_distortion_to_fs_volume,
            fs_domain_checksum,
            is_distortion_active,
        )

        if self._raw_data is None:
            return False
        entry = self._current_entry()
        if entry is None:
            return False
        if not bool(getattr(entry, "propagate_distortion_to_fs", False)):
            self._restore_fs_data_original()
            return False
        meta = self._raw_data.get("metadata", {}) or {}
        if meta.get("fs_data") is None:
            return False
        cfg = self._distortion_cfg_for_current_fs(entry, meta)
        if not is_distortion_active(cfg):
            self._restore_fs_data_original()
            self._status(
                "⚠ Distortion FS : aucune calibration BM active pour cette FS "
                "(calibrer une BM de même géométrie puis activer ici)."
            )
            self._disable_fs_distortion_after_failure(entry)
            return False

        # Sauve original si pas déjà fait (clé séparée pour restore propre).
        if "fs_data_orig" not in meta:
            meta["fs_data_orig"] = meta["fs_data"]

        kx = np.asarray(meta.get("fs_kx"), dtype=float)
        ky = np.asarray(meta.get("fs_ky"), dtype=float)
        ev = np.asarray(meta.get("fs_energy"), dtype=float)
        fs_orig = np.asarray(meta["fs_data_orig"])

        # Cache LRU 1 entrée par signature.
        if not hasattr(self, "_fs_distortion_cache"):
            self._fs_distortion_cache = {}
        from arpes.physics.distortion import cache_signature
        sig = (id(fs_orig), cache_signature(cfg))
        cached = self._fs_distortion_cache.get(sig)
        if cached is not None:
            meta["fs_data"] = cached
            self._raw_data["metadata"] = meta
            return True

        try:
            bm_chk = fs_domain_checksum(kx, ev)
            corrected, info = apply_distortion_to_fs_volume(
                fs_orig, kx, ky, ev, cfg, bm_checksum=bm_chk,
            )
        except ValueError as exc:
            self._status(f"⚠ Distortion FS : {exc}")
            self._restore_fs_data_original()
            self._disable_fs_distortion_after_failure(entry)
            return False
        except Exception as exc:
            self._status(f"✗ Distortion FS : échec {exc}")
            self._restore_fs_data_original()
            self._disable_fs_distortion_after_failure(entry)
            return False

        if not info.get("applied"):
            self._restore_fs_data_original()
            self._disable_fs_distortion_after_failure(entry)
            return False

        # Cache LRU 1 entrée (clear avant insert)
        self._fs_distortion_cache.clear()
        self._fs_distortion_cache[sig] = corrected
        meta["fs_data"] = corrected
        self._raw_data["metadata"] = meta
        self._status(
            f"✓ Distortion BM propagée au volume FS "
            f"(n_slices={info.get('n_slices', 0)}, "
            f"drift={info.get('drift_ratio', 0.0):.3f})"
        )
        return True

    def _disable_fs_distortion_after_failure(self, entry) -> None:
        try:
            entry.propagate_distortion_to_fs = False
            self._session.save()
        except Exception:
            pass
        self._sync_distortion_fs_toggles(False)

    def _distortion_cfg_for_current_fs(self, entry, meta: dict) -> dict:
        """Retourne la config distortion à utiliser pour FS.

        Priorité :
        1. config stockée directement sur l'entrée FS ;
        2. calibration partagée `(lens_mode, pass_energy, hv)` créée depuis une BM.
        """
        from arpes.physics.distortion import calib_key_for_meta, is_distortion_active
        from arpes.ui.controllers.distortion_controller import _load_calib_store

        cfg = getattr(entry, "bm_distortion", {}) or {}
        if is_distortion_active(cfg):
            return cfg
        try:
            key = "|".join(calib_key_for_meta(meta))
            calib = (_load_calib_store() or {}).get(key)
        except Exception:
            calib = None
        if not calib or not is_distortion_active(calib):
            return {}
        cfg = {
            "enabled": True,
            "trapezoid": dict(calib.get("trapezoid") or {}),
            "parabola": dict(calib.get("parabola") or {}),
            "calib_key": list(calib_key_for_meta(meta)),
            "source": "shared_calib_fs",
        }
        entry.bm_distortion = cfg
        try:
            self._session.save()
        except Exception:
            pass
        if hasattr(self, "_params") and hasattr(self._params, "set_bm_distortion_state"):
            try:
                self._params.set_bm_distortion_state(cfg)
            except Exception:
                pass
        return cfg

    def _restore_fs_data_original(self):
        if self._raw_data is None:
            return
        meta = self._raw_data.get("metadata", {}) or {}
        orig = meta.pop("fs_data_orig", None)
        if orig is not None:
            meta["fs_data"] = orig
            self._raw_data["metadata"] = meta

    def _draw_fs_distortion_badge(self, active: bool):
        """Badge orange coin haut-droit canvas FS quand propagation active."""
        if not hasattr(self, "_fs_canvas"):
            return
        if not active:
            return
        try:
            txt = self._fs_canvas.ax.text(
                0.98, 0.98, "FS distordue (BM-calib)",
                transform=self._fs_canvas.ax.transAxes,
                ha="right", va="top",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#FF8800", alpha=0.85,
                          edgecolor="black", boxstyle="round,pad=0.3"),
                zorder=10,
            )
            self._fs_canvas._overlay_artists.append(txt)
            self._fs_canvas.canvas.draw_idle()
        except Exception:
            pass

    def _restore_fs_crystal_settings_from_entry(self, entry):
        """Sync widgets BZ-crystal depuis FileEntry (au chargement de fichier)."""
        if not hasattr(self, "_fs_controls") or entry is None:
            return
        c = self._fs_controls
        widgets = [c.sp_v0, c.cmb_kz_plane, c.sp_phi_c,
                   c.chk_bz_xtal, c.chk_hs_xtal, c.ed_mp_id]
        for w in widgets:
            w.blockSignals(True)
        try:
            c.sp_v0.setValue(float(getattr(entry, "fs_v0", 12.0) or 12.0))
            plane = str(getattr(entry, "fs_kz_plane", "Auto") or "Auto")
            idx = c.cmb_kz_plane.findText(plane)
            if idx >= 0:
                c.cmb_kz_plane.setCurrentIndex(idx)
            c.sp_phi_c.setValue(float(getattr(entry, "fs_phi_c_deg", 0.0) or 0.0))
            c.chk_bz_xtal.setChecked(bool(getattr(entry, "fs_bz_crystal_visible", False)))
            c.chk_hs_xtal.setChecked(bool(getattr(entry, "fs_hs_crystal_visible", False)))
            lat = getattr(entry, "fs_lattice", None) or {}
            c.ed_mp_id.setText(str(lat.get("mp_id", "") or ""))
        finally:
            for w in widgets:
                w.blockSignals(False)
        # Sync chk propagation distortion (panel distortion vit dans _params).
        if hasattr(self, "_params") and hasattr(self._params, "chk_distortion_fs_propagate"):
            self._sync_distortion_fs_toggles(
                bool(getattr(entry, "propagate_distortion_to_fs", False))
            )
        # Invalide cache distortion FS (changement de fichier)
        self._fs_distortion_cache_invalidate()

    def _sync_distortion_fs_toggles(self, enabled: bool) -> None:
        for obj in (
            getattr(getattr(self, "_params", None), "chk_distortion_fs_propagate", None),
            getattr(getattr(self, "_fs_controls", None), "chk_distortion_fs", None),
        ):
            if obj is None:
                continue
            obj.blockSignals(True)
            try:
                obj.setChecked(bool(enabled))
            finally:
                obj.blockSignals(False)

    def _lattice_summary(self) -> str:
        entry = self._current_entry()
        if entry is None:
            return "non chargé"
        lat = getattr(entry, "fs_lattice", None) or {}
        if not lat:
            return "non chargé"
        return (f"{lat.get('bravais', '?')} "
                f"a={float(lat.get('a', 0)):.2f} Å "
                f"c={float(lat.get('c', 0)):.2f} Å")
