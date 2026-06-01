"""Gamma-related UI controller for ArpesExplorer."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from arpes.ui.widgets.fs_panel import FermiSurfaceCanvas, FSControlPanel
from arpes.physics.gamma import (
    apply_bm_gamma_axis_shift as _gamma_apply_bm_axis_shift,
    build_gamma_reference as _gamma_build_reference,
    gamma_reference_to_bm_center as _gamma_ref_to_bm_center,
    k_to_angle_offset_deg as _gamma_k_to_angle_offset_deg,
    angle_offsets_from_k_center as _gamma_angle_offsets_from_k_center,
    project_gamma_by_azi as _gamma_project_by_azi,
    stored_gamma_reference as _gamma_stored_reference,
)
from arpes.physics.gamma_resolver import ResolvedGamma, resolve as _gamma_resolve


_GAMMA_META_KEYS = (
    "bm_gamma_axis_centered", "bm_gamma_axis_shift", "bm_gamma_axis_note",
    "fs_gamma_axis_centered", "fs_gamma_axis_shift_kx", "fs_gamma_axis_shift_ky",
    "bm_gamma_reference_source", "bm_gamma_reference_path", "bm_gamma_reference_azi",
)


def _snapshot_meta_gamma(meta: dict | None) -> dict:
    if not meta:
        return {}
    return {k: meta[k] for k in _GAMMA_META_KEYS if k in meta}


def _is_axis_locked(meta: dict | None) -> tuple[bool, str]:
    """Renvoie (locked, raison_user) si l'axe Γ est déjà déterminé.

    Une fois locked, les détecteurs Γ (auto BM / auto FS / pick manuel) doivent
    refuser de re-écrire la référence pour éviter drift cumulé / double-shift.
    """
    if not meta:
        return False, ""
    if meta.get("angle_offsets_applied"):
        return True, "Γ déjà appliqué par offset angulaire loader. Utilise « Oublier Γ » (menu Γ ou _forget_gamma) pour réinitialiser."
    if meta.get("bm_gamma_axis_centered") or meta.get("fs_gamma_axis_centered"):
        return True, "Γ déjà appliqué (axe recentré). Utilise « Oublier Γ » (menu Γ ou _forget_gamma) pour réinitialiser."
    return False, ""


def _shift_fit_result_in_place(fr: dict | None, delta: float) -> None:
    """Apply -delta to all kF / Γ entries of a fit_result dict (mutates in place).

    Why: axis recenter (kpar -= delta) leaves stale kF in fit_result, overlay drifts.
    """
    if not fr or abs(delta) < 1e-12:
        return
    for key in ("kF_minus", "kF_plus", "gamma_corrige"):
        pairs = fr.get(key)
        if not pairs:
            continue
        new_pairs = []
        for series in pairs:
            new_pairs.append([
                (float(v) - delta) if v is not None and np.isfinite(v) else v
                for v in series
            ])
        fr[key] = new_pairs


class GammaController:
    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name == "_parent":
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)

    def _store_fs_center_reference(self, kx: float, ky: float, *, source: str):
        if self._raw_data is None:
            return
        meta = (self._raw_data or {}).get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        self._session.gamma_reference = _gamma_build_reference(
            kx=kx, ky=ky,
            metadata=meta,
            hv=self._raw_data.get("hv"),
            path=self._raw_data.get("path"),
            azi=azi_ref,
            source=source,
            direction=(entry_now.meta.direction if entry_now else None),
        )
        offsets = self._angle_offsets_from_k_center(
            float(kx), float(ky),
            hv=self._raw_data.get("hv"),
            source=source,
            ref_path=self._raw_data.get("path"),
            azi=azi_ref,
        )
        if offsets:
            self._session.angle_offsets = offsets
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fs_center_kx = float(kx)
            entry.fs_center_ky = float(ky)
        self._session.save()

    def _k_to_angle_offset_deg(self, k_pi_a: float, *, hv: float | None = None) -> float | None:
        """Convertit un decalage k (pi/a) en offset angulaire CLS (wrapper UI)."""
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return None
        return _gamma_k_to_angle_offset_deg(k_pi_a, hv=hv_val, work_func=work_func)

    def _angle_offsets_from_k_center(
        self,
        kx: float,
        ky: float = 0.0,
        *,
        hv: float | None = None,
        source: str = "",
        ref_path: str | None = None,
        azi: float | None = None,
    ) -> dict:
        try:
            hv_val = float(hv if hv is not None else self._params.sp_hv.value())
            work_func = float(self._params.sp_phi.value())
        except Exception:
            return {}
        return _gamma_angle_offsets_from_k_center(
            kx, ky,
            hv=hv_val, work_func=work_func,
            source=source, ref_path=ref_path, azi=azi,
        )

    def _project_gamma_by_azi(
        self,
        ref: dict,
        azi_target: float | None,
        *,
        warn_label: str = "Γ",
    ) -> tuple[float, float]:
        """Projette le Γ de référence dans le repère du fichier courant (wrapper UI)."""
        return _gamma_project_by_azi(
            ref, azi_target,
            on_warn=self._status,
            warn_label=warn_label,
        )

    def _set_fs_center_pick_mode(self, active: bool):
        active = bool(active)
        self._fs_pick_center_active = active
        if hasattr(self, "_fs_controls") and hasattr(self._fs_controls, "set_manual_center_active"):
            self._fs_controls.set_manual_center_active(active)
        if hasattr(self, "_fs_canvas") and hasattr(self._fs_canvas, "canvas"):
            if active:
                self._fs_canvas.canvas.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._fs_canvas.canvas.unsetCursor()
        if active:
            if self._tabs.currentIndex() != 3:
                self._tabs.setCurrentIndex(3)
            self._status("Centrage manuel Γ : clique sur le centre à viser dans la carte FS.")

    def _sync_current_fs_gamma_to_bm_center(self, kx: float) -> None:
        """Keep BM fit center aligned when Γ is picked/detected on current FS."""
        if self._raw_data is None:
            return
        meta = self._raw_data.get("metadata", {}) or {}
        if meta.get("fs_data") is None:
            return
        centered_axis = bool(
            meta.get("angle_offsets_applied")
            or meta.get("bm_gamma_axis_centered")
            or meta.get("fs_gamma_axis_centered")
        )
        center = 0.0 if centered_axis else float(kx)
        sp_cx = getattr(self._params, "sp_cx", None)
        if sp_cx is not None and hasattr(sp_cx, "setValue"):
            sp_cx.setValue(center)
        if self._current_path:
            entry = self._session.get_or_create(self._session.key_for_path(self._current_path))
            entry.fit_params.center_init = center
            self._session.save()

    def _on_fs_map_click(self, event):
        if not getattr(self, "_fs_pick_center_active", False):
            return
        if self._raw_data is None or not self._current_is_fs():
            self._set_fs_center_pick_mode(False)
            QMessageBox.warning(self._parent, "Centrage manuel Γ", "Charge d'abord une FS.")
            return
        if FSControlPanel is None or not hasattr(self, "_fs_controls"):
            return
        if not hasattr(self, "_fs_canvas") or event.inaxes is not getattr(self._fs_canvas, "ax", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        locked, reason = _is_axis_locked(self._raw_data.get("metadata", {}))
        if locked:
            self._set_fs_center_pick_mode(False)
            self._status(reason)
            return
        params = self._fs_controls.params()
        kx = float(params.kx_center + event.xdata)
        ky = float(params.ky_center + event.ydata)
        self._fs_controls.set_center(kx, ky)
        self._store_fs_center_reference(kx, ky, source="fs_manual")
        # P2.bis : single-setter — recompose la décision à partir de la nouvelle
        # ref et applique (shift axe + sp_cx + center_init + save) en une fois.
        self.apply_resolved_gamma(self._resolve_gamma_for_current(), save_entry=True)
        self._set_fs_center_pick_mode(False)
        self._draw_fs_tab()
        msg = f"Gamma FS manuel : kx={kx:+.4f}, ky={ky:+.4f} π/a"
        self._status(msg)
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"Gamma FS manuel mémorisé ({kx:+.4f}, {ky:+.4f})")
        try:
            self._fs_controls.lbl_info.setText(msg)
        except Exception:
            pass

    def _detect_fs_gamma(self):
        if FermiSurfaceCanvas is None or FSControlPanel is None:
            return
        if self._raw_data is not None:
            locked, reason = _is_axis_locked(self._raw_data.get("metadata", {}))
            if locked:
                self._status(reason)
                return
        try:
            params = self._fs_controls.params()
            res = self._fs_canvas.detect_gamma(self._raw_data, params)
            self._fs_controls.set_center(res["kx"], res["ky"])
            self._store_fs_center_reference(res["kx"], res["ky"], source="fs_auto")
            # P2.bis : single-setter (cf _on_fs_map_click).
            self.apply_resolved_gamma(self._resolve_gamma_for_current(), save_entry=True)
            self._draw_fs_tab()
            score = res.get("symmetry_score")
            score_txt = ""
            try:
                score_f = float(score)
                score_txt = f", corr={score_f:.2f}" if np.isfinite(score_f) else ""
            except (TypeError, ValueError):
                pass
            quality = res.get("quality", "?")
            delta = res.get("gamma_delta_kx")
            delta_txt = ""
            try:
                delta_f = float(delta)
                delta_txt = f", Δkx={delta_f:+.3f}" if np.isfinite(delta_f) else ""
            except (TypeError, ValueError):
                pass
            msg = (f"Gamma FS détecté : kx={res['kx']:+.4f}, ky={res['ky']:+.4f} π/a "
                   f"| {len(res.get('gamma_kx_list', []))} coupes kx, "
                   f"{len(res.get('gamma_ky_list', []))} coupes ky"
                   f" | qualité={quality}{score_txt}{delta_txt}")
            self._status(msg)
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"Gamma FS détecté ({res['kx']:+.4f}, {res['ky']:+.4f})")
            try:
                self._fs_controls.lbl_info.setText(msg)
            except Exception:
                pass
        except Exception as exc:
            QMessageBox.warning(self._parent, "Détection Gamma", str(exc))

    def _stored_gamma_reference(self) -> dict:
        return _gamma_stored_reference(self._session.gamma_reference)

    def _gamma_reference_to_bm_center(self, ref: dict) -> tuple[float, float]:
        """Wrapper UI : délègue à `arpes_gamma.gamma_reference_to_bm_center`."""
        if self._raw_data is None:
            return np.nan, 0.0
        meta = self._raw_data.get("metadata", {}) or {}
        entry_now = self._current_entry()
        azi_bm = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
        return _gamma_ref_to_bm_center(
            ref,
            bm_metadata=meta,
            bm_hv=self._raw_data.get("hv"),
            work_func=float(self._params.sp_phi.value()),
            bm_azi=azi_bm,
            on_warn=self._status,
        )

    def _center_current_bm_axis_on_gamma(self, gamma_bm: float, ref: dict | None = None) -> bool:
        """Center the current BM display axis from an explicit Γ action.

        Si raw est une FS, propage `allow_fs=True` + gamma_ky pour garder le
        marker FS aligné (bug P1.3). Snapshot ensuite l'état meta vers
        `entry.meta_gamma_state` pour survivre save/load (P1.1).
        """
        if self._raw_data is None:
            return False
        meta_before = self._raw_data.get("metadata", {}) or {}
        try:
            previous_shift = float(meta_before.get("bm_gamma_axis_shift", 0.0) or 0.0)
        except (TypeError, ValueError):
            previous_shift = 0.0
        is_fs = meta_before.get("fs_data") is not None
        gamma_ky = 0.0
        if is_fs and ref is not None:
            try:
                gamma_ky = float(ref.get("ky", 0.0) or 0.0)
            except (TypeError, ValueError):
                gamma_ky = 0.0
        applied = _gamma_apply_bm_axis_shift(
            self._raw_data, gamma_bm, ref=ref,
            allow_fs=is_fs, gamma_ky=gamma_ky,
        )
        if not applied:
            return False
        meta_after = self._raw_data.get("metadata", {}) or {}
        try:
            new_shift = float(meta_after.get("bm_gamma_axis_shift", previous_shift) or previous_shift)
        except (TypeError, ValueError):
            new_shift = previous_shift
        delta = new_shift - previous_shift
        if hasattr(self, "_sel_k"):
            self._sel_k = float(self._sel_k - delta)
        if abs(delta) > 1e-12:
            self._remap_fit_results_by_delta(delta)
        self._snapshot_gamma_meta_to_entry()
        return True

    def _snapshot_gamma_meta_to_entry(self) -> None:
        """P1.1 — persiste les flags meta gamma sur l'entry pour survivre reload."""
        entry = self._current_entry()
        if entry is None or self._raw_data is None:
            return
        meta = self._raw_data.get("metadata", {}) or {}
        entry.meta_gamma_state = _snapshot_meta_gamma(meta)

    def _restore_gamma_meta_from_entry(self) -> None:
        """P1.1 — restaure les flags meta gamma depuis entry vers raw_data.

        Appelé par load_controller juste avant `_apply_stored_gamma_to_current_file`
        pour éviter le drift au reload (le code applicateur voit `previous_shift`
        cohérent et n'applique aucun delta supplémentaire).
        """
        entry = self._current_entry()
        if entry is None or self._raw_data is None:
            return
        state = getattr(entry, "meta_gamma_state", None) or {}
        if not state:
            return
        meta = self._raw_data.setdefault("metadata", {})
        for k, v in state.items():
            meta.setdefault(k, v)

    def _remap_fit_results_by_delta(self, delta: float) -> None:
        """Shift kF/Γ entries in fit_result (+ all fit_zones) by -delta after axis recenter.

        P1.6 — snapshot avant mutation, restore si save() échoue, pour éviter
        divergence mémoire/disque.
        """
        entry = self._current_entry()
        if entry is None:
            return
        import copy
        backup_fit = copy.deepcopy(getattr(entry, "fit_result", None))
        backup_zones = copy.deepcopy(getattr(entry, "fit_zones", None) or [])
        _shift_fit_result_in_place(getattr(entry, "fit_result", None), delta)
        for z in getattr(entry, "fit_zones", None) or []:
            _shift_fit_result_in_place(z.get("fit_result"), delta)
        try:
            self._session.save()
        except Exception as exc:
            entry.fit_result = backup_fit
            entry.fit_zones = backup_zones
            self._status(f"Attention: sauvegarde après remap Γ échouée : {exc} — état restauré")

    # ---------------------------------------------------------------
    # P2 — chemin resolver/single-setter
    # ---------------------------------------------------------------
    def _resolve_gamma_for_current(self) -> ResolvedGamma:
        """Pure : appelle le resolver Γ avec les attributs courants."""
        entry = self._current_entry()
        azi = entry.meta.azi if (entry and entry.meta.azi is not None) else None
        work_func = (
            float(self._params.sp_phi.value())
            if hasattr(self._params, "sp_phi") else 4.031
        )
        ref = self._stored_gamma_reference()
        return _gamma_resolve(
            self._raw_data, ref,
            work_func=work_func,
            bm_hv=self._raw_data.get("hv") if self._raw_data else None,
            entry_azi=azi,
        )

    def apply_resolved_gamma(self, resolved: ResolvedGamma, *, save_entry: bool = False) -> None:
        """Single-setter Γ : seul point qui mute `sp_cx`, `entry.fit_params.center_init`,
        l'axe k de raw_data, le marker FS, `_sel_k`, et déclenche le remap fit_result.

        Tous les handlers Γ (auto BM, auto FS, click manuel, auto-apply load, FS→BM)
        doivent passer par ici. C'est l'analogue de `set_fit_result` pour Γ.
        """
        if resolved.mode == "none":
            return
        ref = self._stored_gamma_reference()
        entry = self._current_entry()

        # 1. Push display center (sp_cx) avec blockSignals pour éviter cascade
        sp_cx = getattr(self._params, "sp_cx", None)
        if sp_cx is not None and hasattr(sp_cx, "setValue"):
            had_block = False
            if hasattr(sp_cx, "blockSignals"):
                try:
                    had_block = sp_cx.blockSignals(True)
                except Exception:
                    had_block = False
            try:
                sp_cx.setValue(float(resolved.display_center))
            finally:
                if hasattr(sp_cx, "blockSignals"):
                    try:
                        sp_cx.blockSignals(had_block)
                    except Exception:
                        pass

        # 2. Push fit_center_init dans l'entry
        if entry is not None:
            entry.fit_params.center_init = float(resolved.fit_center_init)

        # 3. Marker FS (si applicable)
        if (
            resolved.is_fs
            and FSControlPanel is not None
            and hasattr(self, "_fs_controls")
            and np.isfinite(resolved.fs_marker_kx)
            and np.isfinite(resolved.fs_marker_ky)
        ):
            self._fs_controls.set_center(float(resolved.fs_marker_kx),
                                         float(resolved.fs_marker_ky))
            if entry is not None:
                entry.fs_center_kx = float(resolved.fs_marker_kx)
                entry.fs_center_ky = float(resolved.fs_marker_ky)

        # 4. Shift axe k si delta non nul (réutilise le chemin existant pour
        #    bénéficier du remap fit_result + snapshot meta + sel_k update).
        if resolved.mode == "axis_shifted" and abs(resolved.axis_shift_delta) > 1e-12:
            self._center_current_bm_axis_on_gamma(
                float(resolved.axis_shift_target),
                ref=({**ref, "ky": float(resolved.fs_marker_ky)}
                     if ref and resolved.is_fs and np.isfinite(resolved.fs_marker_ky)
                     else ref),
            )

        # 5. Persistance
        if save_entry:
            try:
                self._session.save()
            except Exception as exc:
                self._status(f"Attention: sauvegarde Γ échouée : {exc}")

        # 6. Feedback
        if resolved.reason:
            self._status(resolved.reason)

    def _forget_gamma(self) -> None:
        """P2.bis — réinitialise tout l'état Γ (session + entry + raw_data).

        Porte de sortie aux gardes `_is_axis_locked` : permet de repartir
        d'un axe brut et re-détecter Γ. Inverse le shift d'axe en utilisant
        la valeur courante de `bm_gamma_axis_shift`, remap `fit_result` en
        sens inverse, puis efface tous les flags Γ + références session.
        """
        meta = self._raw_data.get("metadata", {}) if self._raw_data else {}
        try:
            previous_shift = float(meta.get("bm_gamma_axis_shift", 0.0) or 0.0)
        except (TypeError, ValueError):
            previous_shift = 0.0
        was_centered = bool(
            meta.get("bm_gamma_axis_centered") or meta.get("fs_gamma_axis_centered")
        )

        # 1. Inverse shift kpar / fs_kx / fs_ky si axe centré
        if self._raw_data is not None and was_centered and abs(previous_shift) > 1e-12:
            kpar = np.asarray(self._raw_data.get("kpar"), dtype=float)
            if kpar.size:
                self._raw_data["kpar"] = kpar + previous_shift
            if meta.get("fs_data") is not None:
                fs_kx = meta.get("fs_kx")
                if fs_kx is not None:
                    meta["fs_kx"] = np.asarray(fs_kx, dtype=float) + previous_shift
                try:
                    previous_ky = float(meta.get("fs_gamma_axis_shift_ky", 0.0) or 0.0)
                except (TypeError, ValueError):
                    previous_ky = 0.0
                if abs(previous_ky) > 1e-12:
                    fs_ky = meta.get("fs_ky")
                    if fs_ky is not None:
                        meta["fs_ky"] = np.asarray(fs_ky, dtype=float) + previous_ky
            self._remap_fit_results_by_delta(-previous_shift)
            if hasattr(self, "_sel_k"):
                self._sel_k = float(self._sel_k + previous_shift)

        # 2. Clear tous les flags meta Γ
        if self._raw_data is not None:
            for k in _GAMMA_META_KEYS:
                meta.pop(k, None)

        # 3. Clear session-level state
        self._session.gamma_reference = {}
        self._session.angle_offsets = {}

        # 4. Clear entry-level state (current file)
        entry = self._current_entry()
        if entry is not None:
            entry.meta_gamma_state = {}
            entry.fs_center_kx = None
            entry.fs_center_ky = None
            entry.fit_params.center_init = 0.0

        # 5. Reset sp_cx
        sp_cx = getattr(self._params, "sp_cx", None)
        if sp_cx is not None and hasattr(sp_cx, "setValue"):
            had_block = sp_cx.blockSignals(True) if hasattr(sp_cx, "blockSignals") else False
            try:
                sp_cx.setValue(0.0)
            finally:
                if hasattr(sp_cx, "blockSignals"):
                    try:
                        sp_cx.blockSignals(had_block)
                    except Exception:
                        pass

        # 6. Reset FS marker
        if FSControlPanel is not None and hasattr(self, "_fs_controls"):
            try:
                self._fs_controls.set_center(0.0, 0.0)
            except Exception:
                pass

        try:
            self._session.save()
        except Exception as exc:
            self._status(f"Attention: sauvegarde après reset Γ échouée : {exc}")

        self._status("Γ réinitialisé : références session, axes et flags effacés.")
        try:
            self._draw_current_view()
        except Exception:
            pass

    def _apply_stored_gamma_to_current_file(self, *, save_entry: bool = False):
        """Auto-apply Γ stocké sur le fichier courant (chemin load / switch).

        P2 : délègue intégralement au resolver + single-setter.
        """
        if self._raw_data is None:
            return
        resolved = self._resolve_gamma_for_current()
        self.apply_resolved_gamma(resolved, save_entry=save_entry)

    def _estimate_gamma_bm(self):
        if self._raw_data is not None:
            locked, reason = _is_axis_locked(self._raw_data.get("metadata", {}))
            if locked:
                self._status(reason)
                return
        try:
            import arpes.ui.widgets.plots as ap
        except Exception:
            self._status("Attention: arpes_plots non chargé")
            return
        data, kpar, ev = self._get_work_data()
        if data is None:
            return
        try:
            res = ap.estimate_gamma_bm_mdc(
                data, kpar, ev,
                ev_range=(self._params.sp_evs.value(), self._params.sp_eve.value()),
                k_range=(self._params.sp_kmin.value(), self._params.sp_kmax.value()),
                center_guess=self._params.sp_cx.value(),
                center_window=max(self._params.sp_xg.value() * 2.0, 0.25),
                smooth_sigma=self._params.sp_sfd.value(),
                verbose=False,
            )
            gamma = float(res["gamma"])
            if not np.isfinite(gamma):
                QMessageBox.warning(
                    self, "Auto Γ BM",
                    "Impossible d'estimer Γ : pas assez de paires MDC valides. "
                    "Ajuste la plage d'énergie, k_min/k_max ou centre_init."
                )
                return
            entry_now = self._current_entry()
            azi_ref = entry_now.meta.azi if (entry_now and entry_now.meta.azi is not None) else None
            meta_now = self._raw_data.get("metadata", {}) or {}
            self._session.gamma_reference = _gamma_build_reference(
                kx=gamma, ky=0.0,
                metadata=meta_now,
                hv=self._raw_data.get("hv"),
                path=self._raw_data.get("path"),
                azi=azi_ref,
                source="bm",
                direction=(entry_now.meta.direction if entry_now else None),
            )
            if not meta_now.get("angle_offsets_applied") and not meta_now.get("bm_gamma_axis_centered"):
                offsets = self._angle_offsets_from_k_center(
                    float(gamma), 0.0,
                    hv=self._raw_data.get("hv"),
                    source="bm_auto",
                    ref_path=self._raw_data.get("path"),
                    azi=azi_ref,
                )
                if offsets:
                    self._session.angle_offsets = offsets
            # P2.bis : single-setter — pose sp_cx + center_init + shift axe
            # + remap fit_result + save en une fois via la décision resolver.
            self.apply_resolved_gamma(self._resolve_gamma_for_current(), save_entry=True)
            self._params.lbl_res.setText(
                f"Γ BM = {gamma:+.4f} π/a\n"
                f"n={res['n']}  MAD={res['mad']:.4f}"
            )
            if hasattr(self._params, "mark_action_done"):
                self._params.mark_action_done(f"Auto Γ BM appliqué ({gamma:+.4f} π/a)")
            self._draw_current_view()
            self._status(f"Γ BM estimé : {gamma:+.4f} π/a  n={res['n']}  MAD={res['mad']:.4f}")
        except Exception as exc:
            QMessageBox.warning(self._parent, "Auto Γ BM", str(exc))
            self._status(f"Attention: Auto Γ BM : {exc}")

    def _apply_gamma_reference_to_bm(self):
        """Bouton « Γ FS → BM ».

        P2.bis : délègue intégralement au resolver + single-setter. Le mode et
        le message de feedback viennent de `ResolvedGamma.mode` / `.reason`.
        """
        ref = self._stored_gamma_reference()
        if not ref:
            QMessageBox.warning(
                self._parent, "Γ FS → BM",
                "Aucun Γ de référence. Va sur l'onglet FS et clique d'abord sur Détecter Γ FS.",
            )
            return
        if self._raw_data is None:
            return
        resolved = self._resolve_gamma_for_current()
        if resolved.mode == "none":
            QMessageBox.warning(self._parent, "Γ FS → BM", "La référence Γ stockée est invalide.")
            return
        self.apply_resolved_gamma(resolved, save_entry=True)
        mode_msg = (
            "offset angulaire loader" if resolved.mode == "loader_baked"
            else ("axe k recentré" if abs(resolved.axis_shift_delta) > 1e-12
                  else "axe déjà à jour")
        )
        if hasattr(self._params, "lbl_res"):
            self._params.lbl_res.setText(
                f"Γ FS→BM appliqué\n"
                f"target={resolved.axis_shift_target:+.4f} π/a\n"
                f"{mode_msg}"
            )
        if hasattr(self._params, "mark_action_done"):
            self._params.mark_action_done(f"Γ FS appliqué à la BM ({resolved.axis_shift_target:+.4f} π/a)")
        self._update_display_data()
        self._draw_current_view()
