# Audit Updates History

## 2026-06-05 — P3 résiduel (P3.4 / P3.6 b·d / P3.7)

Env `peaks`. Suite 816 OK / 9 skip ; launch réel + garde runtime vérifiés.

- **P3.6b** — supprimé le `try/except` mort de app.py qui forçait 7 symboles
  (load_arpes*, detect_*, ARPESData, FSControlPanel…) + `ERLAB_OK` à None.
  Inutilisés par app.py ; loaders/fs_panel déjà importés (et fail-loud) par les
  controllers/builders → plus de dégradation silencieuse. Test helper mis à jour
  (patch `arpes.app.detect_format` mort → seul `browsers/files.py` patché).
- **P3.6d** — 17 instanciations de controllers extraites dans
  `ArpesExplorer._install_controllers()` (ordre documenté, AVANT QTimer.connect,
  règle #5). Les 4 imports lazy regroupés là.
- **P3.4** — `FitZone` dataclass (schéma typé) + `from_dict` (remplit défauts,
  **warn fort** sur clé inconnue) + `to_dict` + `normalize_fit_zones`, appliqué
  au load (`Session.load_from_payload`). Tue « pertes silencieuses au load » sans
  toucher les 93 sites d'accès par clé (stockage reste `dict` ; conversion
  complète des consumers = dette à terme, risque 93-sites). 4 tests.
- **P3.7** — garde de ré-entrance `_fit_busy` sur `_fit_full` + `_fit_ensemble` :
  `processEvents()` peut livrer un « Fit » re-cliqué → ré-entrée corrompant
  `_fit_res`/entry. Le batch (séquentiel + QProgressDialog modal) et le zone-runner
  (`run_full_fit` direct) ne sont pas bloqués. Vrai QThreadPool différé (refactor
  concurrence lourd, numpy GIL-bound). 2 tests.
- Fichiers : `arpes/app.py`, `arpes/core/session.py`,
  `arpes/ui/controllers/fit_runner_controller.py` ; tests
  `test_session.py` (+4), `test_fit_reentrancy_p37.py` (new, 2),
  `test_arpes_explorer_helpers.py` (patch obsolète).

## 2026-06-05 — P4 Publication & UX (batch)

Env `peaks` (PyQt6). Suite 810 OK / 9 skip ; launch réel vérifié.

- **P4.1 fond clair export** — `export_styles.savefig_with_preset(light_background=)`
  recolore une figure sombre (fig/axes/ticks/spines/labels) en clair puis
  RESTAURE les couleurs d'origine (canvas écran inchangé). Forcé pour les
  presets de publication (`LIGHT_BG_PRESETS`).
- **P4.2 labels LaTeX** — `$k_\parallel$/$k_x$/$k_y$ (π/a)`, `$E-E_F$ (eV)`,
  `$I/I_{\max}$`, `$\Gamma_k$ (HWHM, π/a)` dans results, fs_panel, ef_calibration,
  imag_self_energy, plots/{fermi_surface,mdc_diagnostics,mdc_edc,common}.
- **P4.3 presets Nature/Science + colorbars** — presets 7 pt sans-serif, PDF
  vectoriel par défaut ; `figure_size_mm` + `NATURE/SCIENCE_WIDTHS_MM` ;
  `savefig_with_preset(metadata=)` (PDF/EXIF) ; combo FS + cividis (CB-safe) + RdBu_r.
- **P4.4 logbook ranges** — hv [4,200]→[4,10000] eV (XPS/HAXPES), T [3,500]→
  [0.001,1000] K (mK→HT). Sniff numérique = repli après matching par nom.
  Dialogs modaux hv-manquant / BESSY ambigu : différés (UI interactive lourde).
- **P4.5 garde-fous adaptatifs** — `pocket_quality` tol bord = max(2, 2 % min(nx,ny)) ;
  SNR seuil 3.0→5.0 ; BESSY P-Axis off-center → drapeau structuré
  `meta["fs_p_axis_offcenter"]` (l'io ne peut pas ouvrir de dialog ; UI consomme).
- **P4.6 workflow** — onglet « Démarrage » (quickstart nouveau labo, additif,
  index 7) ; bandeau « EF non calibré » (BM, visible si ni fit poly ni offset).
  Stepper FS + réordonnancement onglets : différés (le reorder touche 15+ index
  d'onglet codés en dur dans 6 controllers → exige d'abord un refactor en index
  symboliques, pas un shuffle aveugle).
- **P4.7 sources de vérité** — preset poche (Fin/Standard/Stable) tracé dans
  l'export CSV (`preset_quality`) ; tooltip sp_ef : scalaire ignoré si calib EF
  polynomiale active. Bind a / refus double saisie : déjà couvert par P1.1.
- **P4.8 cleanup** — magic `0.052` → `DEFAULT_EF_OFFSET_EV` (session.py, partout) ;
  ZDB trigonal/triangulaire → hexagonale (alias bz.py + combo FS).
  « ⤢ Vue init » CONSERVÉ : ce n'est pas un doublon de Home mpl (autoscale sur
  les données courantes vs vue initiale figée). Warn Solaris pré-normalisé :
  non implémentable (pas de widget chk_norm ni de métadonnée hardware-norm).
- Tests : `tests/test_export_styles.py` +3 (presets Nature/Science, figure_size_mm,
  recolor+restore).

## 2026-06-05 — P3.1 Forwarding controllers fail-loud on unknown write

- Scope: 8 controllers forward attribute access to a parent/panel via blanket
  `__getattr__`/`__setattr__` ("god-object camouflé"). Blind write-forward = a
  typo silently created a junk attribute on the parent while real state went
  stale (silent-correctness). Reads already fail loud (typo read raises through
  the parent), so only writes were hardened.
- Prerequisite found this session: env micromamba `peaks` has PyQt6 → the 10 Qt
  tests that had never run locally now execute. Fixed 4 stale Qt tests first
  (test_load_cache lattice fixture, test_ui_smoke "Compare pol" tab removed,
  test_pocket_controller fake `_fs_controls`/canvas + over-smoothing guard).
- Method: complete *static* write inventory (not runtime probe, which
  undercounts untested paths) — `self.X=` in each controller + `ctrl/p.X=` in
  helper free-functions (mdc_edc_drawer `_kf_drag_lines`, gamma_lifecycle
  `_sel_k`) + confirmed no dynamic `setattr` and no external `_ctrl.X=` writes.
- Change: each controller gains `_OWN_ATTRS` + `_PARENT_WRITES` frozensets;
  `__setattr__` routes own→`object.__setattr__`, allow-listed→parent, else
  raises `AttributeError`. `__getattr__` unchanged.
  - plot: `_data_disp{,_ev,_kpar}`, `_disp_cache_key`,
    `_distortion_display_info`, `_grid_display_info`, `_kf_drag_lines`
  - fs: `_fs_distortion_cache` · norm: `_grid_display_info`
  - gamma: `_fs_pick_center_active`, `_sel_k`
  - pocket: own preview seeds only · distortion/browser/pairing: no writes
- Files: `arpes/ui/controllers/{distortion,browser,plot,fs,norm,gamma,pocket,
  pairing}_controller.py`; `tests/test_controller_forward_p31.py` (new, 3 tests).
- Verification (env `peaks`): full suite 806 passed / 9 skipped; real
  `ArpesExplorer` launch OK (offscreen, event loop); typo write raises +
  legit write forwards confirmed at runtime.

## 2026-06-05T03:18:51Z — P1.1 Tranche 1

- Scope: introduce central `SampleConfig` model without removing existing fallbacks yet.
- Files changed:
  - `arpes/core/sample.py`
  - `arpes/core/session.py`
  - `tests/test_session.py`
- Behavior:
  - Added explicit sample metadata model with formula, lattice `a/c`, work function, space group, MP id, and lattice source.
  - Added `Session.current_sample` as a session-level default.
  - Added `FileMeta.sample_config` plus compatibility fields for missing sample data.
  - Preserved legacy `FileMeta.crystal_a_angstrom`, `formula`, and `mp_id` so old sessions still load.
- Verification:
  - `python3 -m pytest tests/test_session.py` -> 9 passed
  - `python3 -m pytest tests/test_session.py tests/test_models.py` -> 16 passed

## 2026-06-05T03:25:00Z — P1.1 Tranche 2

- Scope: pass known sample lattice `a` into raw loaders and cache identity.
- Files changed:
  - `arpes/io/loader_orchestrator.py`
  - `arpes/ui/controllers/load_controller.py`
  - `arpes/app_angle_offsets.py`
  - `tests/test_loader_orchestrator.py`
- Behavior:
  - `LoadController` resolves `SampleConfig` for the file before raw load.
  - `LoaderOrchestrator.load` forwards `a_lattice` only when it is known and positive.
  - Best-angle CLS reload path receives the same `a_lattice`.
  - Raw-load cache version/key now includes lattice `a`, preventing stale k-space data after sample changes.
- Verification:
  - `python3 -m pytest tests/test_loader_orchestrator.py tests/test_session.py` -> 15 passed
  - `python3 -m pytest tests/test_load_cache.py tests/test_loader_orchestrator.py` -> 6 passed, 10 skipped

## 2026-06-05T03:35:00Z — P1.1 Tranche 3

- Scope: remove silent `4.143 Å` fallback from publishable physics exports/aggregation.
- Files changed:
  - `arpes/core/sample.py`
  - `arpes/io/export.py`
  - `arpes/analysis/aggregation.py`
  - `arpes/ui/widgets/results.py`
  - `tests/test_export.py`
  - `tests/test_multifile_aggregation.py`
- Behavior:
  - Added `require_lattice_a` for user-facing refusal when lattice `a` is missing.
  - `physics_rows` now reads `SampleConfig` and raises instead of using BaNi-like fallback.
  - Multi-file aggregation skips entries without lattice `a` and records a warning.
  - Results panel displays an explicit missing-lattice row instead of silently computing with fallback.
  - Physics export dialog shows the missing-lattice error without writing a file.
- Verification:
  - `python3 -m pytest tests/test_export.py tests/test_multifile_aggregation.py tests/test_session.py` -> 18 passed
  - `python3 -m pytest tests/test_ui_smoke.py` -> 16 skipped
  - `python3 -m pytest tests/test_export.py tests/test_multifile_aggregation.py tests/test_session.py tests/test_analysis_results.py` -> 33 passed

## 2026-06-05T03:45:00Z — P1.1 Tranche 4

- Scope: remove UI-side `4.143 Å` fallbacks that made unknown lattice look valid.
- Files changed:
  - `arpes/ui/controllers/band_analysis_controller.py`
  - `arpes/ui/controllers/load_controller.py`
  - `arpes/ui/widgets/params_theory.py`
  - `tests/test_band_analysis_controller.py`
- Behavior:
  - Band analysis now resolves lattice `a` from `SampleConfig`/entry metadata and returns `0.0` when unknown.
  - Loading a file restores `sp_crystal_a=0.0` when metadata lacks lattice `a`.
  - Theory/analysis lattice spinbox starts at `0.0`, making “unknown” representable.
- Verification:
  - `python3 -m pytest tests/test_band_analysis_controller.py tests/test_band_analysis_extras.py tests/test_load_cache.py` -> 26 passed, 14 skipped
  - `python3 -m pytest tests/test_band_analysis_controller.py tests/test_band_analysis_extras.py tests/test_load_cache.py tests/test_export.py` -> 32 passed, 14 skipped

## 2026-06-05T03:55:00Z — P1.1 Tranche 5

- Scope: prefer `SampleConfig.work_function_eV` during raw loading/cache while keeping UI `sp_phi` fallback.
- Files changed:
  - `arpes/core/sample.py`
  - `arpes/io/loader_orchestrator.py`
  - `arpes/ui/controllers/load_controller.py`
  - `arpes/app.py`
  - `arpes/app_angle_offsets.py`
  - `tests/test_loader_orchestrator.py`
  - `tests/test_load_cache.py`
- Behavior:
  - Raw load resolves work function from sample metadata before manual UI `sp_phi`.
  - Raw-load cache key uses the effective work function, preventing stale k-space data after sample φ changes.
  - CLS best-angle reload path receives the same effective work function.
- Verification:
  - `python3 -m pytest tests/test_loader_orchestrator.py tests/test_session.py` -> 16 passed
  - `python3 -m pytest tests/test_load_cache.py tests/test_loader_orchestrator.py tests/test_session.py` -> 16 passed, 12 skipped
  - `python3 -m py_compile arpes/core/sample.py arpes/io/loader_orchestrator.py arpes/app_angle_offsets.py` -> passed

## 2026-06-05T04:05:00Z — P1.1 Tranche 6

- Scope: use effective sample work function in Γ/KZ/FS post-load physics paths.
- Files changed:
  - `arpes/ui/controllers/gamma_controller.py`
  - `arpes/ui/controllers/kz_controller.py`
  - `arpes/ui/controllers/fs_controller.py`
  - `tests/test_session.py`
- Behavior:
  - Γ angle conversion/resolver now uses `SampleConfig.work_function_eV` before UI fallback.
  - KZ stack load and KZ map computation use effective sample work function.
  - FS kz label computation uses effective sample work function instead of session-global `4.031`.
- Verification:
  - `python3 -m pytest tests/test_session.py tests/test_gamma.py tests/test_kz.py` -> 59 passed
  - `python3 -m py_compile arpes/ui/controllers/gamma_controller.py arpes/ui/controllers/kz_controller.py arpes/ui/controllers/fs_controller.py` -> passed

## 2026-06-05T04:15:00Z — P1.1 Tranche 7

- Scope: stop treating historical `φ=4.031 eV` as session/UI truth.
- Files changed:
  - `arpes/core/session.py`
  - `arpes/ui/widgets/params_ef.py`
  - `arpes/ui/controllers/load_controller.py`
  - `arpes/ui/controllers/gamma_controller.py`
  - `arpes/ui/controllers/kz_controller.py`
  - `arpes/ui/controllers/fs_controller.py`
  - `arpes/ui/controllers/pairing_controller.py`
  - `tests/test_session.py`
- Behavior:
  - New sessions default `work_func=0.0` (unknown), not `4.031`.
  - Legacy sessions without `work_func` load as unknown; legacy sessions with explicit `work_func` preserve it.
  - UI φ spinbox starts at `0.0` and documents that physical loading needs φ or `SampleConfig.work_function_eV`.
  - Raw loading refuses missing φ instead of computing with a hidden default.
  - Γ/KZ/FS/pairing fallback to session/UI/sample values only; no hardcoded `4.031` remains in those controllers.
- Verification:
  - `python3 -m pytest tests/test_session.py tests/test_loader_orchestrator.py tests/test_gamma.py tests/test_kz.py` -> 67 passed
  - `python3 -m py_compile arpes/ui/controllers/load_controller.py arpes/ui/controllers/pairing_controller.py arpes/ui/widgets/params_ef.py arpes/ui/controllers/gamma_controller.py arpes/ui/controllers/kz_controller.py arpes/ui/controllers/fs_controller.py` -> passed
  - `rg -n "4\\.031" <touched files>` -> no matches

## 2026-06-05T04:30:00Z — P1.1 Tranche 8

- Scope: remove remaining hidden `a=3.96 Å` defaults from app/physics/io paths.
- Files changed:
  - `arpes/core/sample.py`
  - `arpes/physics/gamma.py`
  - `arpes/physics/bm_cut_overlay.py`
  - `arpes/physics/fs.py`
  - `arpes/physics/kz.py`
  - `arpes/physics/resolution.py`
  - `arpes/io/kz_dataset.py`
  - `arpes/io/loaders/{common,cls,bessy,solaris}.py`
  - `arpes/ui/widgets/{fs_panel,kz}.py`
  - `arpes/ui/controllers/{fs_controller,gamma_controller,pairing_controller}.py`
  - `tests/test_gamma.py`
  - `tests/test_bm_cut_overlay.py`
- Behavior:
  - `a_lattice=0.0` now means unknown; callers must pass `SampleConfig.a` for angle/k conversion.
  - Γ and BM-cut overlay receive lattice `a` from `SampleConfig` when available.
  - FS/KZ widgets start with unknown lattice instead of BaNi-like defaults.
  - Loader signatures no longer default to `3.96`.
- Verification:
  - `python3 -m pytest tests/test_gamma.py tests/test_bm_cut_overlay.py tests/test_fs.py tests/test_kz.py tests/test_session.py` -> 76 passed, 15 skipped
  - `python3 -m pytest tests/test_loader_orchestrator.py tests/test_loaders_integration.py tests/test_arpes_io.py tests/test_kz_dataset.py tests/test_resolution.py` -> 22 passed, 9 skipped
  - `python3 -m py_compile <touched modules>` -> passed
  - `rg -n "3\\.96|4\\.031" arpes/physics arpes/io arpes/ui` -> no matches

## 2026-06-05T04:45:00Z — P1.2

- Scope: make pocket characterization prefer publication-grade MDC-radial kF.
- Files changed:
  - `arpes/physics/pocket.py`
  - `tests/test_pocket.py`
- Behavior:
  - `characterize_pocket` tries radial MDC Lorentzian fits first in publication mode.
  - Iso-contour remains fallback/preview when MDC fails or gives a sub-grid radius.
  - `PocketProperties` now records `analysis_mode`, `mdc_valid_directions`, and `mdc_total_directions`.
- Verification:
  - `python3 -m pytest tests/test_pocket.py tests/test_pocket_mdc_radial.py` -> 37 passed
  - `python3 -m pytest tests/test_pocket.py tests/test_pocket_mdc_radial.py tests/test_pocket_controller.py tests/test_pocket_quality.py` -> 46 passed, 5 skipped

## 2026-06-05T05:00:00Z — P1.3

- Scope: apply Luttinger spin factor and signed hole-pocket convention.
- Files changed:
  - `arpes/analysis/results.py`
  - `arpes/analysis/bootstrap.py`
  - `arpes/physics/pocket.py`
  - `tests/test_analysis_results.py`
  - `tests/test_analysis_bootstrap.py`
  - `tests/test_pocket.py`
- Behavior:
  - Luttinger density now includes the spin degeneracy factor used by the audit.
  - Branch/bootstrap payloads include `luttinger_units`.
  - Hole pockets now report negative `n_carriers_2D`.
- Verification:
  - `python3 -m pytest tests/test_analysis_results.py tests/test_analysis_bootstrap.py tests/test_pocket.py` -> 52 passed
  - `python3 -m py_compile arpes/analysis/results.py arpes/analysis/bootstrap.py arpes/physics/pocket.py` -> passed

## 2026-06-05T05:15:00Z — P1.4

- Scope: make EF-window integration explicit and warn on asymmetric windows.
- Files changed:
  - `arpes/physics/fs.py`
  - `tests/test_fs.py`
- Behavior:
  - FS extraction now reports empty/asymmetric EF integration windows in the title.
  - `FSParams` adds `ef_resolution_meV` and `temperature_K`.
  - If resolution/temperature are provided, EF integration uses Fermi/resolution weighting; otherwise it reports boxcar EF integration.
- Verification:
  - `python3 -m pytest tests/test_fs.py` -> 2 passed, 17 skipped
  - `python3 -m py_compile arpes/physics/fs.py` -> passed
  - Pure `extract_fs_map` smoke command -> asymmetric warning and Fermi/resolution title asserted

## 2026-06-05T05:35:00Z — P1.5

- Scope: add export provenance for reproducible results and figures.
- Files changed:
  - `arpes/io/export.py`
  - `arpes/ui/widgets/results.py`
  - `tests/test_export.py`
- Behavior:
  - CSV exports from the Results panel now start with audit-required `#` provenance headers.
  - CSV exports also write a sibling `.meta.json` sidecar for tools that cannot parse comment headers.
  - Figure `.meta.json` sidecars include provenance limited to the files visible in the exported figure.
  - Provenance records git commit, session version, UTC timestamp, input hash, input file identity, sample config, fit params, EF corrections, BM distortion, pocket settings, gamma state, angles, hv, temperature, polarization, and instrument/source metadata.
  - Physics CSV rows now include `luttinger_units`.
- Verification:
  - `python3 -m pytest tests/test_export.py tests/test_session.py` -> 22 passed
  - `python3 -m py_compile arpes/io/export.py arpes/ui/widgets/results.py` -> passed
  - Subagent review findings addressed: CSV sidecar added; figure provenance filtered to visible files.

## 2026-06-05T05:55:00Z — P1.6

- Scope: bump `Session.VERSION`, explicit cross-version handling, atomic save.
- Files changed:
  - `arpes/core/session.py`
  - `tests/test_session.py`
- Behavior:
  - `Session.VERSION` bumped 1 -> 2 (3 fields `band_analysis`, `fit_zones`, `active_zone_id` had been added without a bump).
  - `load_from_payload` now migrates via `_migrate_payload`: missing version -> treated as v1, version <= current -> migrated in place, version > current -> raises `SessionVersionError` instead of silently dropping unknown fields.
  - `Session.loaded_version` records the payload version actually read (None for a fresh session).
  - `save()`/`save_to()` go through `_atomic_write_json`: write to a sibling `.tmp`, fsync, then `os.replace`; `save()` rotates the previous file to `.bak` first (no truncation on crash).
- Verification:
  - `python3 -m pytest tests/test_session.py` -> 17 passed
  - `python3 -m pytest tests/test_session.py tests/test_models.py tests/test_export.py tests/test_loader_orchestrator.py` -> 40 passed
  - `python3 -m py_compile arpes/core/session.py` -> passed

## 2026-06-05T06:20:00Z — P2.1a (tilt guard)

- Council: arbiter split P2.1 into P2.1a (mapping + tilt guard, shipped now) and
  P2.1b (full Ishida & Shin 3-angle formula + per-lab sign calibration, deferred
  after P2.6). architect+physicist over redteam: a guard that DISABLES the
  overlay above 2° cannot invert a sign, so P2.6 sign work does not block it.
- Scope: surface tilt to physics, refuse BM↔FS projection when |tilt| > 2°,
  warn on residual ky in the 0–2° grey zone. No tilt correction yet (P2.1b).
- Files changed:
  - `arpes/physics/kpar_geometry.py` (new) — single source for `C_ARPES`,
    `kpar_scale`, `tilt_within_guard`, `ky_residual_pi_a`, `TILT_GUARD_DEG`.
  - `arpes/physics/gamma.py` — import `C_ARPES` from kpar_geometry (drop local
    dup); tilt guard in `gamma_reference_to_bm_center`.
  - `arpes/physics/bm_cut_overlay.py` — import shared helpers (drop local
    `C_ARPES` + reduce `_scale_factor` to a thin wrapper); tilt guard + residual
    ky warning in `compute_bm_cut_in_fs_frame`.
  - `arpes/io/loaders/common.py` — `_C_ARPES` now aliases kpar_geometry.C_ARPES.
  - `arpes/io/loader_orchestrator.py` — `apply_loaded_metadata` propagates raw
    motor tilt (`tilt`/`tilt_ref`) into `FileMeta.tilt` without clobbering a
    logbook-set value (fixes redteam #1: guard could not see the real tilt).
  - `tests/test_kpar_geometry.py` (new), `tests/test_bm_cut_overlay.py`,
    `tests/test_pairing_controller.py`.
- Behavior:
  - `C_ARPES = 0.51233` is defined once; gamma/bm_cut_overlay/common all import it.
  - BM↔FS overlay and Γ FS→BM projection return disabled/NaN with a clear message
    when |tilt| (BM or FS) > 2°.
  - In 0 < |tilt| ≤ 2°, the overlay is kept but its warning reports the estimated
    uncorrected ky error (`≈ scale·sin(tilt)` π/a).
  - tilt absent (None) reproduces the historical result exactly (regression test).
- Pre-existing failure fixed: `test_pairing_controller` assumed the removed
  `a=3.96` default (P1.1); `_fs` fixture now sets `crystal_a_angstrom` explicitly.
- Known follow-ups (out of P2.1a scope):
  - `arpes/physics/resolution.py:62` still inlines `0.51233` in a formula — 4th
    `C_ARPES` site, not in the 3 the arbiter named; fold into kpar_geometry later.
  - P2.1b: real tilt correction (common.py:364 missing `cos(tilt)`,
    bm_cut_overlay.py:199 ignores tilt, gamma.py:200 polar-only), per-lab sign
    calibration (P2.6), and tilt/azi per-file in export provenance.
- Verification:
  - `python3 -m pytest tests/test_kpar_geometry.py tests/test_bm_cut_overlay.py tests/test_gamma.py tests/test_loader_orchestrator.py` -> 77 passed
  - `python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py` -> 652 passed, 79 skipped
  - `python3 -m py_compile` on all touched modules -> passed
  - `grep -rn "= 0.51233" arpes/` -> only kpar_geometry.py (+ resolution.py noted above)

## 2026-06-05T07:10:00Z — P2.6a (data-driven sign + confidence)

- Constraint (user): colleagues rarely measure an Au reference, so the per-beamline
  angle sign convention CANNOT be frozen by Au calibration in the common case. The
  audit P2.6 recommendation ("calibrate once with Au, drop the enumeration") is not
  viable; selection must stay data-driven, with an optional freeze path for later.
- Council: arbiter split P2.6 into P2.6a (this, minimal safe), P2.6b (provenance of
  scoring windows + UI freeze), P2.6c (off-Γ scorer rework, data-driven later).
  redteam CAS1 (scorer minimises abs(gamma) ⇒ assumes Γ_true=0 ⇒ wrong sign for a
  real off-Γ pocket) enters P2.6a as exposure only: surface `gamma_residual_after`
  and warn, full scorer rework deferred to P2.6c. CAS5 (ef_offset sentinel) → P4.8.
- Files changed:
  - `arpes/physics/angle_convention.py` (new) — `BeamlineAngleConvention` dataclass,
    `UNCALIBRATED` sentinel, `ConventionRegistry`, `convention_key` (beamline, hv±5,
    azi±5, polar±2 — redteam CAS4 anti cross-contamination), `get/freeze_convention`,
    `filter_candidates`, `select_best_candidate(score_fn injected)`,
    `evaluate_confidence`. No PyQt.
  - `arpes/physics/gamma.py` — new `score_bm_gamma_residual_detail` returns
    `{score, gamma, mad, n, gamma_residual_after}`; the float scorer now delegates to it.
  - `arpes/app_angle_offsets.py` — `load_with_best_angle_offsets` filters by frozen
    convention, selects via `select_best_candidate`, computes confidence + tie,
    stores `angle_offset_{confidence,ambiguous,candidate_score_2nd,gamma_residual_after}`
    in metadata, and `_emit_sign_warnings` raises status warnings for tie / ambiguity /
    refusal / convention-change-since-last-session (redteam CAS2/CAS6).
  - `arpes/core/session.py` — `convention_registry` field, `VERSION` 2→3, persisted in
    payload, v2→v3 migration (absent → empty dict).
  - `tests/test_angle_convention.py` (new).
- Behavior:
  - Sign selection stays data-driven; the chosen sign now carries a confidence and an
    ambiguity verdict instead of being silent. Ambiguous/refused cases warn the user.
  - When a beamline convention is frozen (future Au calibration / manual), the
    enumeration is restricted to that sign; empty registry = everything UNCALIBRATED.
  - `gamma_residual_after` exposes an off-Γ pocket (kF bias) that the abs(gamma) score
    cannot see by itself.
- Deferred: P2.6b (capture scoring windows sp_evs/eve/kmin/kmax/xg/sfd into provenance
  + UI freeze via convention_action verb-dispatch), P2.6c (off-Γ scorer), P4.8 ef_offset.
- Verification:
  - `python3 -m pytest tests/test_angle_convention.py tests/test_gamma.py tests/test_session.py` -> 76 passed
  - `python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py` -> 671 passed, 79 skipped
  - `python3 -m py_compile` on all touched modules -> passed

## 2026-06-05T (P2.2 — orthogonal regression + linearity gate + block bootstrap)

- Audit P2.2 targets had moved in the α→σ refonte: `results.py`/`bootstrap.py`
  are now under `arpes/analysis/`. Two parallel vF/kF/m* extractors existed
  (`analysis/results.extract_branch_result` for the table/export, and
  `physics/fit.compute_fermi_velocity_mstar` for the Im Σ path); both fitted the
  dispersion with a vertical-only regression and trusted kF=−α/β with no
  linearity check. Both were brought to the same rigour via one shared module.
- Council (architect + physicist + redteam, sonnet/caveman, 1 round):
  - architect: scipy in physics/ is fine (no PyQt); refused result must return NaN
    in the legacy keys (consumers don't read a flag → a wrong value is worse than
    NaN); additive keys preserve backward compat.
  - physicist: blockers — (1) orthogonal regression with UNIT weights is false
    precision (just rotates the residual metric 45° in heterogeneous units); needs
    real per-point σ_k, available as σ_kF (results) / Γ proxy (Im Σ path). (2) m*
    error must use the full 2×2 covariance (slope↔intercept correlated, same fit);
    independence underestimates σ by ~2× in tight windows. Linearity gate sound;
    require a minimum point count first.
  - redteam: must-fix — convergence/degeneracy guards (constant-k, constant-E,
    n=3 quadratic interpolates exactly → gate meaningless), NaN/inf checks on
    fitted params, keep the `|slope|<1e-9` floor, n≥5 absolute minimum (a shrinking
    window trivially passes the linearity gate otherwise), flag that σ is a
    regression uncertainty unless real measurement errors are supplied.
- Decision: synthesis, no arbiter needed (verdicts compose). Orthogonal regression
  by total least squares (eigen-decomposition of the (σ_k, σ_E)-scaled point cloud)
  with covariance from leave-one-out jackknife — NOT scipy.odr, which is deprecated
  in SciPy ≥1.17 and removed in 1.19. OLS `polyfit(cov=True)` fallback when no σ_k.
- Files changed:
  - `arpes/physics/dispersion_fit.py` (new) — `linear_dispersion_fit(k, e, sk)`
    (TLS if σ_k else OLS, returns slope/intercept/2×2 cov/method), `curvature_ratio`
    (|a|·Δk/|b| from the quadratic fit), thresholds `MIN_DISP_POINTS=5`,
    `CURVATURE_MAX=0.10`, `SLOPE_FLOOR=1e-9`. No PyQt, no deprecated deps.
  - `arpes/physics/fit.py` — `compute_fermi_velocity_mstar` rewritten: n≥5 gate,
    linearity gate, TLS/OLS fit, σ_vF/σ_kF/σ_m* via full 2×2 covariance, new keys
    `vF_sigma_eV_A, kF_inv_A_sigma, mstar_sigma, linear_ok, refused_reason,
    sigma_type, n_points, curvature_ratio`; legacy keys NaN on refusal.
  - `arpes/analysis/results.py` — `extract_branch_result` uses the shared TLS fit +
    linearity gate, σ_kF and σ_m* now keep the slope↔intercept covariance term;
    `BranchResult` gains `linear_ok` + `refused_reason` (refusal no longer silent).
  - `arpes/analysis/bootstrap.py` — iid point resampling replaced by moving-block
    bootstrap (`_block_length` 3–5 slices, `_moving_block_indices`): adjacent MDC
    slices are correlated, iid resampling underestimates σ.
  - `arpes/io/export.py` — `linear_ok` + `refused_reason` columns so a refused
    branch reports WHY (avoids silent NaN rows in CSV/LaTeX).
  - `tests/test_dispersion_fit_p22.py` (new, 17 tests).
- Behavior: kF/vF/m* now refuse (NaN + reason) when the band is non-linear near E_F
  or has <5 points, use orthogonal regression weighted by the real per-point k
  uncertainty, report correlated error bars, and bootstrap σ via correlated blocks.
- Known issue surfaced, NOT changed (out of P2.2 scope, changes published numbers):
  `analysis/results.HBAR2_OVER_ME_eV_A2 = 7.6199682e-2` vs `physics/fit._HBAR2_OVER_ME
  = 7.6199682`. ℏ²/m_e = 7.62 eV·Å², so the two m* paths differ by 100×. Needs a
  user decision before fixing (export values change).
- Deferred: ODR-per-iteration inside the bootstrap (kept weighted OLS for speed);
  per-point σ_E (currently a constant median energy step).
- Verification:
  - `python3 -m pytest tests/test_dispersion_fit_p22.py` -> 17 passed
  - `python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py` -> 688 passed, 79 skipped
  - No scipy.odr deprecation warning (dependency removed).

## 2026-06-05T (P2.4 — open pockets : conic ellipse, extrapolation flag, refuse)

- Reframed per user decision ("les deux"): a Fermi pocket that overflows the scan
  edges gives a partial ARC, not a closed contour. Default = fit the visible arc
  with an algebraic conic ellipse, FLAG it extrapolated (not publishable as-is) +
  error bar; REFUSE if the visible arc is too short. The PCA fit (`fit_pocket_ellipse`,
  axes = std·√2) was biased and meaningless on an open arc; the radial contour was
  chord-closed (`_close_contour` appends pt0) → shoelace + PCA saw a fake chord.
- Council (geometry + physicist + redteam, sonnet/caveman, 1 round):
  - geometry: Halir-Flusser 1998 (numerically-stable Fitzgibbon direct LSQ, constraint
    4ac−b²=1) works on arcs; matrix-form conic→geometric; bootstrap σ on axes; FIX
    `arc_coverage_deg` (N_ok/N_total) → contiguous angular span; pass the RAW open arc
    (never `_close_contour`) to the fitter.
  - physicist: Luttinger n_2D MUST be NaN when extrapolated (theorem needs the enclosed
    area; the unseen closure is assumed); REFUSE below ~120° arc (unseen semi-axis is a
    free param); σ must include MODEL error, not just fit scatter; widen by gap fraction.
  - redteam (3 blockers): (1) `characterize_pocket`'s `except Exception: pass` swallows a
    refusal then falls back to the force-closed isocontour preview → defeats the whole
    refuse logic; (2) guard the conic discriminant (4ac−b²≤0 → hyperbola → sqrt of neg →
    NaN axes) BEFORE sqrt + refuse short/collinear arcs; (3) `is_extrapolated` bool must
    not go through the bootstrap `_SCALAR_FIELDS` median — aggregate via any().
- Decision: synthesis. Adopted physicist's more conservative thresholds
  (ARC_REFUSE_DEG=120, ARC_FULL_DEG=340). Coverage measured about the FITTED ellipse
  centre (the arc centroid is biased for a short arc → false coverage).
- Files changed:
  - `arpes/physics/ellipse_conic.py` (new) — `fit_ellipse_conic` (Halir-Flusser, isotropic
    normalisation, discriminant guard, never NaN), `contiguous_coverage_deg` (360 − largest
    gap), `conic_axis_sigma` (bootstrap), `PocketFitRefusedError(ValueError)`, thresholds.
  - `arpes/physics/pocket.py` — `fit_pocket_ellipse` now conic-first (PCA renamed
    `_fit_pocket_ellipse_pca`, fallback for closed only); `_properties_from_contour`
    computes contiguous coverage, refuses <120°, conic fit on raw arc, `is_extrapolated`
    + `arc_coverage_deg` + `ellipse_fit_valid` + `kF_a_sigma`/`kF_b_sigma` (fit scatter ⊕
    gap-fraction model error) + `fit_method`, area = π·a·b when extrapolated, n_carriers_2D
    = NaN when extrapolated; topology confidence gate 0.25→0.50 (`_TOPOLOGY_CONFIDENCE_MIN`);
    `characterize_pocket` re-raises `PocketFitRefusedError` instead of the silent fallback.
    `PocketProperties` gains 6 fields.
  - `arpes/physics/pocket_bootstrap.py` (new) — split out `characterize_pocket_bootstrap` +
    `PocketBootstrap` + `_SCALAR_FIELDS` (pocket.py was 792 > 700 cap; now 670). Flags
    aggregated explicitly: is_extrapolated via any(), ellipse_fit_valid via all().
  - `arpes/ui/controllers/pocket_controller_mdc.py` — conic ellipse + contiguous coverage
    about the fitted centre, refuse <120° (status msg), extrapolation flag + axis σ +
    π·a·b area in the pocket dict.
  - `arpes/ui/controllers/pocket_controller.py` — bootstrap import moved to pocket_bootstrap;
    CSV export gains kF_a/b_sigma, is_extrapolated, fit_method, arc_coverage_contig_deg.
  - `tests/test_open_pocket_p24.py` (new, 11 tests); test_pocket.py + pocket_controller.py
    import update.
- Verified: conic recovers a/b/angle exactly on a FULL ellipse AND a 180° half-arc; 80°
  arc refused; 220° arc → extrapolated, n_2D NaN, area=π·a·b.
- Deferred (council, non-blocking): per-axis gap-orientation gate (which semi-axis lies in
  the gap → NaN only that one) — currently uniform gap-fraction σ widening; topology
  PARTIAL/UNKNOWN penalty by gap size; area>bz_area sanity cap; comb-pattern density guard
  (scattered valid fits vs one contiguous arc) beyond the contiguous-span metric.
- Verification:
  - `python3 -m pytest tests/test_open_pocket_p24.py tests/test_pocket.py` -> 43 passed
  - `python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py` -> 699 passed, 79 skipped
  - all touched files ≤ 700 LOC; no PyQt in physics/.

## 2026-06-05T (P2.3 self-energy cohérent + P2.5 kz/gap — P2 terminé)

User: finish P2, solo unless a real physics judgment forces council; subagents sonnet.
Did P2.3.1/.2/.3 + P2.5.1/.3 solo; P2.5.2 (Dynes→Norman) needed the physicist — one
sonnet council agent (a genuine correctness fork, see below).

### P2.3 — self-energy cohérent (arpes/analysis/self_energy.py)
- .1 Unified λ: `_estimate_kink` no longer uses the ad-hoc near/deep median-slope
  difference; λ now comes from `kink_analysis.extract_lambda` (λ = −∂ReΣ/∂ω|_{ω=0},
  linear fit on |ω|<50 meV) — single definition in the codebase. Returns
  (kink_energy, lambda_eff, lambda_err); `RealSelfEnergyResult` gains `lambda_err`.
- .2 Double-counting warning: `_double_counting_notes(source)` flags when the DFT
  source string looks renormalised (GGA+U / DFT+U / hybrid / HSE / PBE0 / B3LYP /
  SCAN / meta-GGA / mBJ / GW) → Re Σ = E_exp − E_DFT double-counts correlations;
  surfaced in `RealSelfEnergyResult.notes`. Plain GGA/PBE → no note.
- .3 vF per-pair: already satisfied — `imaginary_self_energy` (physics/fit.py) passes
  `pair_index` to `compute_fermi_velocity_mstar`, so vF is per-pair (fixed in refonte).
  Verified, no change needed.

### P2.5 — kz et gap
- .1 kz warnings (arpes/physics/kz.py): `kz_from_hv_kpar` warns (RuntimeWarning) on a
  negative radicand (k// > k_tot → kz undefined, free-final-state breaks); new
  `_warn_energy_center` + `FS_ENERGY_TOL_EV=0.05` warns in `compute_kz_map` /
  `compute_hv_k_map` when |E_center| > 0.05 eV (no longer the Fermi surface).
- .2 Dynes → Norman (arpes/physics/gap_extraction.py). COUNCIL (physicist, sonnet):
  the audit said "remove |Re[…]|, follow Norman 1998" — but naively dropping np.abs
  gives a NEGATIVE DOS at ω=0 (np.sqrt principal branch flips sign inside the gap;
  the abs was load-bearing). Resolution:
  * `dynes` fixed properly: force Im(denom) ≤ 0 (same sign as Im(ω−iΓ)) → positive
    DOS everywhere without the magnitude hack. Stays as the *tunneling* DOS (STS).
  * Added `norman_spectral(ω,Δ,Γ1,Γ0)` + `norman_multi` — Norman PRB 57 R11093 (1998)
    ARPES symmetrized-EDC spectral function A=−(1/π)Im G, Σ=−iΓ1+Δ²/(ω+iΓ0),
    positive-definite, peaks at ±Δ, single peak when Δ→0. Γ0 regularises the ω=0
    singularity (`_gamma0_for` = max(1, 0.3·resolution) meV, fixed not fitted).
    NOTE: the physicist's first code had a sign slip on ReΣ (gave a peak at 0); I
    re-derived ReΣ=+Δ²ω/(ω²+Γ0²) ⇒ ω−ReΣ zeros at ±√(Δ²−Γ0²) and verified ±Δ peaks.
  * Added `fit_norman_single` / `fit_norman_two_gap` (Γ0 regularised, Δ bound 100 meV,
    flag Γ1/Δ>0.5 = filled gap, flag Δ at bound vs phonon kink). Both observables
    coexist (physicist: Dynes=tunneling, Norman=ARPES).
  * Wired `band_analysis_controller` gap fit to `fit_norman_*` (ARPES EDC path).
- .3 Δ bound 50→100 meV (pseudogap) in fit_dynes_single/two_gap and the new Norman fits.

### Files
- self_energy.py, gap_extraction.py (+150 LOC, 465 total), kz.py, band_analysis_controller.py.
- tests/test_p2_self_energy_gap_kz.py (new, 20 tests).
### Verification
- `pytest tests/test_p2_self_energy_gap_kz.py tests/test_gap_extraction.py` -> all pass;
  Dynes positive, Norman peaks at ±Δ, fit recovers Δ=8.0, kz warns on neg radicand.
- `pytest tests/ --ignore=...annotations --ignore=...local_dft_loaders` -> 719 passed, 79 skipped.
- all touched files ≤ 700 LOC; no PyQt in physics/.
### P2 status: P2.1a/P2.2/P2.3/P2.4/P2.5/P2.6a DONE. Deferred sub-items: P2.1b (full
  Ishida), P2.6b/P2.6c. Still open decision: m* constant ×100 discrepancy (P2.2 note).

## 2026-06-05T (P2.1b géométrie tilt + P2.6c scorer off-Γ — sous-tranches différées)

User asked to finish the two deferred physics sub-tranches; explained novice-level first.

### P2.1b — conversion 6-axes angle→k (Ishida & Shin RSI 89,043903 2018)
COUNCIL (geometry, sonnet) gave the cumulative rotation-matrix result R=R_φ·R_β·R_θ
on n̂_det, the closed-form (kx,ky), 4 analytic validation invariants, and flagged the
slit-axis (x vs y) convention ambiguity.
- `arpes/physics/kpar_geometry.py` — new `kpar_from_angles(slit, polar, tilt, azi, *,
  ek, slit_axis='x')` → (kx,ky) Å⁻¹. mx=cosθ·sinα ; my=cosβ·(sinθ·sinα)−sinβ·cosα ;
  R_φ rotates (mx,my). Reduces EXACTLY to the historic 1-angle `C·√Ek·sinα, ky=0`
  when tilt=azi=polar=0. Sign of angles NOT hardcoded (calibrate from data, P2.6).
- Wiring (real user win — tilt no longer blocks the overlay): the app maps polar→ky
  (FS = polar scan), so manipulator tilt (rotation about the slit axis) shifts ky
  ADDITIVELY with polar, and the BM cut is drawn as a constant-ky line → the tilt
  offset is exact at the cut centre. So:
  * `bm_cut_overlay.py` — removed the hard `tilt>2° → incompatible` refusal; ky now
    = scale·(sin(polar_bm−polar_fs) + sin(tilt_bm−tilt_fs)). Note only when |Δtilt|>10°
    (constant-ky line drifts far from centre). Stale guard imports dropped.
  * `gamma.py gamma_reference_to_bm_center` — the returned `gamma` is the kx position
    of Γ; tilt shifts ky not kx (1st order), leaking into kx only via the azimuth
    rotation (2nd order). Hard refusal → soft note; projection proceeds.
- Validation WITHOUT real tilt data: tests/test_kpar_6axis_p21b.py — (a) reduction to
  1-angle, (b) |k|²≤C²Ek norm bound, (c) azimuth rotation covariance, (d) normal
  emission → 0, + tilt ky-shift sign. test_bm_cut_overlay tilt tests rewritten P2.1a→b
  (corrected, not refused).
- NOTE: geometry agent flagged a genuine convention question (manipulator-rotation vs
  deflector two-angle, and slit∥x vs y). The engine is correct/tested for manipulator
  rotations; the overlay wiring uses the constant-ky-line model already in the app.
  Per-beamline slit/sign convention still to be confirmed against a known Γ scan before
  using `kpar_from_angles` for full 2-D FS remapping (loaders still use the 1-angle
  conversion; that re-plumbing is the remaining P2.1 work).

### P2.6c — scorer signe angle pour bandes hors-Γ
- `arpes/physics/gamma.py` — `score_bm_gamma_residual[_detail]` gain `gamma_expected`
  (π/a, default 0.0). Old score `|gamma|` ASSUMED Γ_true=0 → wrong sign for a real
  off-Γ band (redteam CAS1). Now `|gamma − gamma_expected|`; `center_guess` also uses
  it; `gamma_residual_after = gamma − gamma_expected` (deviation from expected).
  Default 0 = unchanged behaviour.
- `arpes/app_angle_offsets.py` — `_score_detail` + the candidate scorer thread
  `gamma_expected` from `entry.meta_gamma_state["gamma_expected"]` (default 0). A known
  off-Γ band can set it so the sign selection no longer forces the band toward Γ.
- tests/test_p26c_scorer.py (3).

### Verification
- `pytest tests/test_kpar_6axis_p21b.py tests/test_bm_cut_overlay.py tests/test_gamma.py
  tests/test_p26c_scorer.py` -> all pass.
- full suite -> 730 passed, 79 skipped. All touched files ≤700 LOC; no PyQt in physics/.
### Remaining: P2.1 loader re-plumbing to `kpar_from_angles` (needs per-beamline slit/sign
  convention confirmed against a Γ scan). m* ×100 constant decision (P2.2) still open.

## 2026-06-05T (FIX bug m* ×100 — décision user)

- `arpes/analysis/results.py` : `HBAR2_OVER_ME_eV_A2 = 7.6199682e-2` → `7.6199682`.
  ℏ²/m_e = 7.61996 eV·Å² (réf ℏ²/2m_e = 3.80998). L'ancienne valeur (e-2) sous-estimait
  m*/m_e d'un facteur 100 dans la table exportée (extract_branch_result → export CSV/
  LaTeX) et le bootstrap (qui importe la constante). Le chemin Im Σ (physics/fit.py,
  `_HBAR2_OVER_ME = 7.6199682`) était déjà correct ; les deux sont désormais cohérents.
- Impact : toute valeur m* exportée AVANT ce commit est 100× trop petite — re-exporter.
- Tests : aucun ne figeait la valeur fausse (test_m_star_with_crystal_a dérive l'attendu
  de la constante du module → suit automatiquement). 730 passed, 79 skipped.

## 2026-06-05T (P3.3 + P3.5 + P3.6 partiel — petits/sûrs, exécution Sonnet)

Execution delegated to a Sonnet caveman subagent (mechanical, audit-specified, no design
latitude); reviewed + verified on the main thread.

- P3.3 — single-setter respecté partout. `arpes/core/fit_result_store.py` : nouveau
  `restore_fit_result(entry, *, fit_result, fit_zones)` (rollback complet legacy slot +
  liste de zones, sans le mirroring d'active-zone de set_fit_result).
  `gamma_controller.py:398` : la mutation directe `entry.fit_result = backup_fit ;
  entry.fit_zones = backup_zones` (rollback après save échoué) → appel `restore_fit_result`.
  Plus AUCUNE écriture directe `entry.fit_result =` hors du store (vérifié rg).
- P3.5 — import circulaire latent : `arpes/ui/builders/panels.py` — 5 imports lazy
  `from arpes.app import X` re-sourcés directement vers ui.widgets :
  FileBrowserPanel→browsers, FitParamsPanel→params, MplCanvas→canvas (×2),
  ResultsPanel→results. Casse le cycle panels→app→widgets.
- P3.6 partiel — `arpes/app.py` : (a) suppression du cimetière de ~15 lignes vides
  post-extraction (607→552 LOC) ; (c) retrait de l'artefact `arpes_plots(1).py`
  ("downloaded again") de la liste de fallback `_load_ap`.
- DIFFÉRÉ (pas "petit/sûr") : P3.6(b) try/except qui set 5 globaux à None → raise au
  démarrage (changement de comportement, ERLAB_OK est un flag de dégradation gracieuse) ;
  P3.6(d) extraction `_install_controllers` (ordre d'init controllers vs QTimer, règle
  CLAUDE.md #5 — à faire avec P3.1/P3.2).
- Tests : test_fit_result_store.py +2 (restore). Full suite 732 passed, 79 skipped.
  py_compile OK sur les 4 fichiers ; aucune mutation fit_result directe restante.

## 2026-06-05T (P3.2 — plafonds LOC restaurés)

- État réel vs audit (chiffres 2026-06-04 périmés) : SEUL `fs_panel.py` (706) violait le
  cap 700. **PROXY_MAP = 143** (pas 162) → déjà sous le cap 150, verb-dispatch déjà fait
  en refonte, RIEN à faire. pocket_controller (616), fit_runner (594), load (634),
  interaction (602) sont SOUS 700 → dette à surveiller, pas des violations → non splittés.
- `fs_panel.py` 706→635 LOC : extraction des 2 groupes BZ (ZDB théorique + Mapping BZ
  cristal MP) de `_build()` vers `arpes/ui/widgets/fs_panel_bz_controls.py` (nouveau,
  101 LOC) via pattern free-function `build_bz_theoretical_group(panel,lay)` +
  `build_bz_crystal_group(panel,lay)`. Exécution Sonnet caveman ; revue diff stricte
  (87 substitutions self→panel, signaux/tooltips/attributs byte-for-byte préservés).
- Vérif : py_compile OK ; consumers (fs_controller.py lit .ed_mp_id/.sp_v0/.chk_bz_xtal)
  intacts (attrs posés sur panel au _build) ; full suite 732 passed ; ui_smoke skip (PyQt
  absent local → câblage Qt non vérifiable headless, garanti par revue + compile).
