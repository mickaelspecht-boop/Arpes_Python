# Audit Updates History

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
