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
