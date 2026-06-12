# ARPES Explorer

Desktop application for ARPES visualization, calibration, fitting, and data-analysis workflows.

[![CI](https://github.com/mickaelspecht-boop/Arpes_Python/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mickaelspecht-boop/Arpes_Python/actions/workflows/build.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Version](https://img.shields.io/badge/version-v1.0.2-informational)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

## Summary

ARPES Explorer is a PyQt6 desktop application for working with angle-resolved photoemission spectroscopy data. It combines file loading, logbook metadata, band-map and Fermi-surface visualization, energy/momentum calibration, MDC fitting, result tables, and scientific export. The codebase separates pure numerical routines from the Qt interface so that loaders, physics helpers, and result calculations can be tested headlessly. The project is actively developed; several workflows are mature, while some beamline-specific loaders and advanced analyses should still be validated on new datasets before publication use.

## Why This Project?

ARPES datasets are often large, metadata-rich, and sensitive to calibration choices. A useful analysis tool needs more than plotting: it must keep track of photon energy, work function, lattice constants, EF offsets, momentum conventions, fit windows, rejected points, and export provenance.

ARPES Explorer focuses on an end-to-end workflow:

- inspect band maps and Fermi-surface volumes;
- keep file, logbook, and sample metadata together;
- fit MDC peak pairs and review slice-level diagnostics;
- compare fitted dispersions with optional DFT references;
- export tables, figures, and provenance in a reproducible way.

## Features

### Existing

- PyQt6 desktop interface with tabs for BM, FS, FS Explorer, KZ, Results, MDC Fit, Notes, Help, and Start.
- Session persistence through `.arpes_session.json`, including loaded files, sample metadata, EF offsets, fit parameters, fit zones, notes, annotations, and results.
- Loader registry with strict `ARPESData` validation and a common internal convention:
  - `data` as `(n_k, n_E)`;
  - energy axis as `E - EF` in eV;
  - FS volumes as `(n_ky, n_kx, n_E)`;
  - momentum axes in `pi/a` when calibrated.
- Supported experimental loaders:
  - Solaris/DA30 via optional `erlab`: `.ibw`, `.pxt`, `.zip`;
  - BESSY Scienta/SES R8000 Igor Binary Wave v5: `.ibw`;
  - CLS/LNLS text: individual BM files with `*_param.txt`, or FS folders with `*_Cycle_*_Step_*.txt`;
  - ALLS SpecsLab Prodigy Igor Text exports: `.itx`.
- Browse-only mode for incomplete metadata, keeping raw axes instead of forcing a momentum calibration.
- Band-map display modes: Raw, EDC normalization, second derivative, curvature.
- EF calibration tools, Gamma centering, BM/FS pairing, distortion preview/correction, and FFT grid-artifact display filtering.
- MDC fitting with Lorentzian peak pairs, fit zones, waterfall diagnostics, point deletion/undo/redo, annotations, and batch fitting.
- Results tab with kF dispersion, Gamma(E), slice diagnostics, physical result tables, bootstrap uncertainty option, multi-file analysis, and figure/table export.
- Fermi-surface tools: BZ overlays, compatible BM cut overlays, pocket measurements, FS map export.
- FS Explorer for calibrated FS volumes: interactive iso-energy map, draggable cut line, extracted BM, angle/energy sweep.
- KZ maps for variable-photon-energy BM series, including raw `k//` vs `hv` and converted `k//` vs `kz` views.
- Band Analysis tools for TB fits, kink/self-energy analysis, and gap-oriented analysis.
- Optional theory overlay support:
  - local DFT imports from YAML/JSON, VASP `vasprun.xml`, and QE-style `.dat`/`.txt`;
  - optional Materials Project lookup/cache when the relevant API dependency and credentials are available.
- Export of per-slice results, physical summaries, provenance sidecars, CSV/TXT/LaTeX tables, and publication-style figures.
- PyInstaller build recipe and GitHub Actions workflow for tests and tagged release artifacts.

### Experimental or Limited

- Solaris/DA30 support depends on optional `erlab`; it is excluded from bundled PyInstaller binaries.
- ALLS 3D ITX data keeps the third axis as a raw scan coordinate unless later calibrated; geometry confidence is intentionally low.
- Resolution correction quality depends on analyzer metadata. Defaults and manual values should be treated as assumptions, not measured instrument functions.
- Materials Project support is optional and network/API dependent.
- KZ conversion uses a free-electron final-state model and depends on reliable photon energy, work function, lattice `c`, and inner potential `V0`.

### Roadmap

- Document beamline-specific loader assumptions with small public example files.
- Add a dedicated EDC file loader if standalone EDC acquisition formats become part of the workflow.
- Add screenshots and a short tutorial dataset that can be shared publicly.
- Add packaging metadata (`pyproject.toml`) if the project is distributed as an installable Python package.
- Add signed/notarized binaries if the release process needs production-grade desktop distribution.

## Screenshots and Demos

No application screenshots are currently committed.

Suggested placeholders to add:

- `docs/assets/bm-tab.png` — BM tab with raw and processed views.
- `docs/assets/fs-explorer.png` — FS Explorer with a cut line and extracted BM.
- `docs/assets/results-tab.png` — fitted dispersions and physical result table.

Architecture overview:

![ARPES Explorer architecture diagram](docs/arpes_architecture.svg)

## Installation

This repository is currently source-run and PyInstaller-packaged. It does not yet provide a `pyproject.toml`, `setup.py`, Conda environment file, or published package.

### From Source

```bash
git clone https://github.com/mickaelspecht-boop/Arpes_Python.git
cd Arpes_Python

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
```

On Linux, Qt may also need system libraries:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0
```

### Optional Dependencies

- `erlab` is needed for the Solaris/DA30 source loader. It is optional and not bundled in the PyInstaller binaries.
- Materials Project features require the relevant `mp-api` stack and an API key.

## Quick Start

Launch the GUI:

```bash
python arpes_explorer.py
```

Recommended first workflow:

1. Open a measurement folder from the left browser.
2. Fill Sample setup with work function, lattice constants, and cut direction when known.
3. Attach a global or scoped logbook if metadata is not fully stored in the files.
4. Load a BM or FS file.
5. Check EF and Gamma before fitting.
6. Define an MDC fit window, run a single-slice estimate, then run the full fit.
7. Inspect Waterfall, EDC, Results, and Notes before exporting.

Build a local executable:

```bash
python -m pip install pyinstaller
pyinstaller arpes.spec --noconfirm
```

Release builds are produced by GitHub Actions on `v*` tags. See [docs/BUILD_EXECUTABLE.md](docs/BUILD_EXECUTABLE.md).

## Data-Analysis Workflow

```mermaid
flowchart TD
    A[Open data folder] --> B[Load file or FS volume]
    B --> C[Attach logbook and sample metadata]
    C --> D[Calibrate EF and momentum axes]
    D --> E[Inspect BM, FS, or FS Explorer views]
    E --> F[Choose ROI, Gamma, and fit parameters]
    F --> G[Run MDC estimate and full fit]
    G --> H[Review Waterfall, EDC, residuals, and QC tables]
    H --> I[Compute physical tables and optional band analysis]
    I --> J[Export figures, tables, and provenance]
    J --> K[Save session and notes]
```

The key reproducibility rule is simple: calibration choices, sample constants, fit windows, rejected points, and export settings should stay in the session or Notes.

## Architecture

Simplified tree:

```text
.
├── arpes_explorer.py        # GUI entry-point shim
├── arpes_plots.py           # compatibility shim
├── arpes.spec               # PyInstaller recipe
├── requirements.txt         # runtime dependencies
├── arpes/
│   ├── app.py               # PyQt6 main window and controller wiring
│   ├── analysis/            # result aggregation, bootstrap, self-energy
│   ├── core/                # session dataclasses, sample config, undo
│   ├── io/                  # loaders, logbooks, export, cache, KZ datasets
│   ├── physics/             # NumPy/SciPy calculations, no Qt dependency
│   ├── theory/              # DFT import, Materials Project, overlays
│   └── ui/                  # widgets, builders, controllers
├── arpes/docs/              # in-app Help Markdown
├── docs/                    # project documentation and architecture diagram
├── tests/                   # pytest suite
└── tools/                   # maintenance/audit scripts
```

Design constraints used in the codebase:

- keep PyQt imports out of `arpes/physics` and `arpes/io`;
- keep loader outputs validated through the shared `ARPESData` convention;
- keep shared fit-result writes routed through `arpes/core/fit_result_store.py`;
- prefer small controllers with one responsibility.

## Data Formats

Confirmed by the loader registry:

| Source | Format | Notes |
|---|---|---|
| Solaris/DA30 | `.ibw`, `.pxt`, `.zip` | Uses optional `erlab`; not bundled in binaries. |
| BESSY Scienta/SES R8000 | Igor Binary Wave v5 `.ibw` | Requires SES/R8000 metadata. |
| CLS/LNLS | Text BM + `*_param.txt`; FS folders with `*_Cycle_*_Step_*.txt` | Photon energy is required. |
| ALLS SpecsLab Prodigy | Igor Text `.itx` | 2D BM and 3D FS-like volumes; 3D scan geometry may need manual calibration. |
| KZ series | variable-`hv` BM series | Loaded through existing BM loaders and KZ dataset code. |

Unsupported formats should fail explicitly through the loader dispatcher. Support for HDF5, NeXus, generic Scienta exports, or arbitrary CSV is not claimed unless a loader is added and tested.

## Examples

Run tests:

```bash
python -m pytest tests/ \
  --ignore=tests/test_annotations.py \
  --ignore=tests/test_local_dft_loaders.py -q
```

Run Help-panel tests only:

```bash
python -m pytest tests/test_help_panel.py -q
```

Audit real-data metadata locally:

```bash
python tools/audit_real_data_metadata.py /path/to/data
```

No notebooks or public demo datasets are currently committed.

## Tests and Quality

The CI workflow runs on pushes to `main`, pull requests, `v*` tags, and manual dispatch. It installs `requirements.txt`, adds Qt runtime libraries on Linux, sets `QT_QPA_PLATFORM=offscreen`, and runs:

```bash
python -m pytest tests/ \
  --ignore=tests/test_annotations.py \
  --ignore=tests/test_local_dft_loaders.py -q
```

Known local-test notes:

- `tests/test_annotations.py` and `tests/test_local_dft_loaders.py` are intentionally ignored by the standard command.
- UI tests require PyQt6 and a working Qt platform backend. Use `QT_QPA_PLATFORM=offscreen` for headless Linux runs.
- There is no configured formatter, linter, or type checker in the repository yet.

## Known Limits

- ARPES interpretation depends on experimental geometry, metadata quality, EF calibration, work function, lattice constants, and axis conventions.
- Momentum conversion and KZ conversion should not be treated as automatically correct when metadata is incomplete.
- DFT overlays are references for comparison and alignment, not experimental calibration.
- Publication-quality use requires checking raw data, fit residuals, calibration sources, and exported provenance.
- The project is distributed under Apache-2.0. Check third-party dependencies separately when redistributing binaries.

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md).

Short version:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
python -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py -q
```

When contributing scientific features, include tests with synthetic or shareable data. Do not commit private beamtime data, API keys, local absolute paths, or large generated artifacts.

## Citation

This is scientific software. If you use it in analysis that leads to a report, thesis, preprint, or publication, cite the repository and the version or commit used. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Scientific Background

ARPES measures energy- and momentum-resolved photoemission intensity. Common analysis steps include energy referencing to EF, momentum-axis calibration from geometry and lattice constants, MDC/EDC inspection, peak fitting, linewidth analysis, Fermi-surface mapping, and comparison with calculated band structures. ARPES Explorer implements tools for these steps, but physical conclusions still depend on the sample, beamline geometry, calibration quality, and analyst judgment.
