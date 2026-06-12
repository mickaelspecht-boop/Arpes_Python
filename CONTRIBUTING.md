# Contributing

Thanks for considering a contribution. ARPES Explorer is scientific software, so changes should be easy to test and honest about their physical assumptions.

By contributing to this repository, you agree that your contribution is provided under the Apache License 2.0 unless explicitly agreed otherwise in writing.

## Development Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
```

Launch the application:

```bash
python arpes_explorer.py
```

Run the standard test suite:

```bash
python -m pytest tests/ \
  --ignore=tests/test_annotations.py \
  --ignore=tests/test_local_dft_loaders.py -q
```

For headless Linux UI tests:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/test_ui_smoke.py -q
```

## Scope of Contributions

Useful contributions include:

- loader support for a new beamline or export format;
- tests for existing loaders and analysis routines;
- reproducible examples with shareable synthetic or public data;
- improvements to EF, Gamma, FS, KZ, or MDC workflows;
- documentation that clarifies assumptions and limitations.

## Scientific and Data Policy

- Do not commit private beamtime data, unpublished datasets, access tokens, API keys, or local absolute paths.
- Prefer synthetic fixtures or small public files with clear provenance.
- Document calibration assumptions: photon energy, work function, lattice constants, angle convention, EF offset, and resolution source.
- Do not describe a feature as supported until code and tests exist.

## Code Guidelines

- Keep PyQt-specific code in `arpes/ui` or UI controllers.
- Keep numerical logic in `arpes/physics`, `arpes/analysis`, or `arpes/theory` where it can be tested without a GUI.
- Keep loaders in `arpes/io/loaders` and return validated `ARPESData`.
- Add focused tests for behavior changes.
- Avoid broad refactors in the same change as a scientific or loader fix.

## Pull Requests

Before opening a PR:

1. Run the standard tests or explain why they could not be run.
2. Update `README.md`, `docs/index.md`, or `arpes/docs/*.md` if user-facing behavior changed.
3. Mention any scientific assumptions or calibration limits.
4. Keep generated caches, binary artifacts, and private data out of the diff.

## Releases

Tagged releases use GitHub Actions:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The workflow runs tests, builds PyInstaller artifacts for Linux x86_64, Windows x86_64, and macOS arm64, then attaches them to a GitHub Release.
