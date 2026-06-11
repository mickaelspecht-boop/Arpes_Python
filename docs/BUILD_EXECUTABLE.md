# Build ARPES Explorer as an Executable (Linux / macOS / Windows)

Goal: ship a double-clickable binary. Users no longer need to install
`micromamba`, the `peaks` env, or open a terminal.

> **No application-code change is required.** This document only adds a
> packaging recipe file (`arpes.spec`) and commands. The app remains runnable as
> before with `python arpes_explorer.py`.

---

## 1. Principle & Constraints

- Tool: **PyInstaller**. It freezes Python + PyQt6 + numpy/scipy/matplotlib +
  the `arpes/` code into a standalone deliverable.
- **No cross-compilation.** A build only runs on the OS and architecture that
  produced it:
  - Windows → build on Windows → `.exe`
  - macOS → build on macOS → `.app` (Intel `x86_64` ≠ Apple Silicon `arm64`:
    one build per architecture, or one Mac of each type)
  - Linux → build on Linux → ELF binary (compatible if the build machine's
    glibc is ≤ the target machines' glibc; build on the oldest OS you target)
- Expected size: ~150-350 MB (scipy/numpy/Qt are large). Normal.
- `onedir` mode (one folder) is recommended first: faster startup and simpler
  debugging. `onefile` (single file) is possible afterward.

---

## 2. Prepare the Build Environment (All 3 OSes)

On **each** build machine, in the project env:

```bash
# env identical to development (Python 3.12, PyQt6 6.10, numpy 2.3,
# scipy 1.17, matplotlib 3.10)
micromamba activate peaks         # or equivalent conda/venv
pip install pyinstaller
```

Verify that the app starts from this env **before** packaging:

```bash
cd code/app
python arpes_explorer.py          # must open the UI without errors
```

If it does not work here, the binary will not work either: fix the env first.

---

## 3. Files in the repo (DONE — 2026-06-11)

Everything below is now **implemented in the repo**; this document is the
reference for how it works.

| File | Role |
|---|---|
| `arpes.spec` | PyInstaller recipe (hiddenimports scipy/pandas, datas arpes/docs + arpes_plots.py, excludes erlab/tkinter/PyQt5) |
| `requirements.txt` | Runtime deps pinned to the validated majors |
| `.github/workflows/build.yml` | CI: tests on every push; binaries (4 targets) on every `v*` tag, attached to a GitHub Release |

## 4. Local build (any OS)

```bash
micromamba activate peaks      # or any env satisfying requirements.txt
pip install pyinstaller
pyinstaller arpes.spec --noconfirm
# result: dist/ARPES_Explorer/   (and dist/ARPES_Explorer.app on macOS)
```

## 5. Release / patch workflow (the normal path)

No build machine needed — GitHub Actions builds all three targets
(Linux x86_64, Windows x86_64, macOS arm64). No Intel macOS binary:
GitHub retired the free macos-13 runners; Intel Macs run from source.

```bash
# 1. fix/commit as usual
git push
# 2. when ready to ship:
git tag v1.0.1
git -c url."ssh://git@ssh.github.com:443/".insteadOf="git@github.com:" push origin v1.0.1
```

→ The `build` workflow runs the test suite, builds the 4 binaries, and
publishes them on the repo's **Releases** page. Users download the archive
for their OS, unzip, double-click `ARPES_Explorer`.

Notes:
- macOS Gatekeeper: unsigned app → first launch via right-click → Open
  (or `xattr -dr com.apple.quarantine ARPES_Explorer.app`). Signing/notarizing
  needs an Apple Developer account — out of scope for now.
- Linux: built on Ubuntu 22.04 so the binary runs on any distro with
  glibc ≥ 2.35.
- The optional `erlab` loader backend is NOT bundled (guarded import; the
  binary simply reports the format as unsupported).
