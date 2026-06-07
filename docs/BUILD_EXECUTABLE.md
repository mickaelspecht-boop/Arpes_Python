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

## 3. The `arpes.spec` Recipe File

Create **`code/app/arpes.spec`** (at the `app/` root, next to
`arpes_explorer.py`) with this content:

```python
# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for ARPES Explorer.
# Build: pyinstaller arpes.spec   (from code/app/)
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

hiddenimports = []
# scipy loads submodules dynamically -> bundle all of them
hiddenimports += collect_submodules("scipy")
# matplotlib Qt backend (imported through matplotlib.use("QtAgg"))
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]
# controllers loaded statically in arpes/app.py -> already tracked,
# but secure dynamic loads too (arpes_plots fallback)
hiddenimports += collect_submodules("arpes")

datas = []
# runtime fallback in arpes/app.py:_load_ap() can load the sibling
# arpes_plots.py file by path -> ship it at the root
datas += [("arpes_plots.py", ".")]
# matplotlib data (fonts, mplstyle)
datas += collect_data_files("matplotlib")

a = Analysis(
    ["arpes_explorer.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PySide6", "PySide2"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ARPES_Explorer",
    console=False,            # no black terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,      # macOS: open files through Finder
    target_arch=None,         # build-machine architecture
    icon=None,                # set "icon.ico"/"icon.icns" if available
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ARPES_Explorer",
)

# macOS: also produces a double-clickable .app bundle
app = BUNDLE(
    coll,
    name="ARPES_Explorer.app",
    icon=None,
    bundle_identifier="org.labo.arpesexplorer",
)
```

> Notes:
> - Controllers (`distortion_controller`, `gamma_controller`, ...) are imported
>   **statically** in `arpes/app.py`: PyInstaller follows them automatically.
>   `collect_submodules("arpes")` is an extra safety belt.
> - `~/.config/arpes/distortion_calib.json` and other runtime files stay in the
>   user's HOME: they are **not affected** by packaging, and behavior is
>   identical.
> - There is no `requirements.txt`/`pyproject.toml` in the repo: the `peaks` env
>   is authoritative. Keep the same env between development and build.

---

## 4. Run the Build (On Each OS)

From `code/app/`, with the env activated:

```bash
pyinstaller arpes.spec
```

Output in `code/app/dist/`:

| OS      | Result                                              |
|---------|-----------------------------------------------------|
| Linux   | `dist/ARPES_Explorer/ARPES_Explorer` (+ folder)     |
| Windows | `dist/ARPES_Explorer/ARPES_Explorer.exe` (+ folder) |
| macOS   | `dist/ARPES_Explorer.app`                           |

To deliver: zip **the entire** `dist/ARPES_Explorer/` folder (Linux/Win) or the
`.app` (macOS). Do not extract only the binary.

---

## 5. OS-Specific Notes

### 5.1 Linux

- Build on the **oldest** distribution you target (glibc is forward-compatible,
  not backward-compatible).
- End user: `chmod +x ARPES_Explorer`, then double-click or run
  `./ARPES_Explorer`.
- If there is a Qt `xcb`/plugin error: install `libxcb-cursor0`
  (Debian/Ubuntu) on the target machine, a common Qt6 system dependency.

### 5.2 macOS

- Build per architecture: an `arm64` `.app` (Apple Silicon) and/or `x86_64`
  (Intel). `target_arch="universal2"` is possible if the env has universal
  wheels (rare for scipy, so prefer one build per architecture).
- **Gatekeeper**: without Apple signing/notarization, macOS blocks first launch
  ("damaged app" / "unidentified developer"). Workarounds:
  - Internal lab distribution: right-click the app → **Open** → confirm (once
    per machine). Enough for a few colleagues.
  - Wider distribution: Apple Developer account ($99/year) + `codesign` +
    `notarytool`. Outside this document's scope.
- `argv_emulation=True` (already in the spec) allows opening a file by dragging
  it onto the icon.

### 5.3 Windows

- Build on Windows 10/11. The `.exe` is in the `dist/` folder.
- **SmartScreen**: without code signing, Windows shows "Windows protected your
  PC" on first launch → `More info` → `Run anyway`. Once per machine.
- Optional signing: paid code-signing certificate + `signtool`. Outside scope.
- Antivirus: a fresh PyInstaller `.exe` can sometimes raise a false positive.
  Prefer `onedir` (fewer false positives than `onefile`).

---

## 6. Test Before Distribution (Do Not Break the App)

Checklist, ideally on a **clean** machine (without the Python env):

1. Launch the binary → UI opens, no console crash.
2. Load an ARPES file (FS + BM).
3. Calibrate EF, detect Γ (auto **and** manual click).
4. **Utilities** tab: open/close every section (Grid filter, DFT/Theory, BM
   Distortion).
5. Distortion: trapezoid (sym/antisym/free) + parabola, dotted overlay visible
   while editing then gone after application, effective k// recentering.
6. Grid filter/DFT stay on the BM tab only.
7. CSV export + session save/load.

If something breaks in the binary but works with `python arpes_explorer.py`, it
is a missing import: add it to `hiddenimports` in the spec and rebuild. Iterate.

---

## 7. Optional Automatic Builds Through CI

`.github/workflows/build.yml` — produces the 3 binaries on each tag:

```yaml
name: build-executables
on:
  push:
    tags: ["v*"]
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-22.04, macos-14, windows-2022]
    runs-on: ${{ matrix.os }}
    defaults:
      run:
        working-directory: code/app
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pyinstaller numpy scipy matplotlib PyQt6 h5py pandas
      - name: Build
        run: pyinstaller arpes.spec
      - uses: actions/upload-artifact@v4
        with:
          name: ARPES_Explorer-${{ matrix.os }}
          path: code/app/dist/
```

> Adapt the `pip install` list to the app's real import set if CI does not
> reproduce the `peaks` env. Most reliable: generate a `requirements.txt` from
> the `peaks` env (`pip freeze > requirements.txt`) and install it in CI.

---

## 8. Quick Recap

| Step         | Command |
|--------------|---------|
| Install tool | `pip install pyinstaller` |
| Verify app   | `python arpes_explorer.py` |
| Build        | `pyinstaller arpes.spec` |
| Deliverable  | `dist/ARPES_Explorer/` or `.app` |
| First launch | macOS: right-click→Open / Win: Run anyway |
