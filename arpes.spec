# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for ARPES Explorer.
#
# Local build:   pyinstaller arpes.spec          (run from app/)
# CI builds:     .github/workflows/build.yml     (Windows / macOS / Linux)
#
# Full background and troubleshooting: docs/BUILD_EXECUTABLE.md
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
# scipy loads submodules dynamically -> bundle all of them.
hiddenimports += collect_submodules("scipy")
# matplotlib Qt backend (selected at runtime through matplotlib.use("QtAgg")).
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]
# pandas/openpyxl are imported lazily inside the logbook loader.
hiddenimports += ["pandas", "openpyxl"]
# Safety belt for any dynamic import inside the package.
hiddenimports += collect_submodules("arpes")

datas = []
# arpes/app.py:_load_ap() can load the sibling arpes_plots.py by path.
datas += [("arpes_plots.py", ".")]
# In-app Help tab reads the Markdown files from arpes/docs at runtime.
datas += [("arpes/docs", "arpes/docs")]
# matplotlib data (fonts, mplstyle).
datas += collect_data_files("matplotlib")

a = Analysis(
    ["arpes_explorer.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # erlab is an OPTIONAL loader backend (guarded import); bundling it would
    # drag in a huge dependency tree, so it stays out of the binary.
    excludes=["tkinter", "PyQt5", "PySide6", "PySide2", "erlab"],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ARPES_Explorer",
    console=False,                      # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == "darwin"),  # macOS: open files via Finder
    target_arch=None,                   # build-machine architecture
    icon=None,                          # add icon.ico / icon.icns when available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ARPES_Explorer",
)

# macOS: also produce a double-clickable .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ARPES_Explorer.app",
        icon=None,
        bundle_identifier="org.stage.arpesexplorer",
    )
