"""ARPES Explorer package.

Main package layout:

    arpes/
      core/         session dataclasses, persistent models
      io/           loaders (Solaris/BESSY/CLS), logbook, export
        loaders/    common + one module per backend
      physics/      pure logic (numpy/scipy):
                      gamma, FS, norm, resolution,
                      CLS geometry, EF calibration, fit, display
      ui/
        app.py      ArpesExplorer (QMainWindow, orchestration)
        builders/   Qt construction (panels.py, menus.py)
        controllers/ UI controllers (load, plot, gamma,
                      norm, fs, browser, logbook)
        widgets/
          plots/    plotting functions (split by category)

CLI entry point: `python3 arpes_explorer.py`.

Loader convention: see `arpes/io/loaders/__init__.py`.
"""
