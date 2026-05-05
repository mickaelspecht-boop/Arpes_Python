"""Package ARPES Explorer — refonte modulaire.

Architecture (issue de la refonte α→μ) :

    arpes/
      core/         dataclasses session, modèles persistents
      io/           loaders (Solaris/BESSY/CLS), logbook, export
        loaders/    common + un module par backend
      physics/      logique pure (numpy/scipy) :
                      gamma, FS, norm, résolution,
                      géométrie CLS, calibration EF, fit, display
      ui/
        app.py      ArpesExplorer (QMainWindow, orchestration)
        builders/   construction Qt (panels.py, menus.py)
        controllers/ controllers extraits (load, plot, gamma,
                      norm, fs, browser, logbook)
        widgets/
          plots/    fonctions plotting (split par catégorie)

Entry-point CLI : `python3 arpes_explorer.py` (shim racine
re-exportant `arpes.app.main`). Module `arpes.app` est canonique.

Convention loaders : voir `arpes/io/loaders/__init__.py`.
"""
