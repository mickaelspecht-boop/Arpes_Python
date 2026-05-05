# Trace refonte ARPES

## Update ζ — finalisation builders UI

- État initial vérifié : commit ζ existant, tests OK, mais API du plan incomplète.
- Correction prévue : ajouter `menus.py`, exposer `build_left_panel`, `build_right_panel`, `build_central_widget`, déplacer les connexions dans `_wire_signals()`.
- Résultat : `arpes_explorer.py` utilise `build_menubar`, `build_left_panel`, `build_right_panel`, `build_central_widget`, puis `_wire_signals()`.
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/builders/panels.py arpes/ui/builders/menus.py` OK ; `python3 -m unittest discover tests` OK (`129` tests, `11` skipped).

## Update η — split `arpes_plots.py`

- État initial vérifié : `arpes_plots.py` faisait `2722` lignes et était chargé via `arpes_explorer._load_ap()`.
- Correction appliquée : `arpes_plots.py` devient un shim de compatibilité ; fonctions déplacées dans `arpes/ui/widgets/plots/` par responsabilité (`common`, `processing`, `band_map`, `fermi_surface`, `mdc_edc`, `mdc_fit`, `mdc_diagnostics`, `mdc_regions`, `edc_fit`, `fit_overlay`).
- Intégration : `arpes_explorer._load_ap()` charge `arpes.ui.widgets.plots` en priorité ; le shim racine reste pour compatibilité.
- Compatibilité : API historique conservée via `from arpes.ui.widgets.plots import *`, y compris helpers privés utilisés par tests (`_resolution_correct_gamma`, etc.).
- Test ajouté : `tests/test_plots_split.py` vérifie imports des sous-modules et réexports du shim.
- Validation : `python3 -m py_compile arpes_plots.py arpes/ui/widgets/plots/*.py tests/test_plots_split.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.
