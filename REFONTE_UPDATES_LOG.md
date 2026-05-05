# Trace refonte ARPES

## Update ζ — finalisation builders UI

- État initial vérifié : commit ζ existant, tests OK, mais API du plan incomplète.
- Correction prévue : ajouter `menus.py`, exposer `build_left_panel`, `build_right_panel`, `build_central_widget`, déplacer les connexions dans `_wire_signals()`.
- Résultat : `arpes_explorer.py` utilise `build_menubar`, `build_left_panel`, `build_right_panel`, `build_central_widget`, puis `_wire_signals()`.
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/builders/panels.py arpes/ui/builders/menus.py` OK ; `python3 -m unittest discover tests` OK (`129` tests, `11` skipped).
