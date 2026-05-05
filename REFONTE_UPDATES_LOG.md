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

## Update θ — `PlotController` absorbe les draws

- État initial vérifié : `arpes_explorer.py` contenait encore `_draw_bm`, `_draw_mdc_energy_map`, `_draw_mdc_waterfall`, `_draw_mdc_edc`, `_draw_kf_overlay`, `_draw_fs_tab`, `_update_display_data`, `_on_scroll_zoom`.
- Correction appliquée : création de `arpes/ui/controllers/plot_controller.py` avec `PlotController`; `ArpesExplorer` instancie `self._plot_ctrl` et conserve seulement des wrappers de compatibilité.
- Résultat : `arpes_explorer.py` passe de `3627` à `3308` lignes ; `plot_controller.py` contient les corps UI de draw (`466` lignes).
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/controllers/plot_controller.py arpes_plot_controller.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.

## Update ι.1 — `GammaController`

- État initial vérifié : logique Γ encore dans `ArpesExplorer` (`_store_fs_center_reference`, `_set_fs_center_pick_mode`, `_detect_fs_gamma`, `_estimate_gamma_bm`, `_apply_gamma_reference_to_bm`, wrappers de référence/azimut).
- Correction appliquée : création de `arpes/ui/controllers/gamma_controller.py`; `ArpesExplorer` instancie `self._gamma_ctrl` et conserve des wrappers de compatibilité.
- Résultat : `arpes_explorer.py` passe de `3308` à `3013` lignes ; `gamma_controller.py` contient les actions UI Γ (`379` lignes).
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/controllers/gamma_controller.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.

## Update ι.2 — `NormController`

- État initial vérifié : logique correction grille/normalisation d'affichage encore dans `ArpesExplorer` (`_load_grid_controls`, `_display_grid_config`, `_grid_status_text`, `_apply_grid_correction`, `_reset_grid_correction`).
- Correction appliquée : création de `arpes/ui/controllers/norm_controller.py`; `ArpesExplorer` instancie `self._norm_ctrl` et conserve des wrappers de compatibilité pour `LoadController`.
- Résultat : `arpes_explorer.py` passe de `3013` à `2964` lignes ; `norm_controller.py` contient les actions UI grille (`81` lignes).
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/controllers/norm_controller.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.
