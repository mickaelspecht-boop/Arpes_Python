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

## Update ι.3 — `FSController`

- État initial vérifié : état FS restant dans `ArpesExplorer`/`PlotController` (`_current_is_fs`, `_on_fs_params_changed`, `_save_current_fs_center`, `_draw_fs_tab`).
- Correction appliquée : création de `arpes/ui/controllers/fs_controller.py`; `ArpesExplorer` instancie `self._fs_ctrl`; `_draw_fs_tab` quitte `PlotController`.
- Résultat : `arpes_explorer.py` passe de `2964` à `2948` lignes ; `fs_controller.py` contient les actions UI FS (`53` lignes).
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/controllers/fs_controller.py arpes/ui/controllers/plot_controller.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.

## Update ι.4 — `BrowserController`

- État initial vérifié : `FileBrowserPanel` contenait encore sélection, navigation, refresh d'item, ouverture/réduction de groupes et repopulation.
- Correction appliquée : création de `arpes/ui/controllers/browser_controller.py`; `FileBrowserPanel` instancie `self._browser_ctrl` et conserve des wrappers de compatibilité.
- Résultat : `arpes_explorer.py` passe de `2948` à `2844` lignes ; `browser_controller.py` contient la logique browser (`147` lignes).
- Validation : `python3 -m py_compile arpes_explorer.py arpes/ui/controllers/browser_controller.py` OK ; `python3 -m unittest discover tests` OK (`130` tests, `12` skipped) ; env `peaks` OK (`130` tests, `5` skipped).
- Validation app : lancement `arpes_explorer.py` en `QT_QPA_PLATFORM=offscreen` avec env `peaks`, processus vivant après `5s` puis terminé.

## Update κ → ν — résumé rapide (déjà committés)

- κ (`747e5f0`) : `arpes_explorer.py` migré dans `arpes/app.py` + shim entry-point racine.
- λ (`3e27fc7`) : extraction logique pure controllers racine → `arpes/physics/` (`gamma.py`, etc.).
- μ (`51f0e41`) : nettoyage pyflakes + docstring package.
- ν (`d673f8c`) : extraction widgets (`canvas`, `browsers/files`, `params`, `results`, `dialogs`) + helpers (`load_arpes_file`, `loader_label`, `apply_ef_correction_to_dict`) hors `arpes/app.py`. `arpes/app.py` 2820 → 1016 LOC.

## Update ξ — `InteractionController` + `FitRunnerController` + proxy dispatch

- État initial vérifié : `arpes/app.py` à `1016` LOC contenait encore toute la logique d'interaction temps-réel (`_on_view_changed`, `_on_ev_spinbox_changed`, ROI fit, `_on_map_click`, debouncers) + tous les fits MDC + calibration EF (`_fit_guess`, `_fit_full`, `_clear_kf`, `_ef_calibrate`, `_apply_ef_calibration_result`, `_apply_ef_reference_to_current`, `_refresh_helper_buttons`, `_copy_params`).
- Correction appliquée :
  - création de `arpes/ui/controllers/interaction_controller.py` (`210` LOC) avec 14 méthodes interaction/scheduling.
  - création de `arpes/ui/controllers/fit_runner_controller.py` (`290` LOC) avec 9 méthodes fit/EF.
  - extension de `_PROXY_MAP` dans `ArpesExplorer` (`+23` entrées) → dispatch via `__getattr__` vers `_interaction_ctrl` / `_fit_runner_ctrl`.
  - réordonnancement `__init__` : controllers instanciés AVANT `QTimer.timeout.connect(self._on_model_changed)` (sinon `__getattr__` lève `AttributeError`).
- Résultat : `arpes/app.py` passe de `1016` à `579` LOC. Aucun `_on_*` / `_apply_*` / `_fit_*` / `_ef_*` / `_clear_*` / `_get_work_*` / `_refresh_helper_*` / `_copy_*` / `_sync_*` / `_schedule_*` / `_set_fit_roi_*` / `_reset_fit_roi_*` ne reste sur `ArpesExplorer` (tous proxiés).
- Note : `FitRunnerController` utilise encore `from arpes import app as _ae; _ae.AP` (lazy global) — sera nettoyé en ο.
- Validation : `python3 -m unittest discover tests` OK (`130` tests, `1` skipped) ; `import arpes_explorer` OK.
- Pas encore committé au moment de l'écriture de cette ligne — commit immédiat dans la foulée.

## Plan restant — à exécuter ensuite (NE PAS enchaîner sans top utilisateur)

Cibles finales architecture : `arpes/app.py` < `600` LOC (uniquement `ArpesExplorer` orchestrateur + `main()`), 0 import circulaire lazy, 0 global mutable, naming clair, tests UI smoke, shims fins.

- **ο — supprimer `AP` global + lazy `from arpes import app as _ae`** :
  - `_load_ap()` ne s'exécute plus comme module-level write sur `arpes.app.AP`. Appelé une fois dans `main()`.
  - `LoadController`, `FitRunnerController`, `GammaController` reçoivent `ap` via constructeur ou via `parent.ap` (attribut d'instance, pas global).
  - Effacer `AP = None` racine + tout `_ae.AP` / `arpes.app.AP` dans le code.
  - Vérifier `_score_bm_gamma_residual` (utilise `AP` directement aussi).

- **π — renames** :
  - `arpes/physics/fit.py::FitController` → `MdcFitter` (naming : ce n'est PAS un controller UI, c'est un runner pur). Update imports : `arpes/app.py`, `arpes/ui/controllers/fit_runner_controller.py`, tests éventuels.
  - `arpes/physics/display.py` → `arpes/physics/plot_compute.py` (le nom `display` est trompeur, ça calcule, ça n'affiche rien). Update tous les `from arpes.physics.display import …`.

- **ρ — `tests/test_ui_smoke.py`** :
  - Crée un `QApplication`, instancie `ArpesExplorer`, vérifie `_PROXY_MAP` résout chaque clé sans `AttributeError`, vérifie `win._tabs.count() >= 4`. `QT_QPA_PLATFORM=offscreen`. Skip si Qt indispo.

- **σ — slim shims** :
  - `arpes_explorer.py` racine → 5 LOC max (re-export `from arpes.app import *` + `if __name__ == "__main__": main()`).
  - Évaluer `arpes_plots.py` racine : si plus aucun caller externe, retirer ; sinon garder shim 5 LOC.

### Précautions reprises pour codex

1. Vérifier toujours après chaque step : `python3 -m unittest discover tests` doit rester `130 OK skipped=1` (env `peaks`, `/Users/alexandrespecht/.local/bin/micromamba run -n peaks`).
2. `import arpes_explorer` doit rester OK (smoke).
3. Branche `refonte`. Commits suivent format `refonte <lettre> : <description>` (cf `git log`).
4. Ordre `__init__` : controllers AVANT debouncers (timer connect résout via `__getattr__`).
5. Mock test patterns : si on déplace une fonction utilisée par un test via `mock.patch.object`, ajouter le mock sur le NOUVEAU module aussi (cf `tests/test_arpes_explorer_helpers.py` qui patche à la fois `arpes_app` et `_files_mod`).
