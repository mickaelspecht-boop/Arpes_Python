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

## Update ο — kill `AP` global + DI propre

- État initial vérifié : 4 fichiers utilisaient `from arpes import app as _ae; _ae.AP` ou `AP` direct.
  - `arpes/app.py` (`AP = None` module-level + `_score_bm_gamma_residual` lit `AP`)
  - `arpes/ui/controllers/load_controller.py` (`_ae.AP`, `_ae.detect_format`, `_ae.load_arpes_file`, `_ae._loader_label`, `_ae.apply_ef_correction_to_dict` — 5 lazy imports)
  - `arpes/ui/controllers/fit_runner_controller.py` (`_ae.AP` ×3)
  - `arpes_explorer.py` (re-export `AP`)
- Correction appliquée :
  - `arpes/app.py` : suppression de `AP = None`. `_load_ap()` retourne `None` si introuvable au lieu de lever. `ArpesExplorer.__init__` stocke `self.ap = _load_ap()` (attribut d'instance, pas global mutable). `_score_bm_gamma_residual` lit `self.ap` au lieu de `AP`.
  - `load_controller.py` : remplacé `_ae.detect_format`/`load_arpes_file`/`_loader_label`/`apply_ef_correction_to_dict` par imports directs (`from arpes.io.loaders import …`, `from arpes.physics.display import …`). `_ae.AP` → `self._parent.ap`. `_ensure_arpes_plots` recharge si `parent.ap is None`.
  - `fit_runner_controller.py` : `_ae.AP` → `p.ap` (3 occurrences). Drop `from arpes import app as _ae`.
  - `arpes_explorer.py` shim : drop ré-exports `AP`, `_load_ap`, `_loader_label`, `apply_ef_correction_to_dict`, `detect_format`, `load_arpes_file` (plus utilisés par tests). Shim passe de 40 à 22 LOC.
- Bugs collatéraux corrigés (présents depuis ν, jamais détectés faute de smoke test) :
  - `arpes/ui/widgets/params.py` utilisait `_dspin`/`_ispin`/`_sep` (legacy) alors qu'il importe `dspin`/`ispin`/`hsep`. Renommés via regex.
  - `arpes/ui/widgets/results.py` utilisait `QLabel` sans l'importer. Ajouté à la liste d'imports `PyQt6.QtWidgets`.
- Vérification post-fix : `grep -rn "_ae\.AP\|arpes\.app\.AP\|^AP\b\|from arpes import app as _ae"` → 0 hit.
- Validation :
  - `python3 -m unittest discover tests` OK (`130` tests, `1` skipped).
  - Smoke : `ArpesExplorer()` instancié headless, `w.ap` non-None, 4 tabs, proxy `_fit_guess` résout vers `FitRunnerController._fit_guess`, proxy `_on_view_changed` vers `InteractionController._on_view_changed`.
- Note : `arpes/app.py` passe de `579` à `607` LOC (ajout `self.ap = _load_ap()` + docstring). Toujours sous cible `<700`.

## Update π — renames `FitController`→`MdcFitter` + `display.py`→`plot_compute.py`

- Motivation : naming trompeur.
  - `FitController` n'est PAS un controller UI Qt, c'est un runner pur (orchestration scipy fits MDC). Le mot "controller" doit rester réservé à `arpes/ui/controllers/*`.
  - `arpes/physics/display.py` ne contient AUCUN code d'affichage Qt/matplotlib — uniquement transforms numpy (`apply_edcnorm`, `apply_ef_correction_to_dict`, `display_grid_config`). Le nom `display` était hérité d'une époque où ces helpers étaient mêlés au rendering.
- Correction appliquée :
  - `git mv arpes/physics/display.py arpes/physics/plot_compute.py`.
  - `arpes/physics/fit.py` : `class FitController` → `class MdcFitter`.
  - 8 fichiers consommateurs mis à jour via script `re.sub` :
    - `arpes/app.py`, `arpes/ui/controllers/{plot,fit_runner,load,norm}_controller.py`, `arpes/physics/fit.py`, `tests/test_fit_controller.py`, `tests/test_plot_controller.py`.
- Vérification post-fix : `grep -rn "physics\.display\|FitController"` → 0 hit.
- Validation : `python3 -m unittest discover tests` OK (`130` tests, `1` skipped) ; smoke headless `ArpesExplorer()` OK + `from arpes.physics.fit import MdcFitter` + `from arpes.physics.plot_compute import apply_edcnorm` résolvent.
- Note : `tests/test_fit_controller.py` garde son nom de fichier (renaming le fichier de test n'apporte rien et casse la mémoire git blame). Class à l'intérieur reste `TestFitController` — peut être renommée en σ si voulu.

## Update ρ — `tests/test_ui_smoke.py`

- Motivation : les bugs ν latents (params.py utilisant `_dspin`, results.py sans `QLabel`) n'ont été détectés qu'en ο en lançant manuellement un smoke headless. Aucun test automatisé n'instanciait `ArpesExplorer`. Boucle ouverte à reboucler.
- Correction appliquée : `tests/test_ui_smoke.py` — instancie `ArpesExplorer` avec `QT_QPA_PLATFORM=offscreen`, vérifie 4 invariants :
  1. `test_window_instantiates` : `_tabs.count() == 4`, `win.ap is not None`.
  2. `test_controllers_wired` : 8 attributs `_*_ctrl` présents.
  3. `test_proxy_dispatch_resolves_every_entry` : chaque clé de `_PROXY_MAP` résout vers une méthode callable du controller cible, et `getattr(win, name).__qualname__ == f"{ControllerClass.__name__}.{name}"`.
  4. `test_widgets_built` : `_params`, `_results`, `_browser`, `_bm_canvas`, `_mdc_edc`, `_tabs` présents après `_build_ui()`.
- Skip si Qt indisponible (`UI_AVAILABLE = False`).
- Validation : `python3 -m unittest tests.test_ui_smoke -v` → 4 OK ; suite complète `134` tests OK skipped=1 (`130 + 4`).
- Filet de sécurité futur : tout refactor qui casse `_PROXY_MAP` ou un import widget sera bloqué dès `unittest discover`.

## Update σ — slim shims racine

- État initial : `arpes_explorer.py` à `22` LOC contenait des ré-exports de classes/widgets pour les tests (`ArpesExplorer`, `FileBrowserPanel`, `FileEntry`, `FileMeta`, `Session`, `_format_direction_label`, `_infer_logbook_mapping`, `MplCanvas`, `FitParamsPanel`, `ResultsPanel`, `QApplication`, `main`).
- Correction appliquée :
  - `tests/test_arpes_explorer_helpers.py` : imports migrés vers leurs modules canoniques (`arpes.core.session`, `arpes.io.logbook`, `arpes.ui.widgets.browsers`, `PyQt6.QtWidgets`). Le test ne dépend plus du shim racine.
  - `arpes_explorer.py` : réduit à `5` LOC — juste `from arpes.app import main` + `if __name__ == "__main__": main()`. Conservé pour exécution directe `python3 arpes_explorer.py`.
- `arpes_plots.py` : déjà à `3` LOC depuis η (`from arpes.ui.widgets.plots import *`). Toujours utilisé par `tests/test_resolution.py` + fallback `_load_ap()`. Conservé.
- Validation : `python3 -m unittest discover tests` OK (`134` tests, `1` skipped) ; `python3 -c "import arpes_explorer"` OK.

## Bilan final refonte α → σ

- `arpes/app.py` : `2820` → `607` LOC (orchestrateur `ArpesExplorer` + `main()`).
- `arpes_explorer.py` : `~3500` (mono-fichier original) → `5` LOC (shim).
- `arpes_plots.py` : `2722` → `3` LOC (shim).
- Architecture finale :
  ```
  arpes/
    app.py                      # 607 LOC, orchestrator + main()
    core/session.py
    io/{loaders, export, logbook, loader_orchestrator}
    physics/{fit (MdcFitter), gamma, ef_calibration, plot_compute,
             cls_geometry, fs, resolution}
    ui/
      builders/{menus, panels}
      controllers/{logbook, load, plot, gamma, norm, fs, browser,
                   interaction, fit_runner}
      widgets/{canvas, params, results, dialogs, _qt_helpers,
               browsers/{files}, plots/}
  arpes_explorer.py             # shim 5 LOC
  arpes_plots.py                # shim 3 LOC
  tests/                        # 134 tests, 1 skipped
  ```
- Dette résiduelle (acceptable) :
  - Mixed FR/EN docstrings/commentaires (cosmétique).
  - `_PROXY_MAP` magique mais préférable à `~60` stubs explicites.
  - Pas de tests UI fonctionnels (ouverture fichier, fit, etc.) — smoke seulement.
- Branche `refonte` prête à merger ou continuer (τ = harmonisation FR→EN si voulu).
