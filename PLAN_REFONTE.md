# Plan refonte ARPES — α → μ

Branche : `refonte` (à créer avant α).

État actuel (post-codex) :
- `arpes_explorer.py` 4136 LOC, 166 méthodes (God class)
- `arpes_plots.py` 2722 LOC (God draw module)
- `arpes_io.py` 1062 LOC (loaders mélangés)
- 14 autres modules `arpes_*.py` à plat à la racine

Cible :
```
Interface/
  arpes_explorer.py            ← shim 5 lignes (entry point)
  arpes/
    __init__.py
    app.py                     ← ArpesExplorer mince (~400 LOC max)
    core/
      models.py                ← arpes_models.py
      session.py               ← arpes_session.py
    io/
      __init__.py
      loader_orchestrator.py   ← arpes_loader_orchestrator.py
      loaders/
        __init__.py
        common.py              ← helpers communs
        solaris.py
        bessy.py
        cls.py
        nexus.py               (si présent)
      logbook.py
      logbook_io.py
      export.py
    physics/
      gamma.py
      ef.py                    ← logique pure de arpes_ef_controller
      fs.py
      norm.py
      resolution.py
      cls_geometry.py
    ui/
      builders/
        menus.py
        panels.py
      widgets/
        plots/
          common.py
          band_map.py
          fermi_surface.py
          mdc_edc.py
          fit_overlay.py
        browsers/
          file_tree.py
          logbook_table.py
      controllers/
        ef_controller.py       ← partie UI (QMessageBox, blockSignals)
        fit_controller.py
        plot_controller.py
        load_controller.py
        logbook_controller.py
        gamma_controller.py
        norm_controller.py
        fs_controller.py
        browser_controller.py
  tests/
    (idem, mais imports refactorés)
```

**Règle clé pour codex** :
1. Avant chaque update : `git checkout refonte && git status` (clean).
2. Après chaque update : `python3 -m unittest discover tests` doit passer.
3. Après chaque update : `python3 arpes_explorer.py` doit lancer l'app sans crash (test manuel utilisateur entre updates).
4. Commit dédié par update : `git commit -m "refonte α : <titre>"`.
5. **Pas de fusion d'updates** — un patch ratée doit pouvoir être `git revert` proprement.

---

## Update α — Initialisation package `arpes/`

**Objectif** : créer la structure de dossiers vide + `__init__.py` partout, sans déplacer de code. Vérifier que tests/app inchangés.

**Étapes codex** :
1. `git checkout -b refonte` (depuis main).
2. Créer arborescence :
   ```
   mkdir -p arpes/core arpes/io/loaders arpes/physics
   mkdir -p arpes/ui/builders arpes/ui/widgets/plots arpes/ui/widgets/browsers arpes/ui/controllers
   ```
3. Créer `__init__.py` vide dans chacun de ces dossiers + `arpes/__init__.py`.
4. Aucun changement de code Python ailleurs.

**Validation** :
- `python3 -m unittest discover tests` → 129 OK skipped=11.
- `python3 arpes_explorer.py` → app démarre.

**Commit** : `refonte α : squelette package arpes/`.

---

## Update β — Migration physics + core (modules pure-Python)

**Objectif** : déplacer les modules sans dépendance UI vers le package. Simple `git mv` + ajustement imports.

**Modules à migrer** :
| Source | Destination |
|---|---|
| `arpes_models.py` | `arpes/core/models.py` |
| `arpes_session.py` | `arpes/core/session.py` |
| `arpes_gamma.py` | `arpes/physics/gamma.py` |
| `arpes_fs.py` | `arpes/physics/fs.py` |
| `arpes_norm.py` | `arpes/physics/norm.py` |
| `arpes_resolution.py` | `arpes/physics/resolution.py` |
| `arpes_cls_geometry.py` | `arpes/physics/cls_geometry.py` |
| `arpes_export.py` | `arpes/io/export.py` |
| `arpes_logbook.py` | `arpes/io/logbook.py` |
| `arpes_logbook_io.py` | `arpes/io/logbook_io.py` |
| `arpes_loader_orchestrator.py` | `arpes/io/loader_orchestrator.py` |
| `arpes_io.py` | `arpes/io/loaders/__init__.py` (temporaire — sera splitté en γ) |

**Étapes codex** :
1. Pour chaque ligne du tableau : `git mv <source> <destination>`.
2. Mettre à jour TOUS les imports (`arpes_models` → `from arpes.core.models import …` etc.) dans :
   - `arpes_explorer.py`
   - tous les fichiers `tests/test_*.py`
   - tous les nouveaux fichiers déplacés (imports croisés entre eux)
3. Ré-exporter pour compat tests : créer `arpes/io/loaders/__init__.py` qui contient l'ex-contenu de `arpes_io.py` tel quel.

**Validation** :
- `grep -rn "^from arpes_\|^import arpes_" --include='*.py' .` → doit retourner uniquement `from arpes_explorer` (entry).
- Tests OK.
- App démarre.

**Commit** : `refonte β : migration physics+io+core dans arpes/`.

---

## Update γ — Split `arpes_io.py` par backend

**Objectif** : `loaders/__init__.py` (1062 LOC) éclaté par source (Solaris, BESSY, CLS, etc.). Chaque loader < 300 LOC.

**Préparation codex (à exécuter avant patch)** :
```
grep -n "^def \|^class " arpes/io/loaders/__init__.py
```
Identifier les fonctions par préfixe / docstring (`load_solaris*`, `load_bessy*`, `load_cls*`, `load_nexus*`, helpers communs).

**Découpage cible** :
| Fichier | Contenu |
|---|---|
| `arpes/io/loaders/common.py` | helpers : parse_header, axis_from_step, normalisation paths, unités, fonctions partagées |
| `arpes/io/loaders/solaris.py` | tout `load_solaris_*`, `_solaris_*` |
| `arpes/io/loaders/bessy.py` | `load_bessy_*` / SES |
| `arpes/io/loaders/cls.py` | `load_cls_*` (texte + param sidecar) |
| `arpes/io/loaders/nexus.py` | si fichiers `.nxs` (sinon supprimer) |
| `arpes/io/loaders/__init__.py` | ré-exporte tout : `from .solaris import *` etc. + dispatcher principal |

**Étapes codex** :
1. Pour chaque famille, créer le nouveau fichier, copier les fonctions, ajouter leurs imports nécessaires.
2. Vider l'ancien `__init__.py`, le remplacer par les ré-exports.
3. Ajouter test smoke `tests/test_loaders_split.py` : importer chaque sous-module, vérifier qu'au moins une fonction publique existe.

**Validation** :
- Tests OK (en particulier `test_loaders_integration.py`).
- App charge un fichier Solaris + un fichier CLS sans erreur.

**Commit** : `refonte γ : split loaders par backend`.

---

## Update δ — Extraction `LogbookIngestController`

**Objectif** : sortir `_read_logbook` (arpes_explorer.py:2909) de la God class.

**Étapes codex** :
1. Lire `_read_logbook` + ses helpers (`_choose_excel_sheet`, `_choose_excel_table`, etc. — `grep -n "_read_logbook\|_choose_excel\|_logbook_" arpes_explorer.py`).
2. Créer `arpes/ui/controllers/logbook_controller.py` :
   ```python
   from PyQt6.QtWidgets import QInputDialog, QMessageBox
   from arpes.io.logbook_io import read_excel_workbook, ...
   
   class LogbookIngestController:
       def __init__(self, parent_widget):
           self._parent = parent_widget
       
       def read(self, path: Path) -> tuple[list[dict], dict[str, str], str]:
           # corps copié depuis _read_logbook
           ...
   ```
3. Dans `arpes_explorer.py`, remplacer le corps de `_read_logbook` par :
   ```python
   def _read_logbook(self, path):
       return self._logbook_ingest.read(path)
   ```
   et instancier `self._logbook_ingest = LogbookIngestController(self)` dans `__init__`.
4. Test : `tests/test_logbook_controller.py` avec mocks `QInputDialog` (cf. `unittest.mock.patch`).

**Validation** :
- Tests OK.
- Import logbook Excel fonctionne depuis l'app.

**Commit** : `refonte δ : LogbookIngestController`.

---

## Update ε — Split `_load_file` → `LoadController`

**Objectif** : casser `_load_file` (arpes_explorer.py:3059, ~169 LOC) en pipeline lisible.

**Découpage proposé** :
1. `prepare_path(path)` — résolution, vérif existence
2. `dispatch_loader(path, meta)` — choix backend
3. `apply_post_load_corrections(raw, entry)` — gamma + EF + norm
4. `update_session(entry, raw)`
5. `refresh_ui(entry)` — appelle PlotController

**Étapes codex** :
1. Créer `arpes/ui/controllers/load_controller.py` avec classe `LoadController` exposant :
   ```python
   def load(self, path: str) -> None:
       path_obj = self._prepare_path(path)
       raw, entry = self._dispatch(path_obj)
       self._apply_corrections(raw, entry)
       self._update_session(entry, raw)
       self._refresh_ui(entry)
   ```
2. Chaque méthode privée < 30 LOC. Code copié-collé depuis `_load_file` puis splitté logiquement (chercher commentaires existants comme repères).
3. Dans `arpes_explorer.py`, `_load_file` devient :
   ```python
   def _load_file(self, path):
       self._load_ctrl.load(path)
   ```
4. Tests : `tests/test_load_controller.py` avec un loader fake pour valider le pipeline.

**Validation** :
- Tests OK.
- Charger BM Solaris : OK.
- Charger FS BESSY : OK.
- Charger CLS texte : OK.
- Recharger après changement EF offset : OK.

**Commit** : `refonte ε : LoadController + pipeline split`.

---

## Update ζ — Split `_build_ui` + extraction builders

**Objectif** : `_build_ui` (arpes_explorer.py:2108, ~174 LOC) → builders dédiés.

**Découpage** :
| Fichier | Rôle |
|---|---|
| `arpes/ui/builders/menus.py` | `build_menubar(window) → QMenuBar` |
| `arpes/ui/builders/panels.py` | `build_left_panel`, `build_right_panel`, `build_central_widget` |

**Étapes codex** :
1. Lire `_build_ui` ligne par ligne, identifier blocs (menubar, browser, params, plots).
2. Pour chaque bloc, créer fonction libre dans le builder approprié, paramétrée par le `parent` (le `ArpesExplorer`).
3. Le builder retourne le widget composé + dict des sous-widgets exposés (`{"sp_ev": ..., "sp_ef": ..., ...}`).
4. `_build_ui` devient :
   ```python
   def _build_ui(self):
       self.setMenuBar(build_menubar(self))
       self._left = build_left_panel(self)
       self._right = build_right_panel(self)
       central = build_central_widget(self, self._left, self._right)
       self.setCentralWidget(central)
       self._wire_signals()
   ```
5. Créer `_wire_signals()` séparée pour tout `connect()`.
6. Tests : pas de test unitaire UI, validation visuelle.

**Validation** :
- App démarre, tous menus présents, tous boutons cliquables.
- Aucune régression visuelle.

**Commit** : `refonte ζ : builders UI + _build_ui mince`.

---

## Update η — Split `arpes_plots.py` (2722 LOC) par type de plot

**Objectif** : éclater le module dessin par responsabilité.

**Préparation codex** :
```
grep -n "^def \|^class " arpes/ui/widgets/plots/__init__.py
```
(le fichier sera déplacé d'abord par β-bis, voir étape 1).

**Étape 0** : `git mv arpes_plots.py arpes/ui/widgets/plots/__init__.py`, ajuster imports dans `arpes_explorer.py` et `arpes_plot_controller.py`.

**Découpage** :
| Fichier | Contenu |
|---|---|
| `arpes/ui/widgets/plots/common.py` | bases `MplCanvas`, helpers `apply_extent`, `default_cmap`, colorbar, ticks, hooks zoom/pan |
| `arpes/ui/widgets/plots/band_map.py` | `BandMapCanvas`, `draw_bm`, MDC/EDC overlays sur BM |
| `arpes/ui/widgets/plots/fermi_surface.py` | `FermiSurfaceCanvas`, projections, contours |
| `arpes/ui/widgets/plots/mdc_edc.py` | dialogs/widgets MDC + EDC standalone, fits overlays |
| `arpes/ui/widgets/plots/fit_overlay.py` | tout overlay de fit (gauss/lorentz/EDC fits) |
| `arpes/ui/widgets/plots/__init__.py` | ré-export public |

**Étapes codex** :
1. Identifier les classes/fonctions par regex sur préfixes (BM/FS/MDC/EDC/FIT) ou par nom de classe.
2. Déplacer un type à la fois, recompiler entre chaque (`python3 -c "import arpes.ui.widgets.plots"`).
3. Supprimer toute fonction morte (`grep -rn "<nom>" .` → 0 hit hors définition = mort).

**Validation** :
- Tests OK.
- App : afficher BM, FS, MDC, EDC, fit gauss → tous OK.

**Commit** : `refonte η : split plots par type`.

---

## Update θ — `PlotController` mince + retirer `_draw_*` du main

**Objectif** : `arpes_plot_controller.py` (491) doit absorber les `_draw_*`, `_update_display_data`, `_on_scroll_zoom` restants dans `arpes_explorer.py`.

**Étapes codex** :
1. `grep -n "_draw_\|_update_display_data\|_on_scroll_zoom" arpes_explorer.py`.
2. Pour chaque méthode trouvée : déplacer dans `arpes/ui/controllers/plot_controller.py` (déplacer aussi via β/η si pas encore fait).
3. Remplacer dans `arpes_explorer.py` par délégation `self._plot_ctrl.<name>(...)`.
4. Si possible, supprimer la délégation et appeler directement `self._plot_ctrl.draw_bm(...)` depuis les call sites.

**Validation** :
- Tests OK.
- Plot mis à jour quand on bouge sp_ev / sp_ef : OK.
- Zoom molette : OK.

**Commit** : `refonte θ : PlotController absorbe draws`.

---

## Update ι — Controllers feature-par-feature

**Objectif** : chaque feature majeure a son controller dédié.

**Controllers à créer** :
| Controller | Responsabilité | Méthodes source à migrer |
|---|---|---|
| `gamma_controller.py` | UI gamma (boutons, dialogs) | `_estimate_gamma_bm`, `_apply_gamma_*`, `_store_fs_center_reference` |
| `norm_controller.py` | UI normalisation | `_apply_norm_*`, `_open_norm_dialog`, `_remove_grid_*` |
| `fs_controller.py` | UI Fermi Surface | `_compute_fs`, `_show_fs_dialog`, projections |
| `browser_controller.py` | tree + selection | `_on_tree_selection`, `_refresh_browser`, `_filter_browser` |

**Étapes codex (par controller, donc 4 sous-commits)** :
1. Créer le fichier dans `arpes/ui/controllers/`.
2. Déplacer méthodes + state local (`self._gamma_*` → `self._gamma_ctrl._*`).
3. `arpes_explorer.py` : instancier dans `__init__`, déléguer.
4. Ajouter test si la logique pure peut être isolée.

**Validation après chaque sous-commit** :
- Tests OK.
- Feature concernée fonctionne dans l'app.

**Commits** :
- `refonte ι.1 : GammaController`
- `refonte ι.2 : NormController`
- `refonte ι.3 : FSController`
- `refonte ι.4 : BrowserController`

---

## Update κ — `ArpesExplorer` → `MainWindow` mince

**Objectif** : ce qui reste dans `arpes_explorer.py` (devrait être < 800 LOC après ι) → `arpes/app.py` comme classe `ArpesExplorer(QMainWindow)` qui ne fait que :
- composer les controllers
- appeler les builders UI
- gérer les hot-keys
- save/restore session sur close

**Étapes codex** :
1. `git mv arpes_explorer.py arpes/app.py`.
2. Couper la classe en sections commentées : `# --- composition ---`, `# --- session lifecycle ---`, `# --- hotkeys ---`.
3. Toute méthode > 30 LOC doit être réexaminée : peut-elle être encore déléguée à un controller existant ?
4. Recréer `arpes_explorer.py` à la racine en shim :
   ```python
   from arpes.app import main
   if __name__ == "__main__":
       main()
   ```
5. Ajouter `main()` dans `arpes/app.py` qui crée `QApplication` + `ArpesExplorer().show()` + `sys.exit(app.exec())`.

**Validation** :
- `python3 arpes_explorer.py` → démarre comme avant.
- `arpes/app.py` < 600 LOC.

**Commit** : `refonte κ : shim entry-point + MainWindow dans arpes/app.py`.

---

## Update λ — Pure logic vs UI controllers split

**Objectif** : pour chaque controller UI, vérifier si la logique métier dedans peut être extraite vers `arpes/physics/` ou `arpes/io/` (pure, testable sans Qt).

**Pattern** : un controller doit ressembler à
```python
class XController:
    def __init__(self, parent_widget, ...):
        self._parent = parent_widget
    def do_action(self, *args):
        try:
            result = pure_compute(args)  # vient de arpes/physics ou arpes/io
        except ValidationError as e:
            QMessageBox.warning(self._parent, "...", str(e))
            return
        self._update_ui(result)
```

**Étapes codex** :
1. Pour chaque `*_controller.py`, identifier les blocs purs (pas de `Q*`, pas de `self._parent`).
2. Les extraire dans le module pure correspondant.
3. Le controller importe la fonction pure et l'appelle.
4. Ajouter tests purs.

**Validation** :
- Tests OK + nouveaux tests pour fonctions extraites.
- App OK.

**Commit** : `refonte λ : pure logic hors controllers`.

---

## Update μ — Nettoyage final

**Objectif** : zéro mort, zéro duplication, conventions cohérentes.

**Étapes codex** :
1. `python3 -m pyflakes arpes/ tests/` → corriger toutes les warnings (imports inutilisés, vars mortes).
2. `grep -rn "TODO\|FIXME\|XXX" arpes/ tests/` → résoudre ou créer issue.
3. `grep -rn "^from arpes_\|^import arpes_" .` → 0 résultat hors `arpes_explorer.py` (shim).
4. Vérifier qu'il ne reste **aucun** `arpes_*.py` à la racine sauf `arpes_explorer.py`.
5. Mettre à jour `ARPES_PROJECT_CONTEXT_FOR_CLAUDE_CODE.txt` avec nouvelle structure.
6. README rapide à la racine décrivant l'arborescence.

**Validation** :
- Tests OK.
- App OK.
- LOC par fichier : tous < 600 idéalement, max 800 toléré.

**Commit** : `refonte μ : nettoyage final + doc`.

---

## Récap LOC cible post-refonte

| Module | LOC max |
|---|---|
| `arpes/app.py` | 600 |
| `arpes/ui/controllers/*.py` | 300 chacun |
| `arpes/ui/widgets/plots/*.py` | 500 chacun |
| `arpes/ui/builders/*.py` | 200 chacun |
| `arpes/io/loaders/*.py` | 300 chacun |
| `arpes/physics/*.py` | 500 chacun |
| `arpes/core/*.py` | 250 chacun |

Aucun fichier > 800 LOC. Aucune classe > 50 méthodes.

---

## Comment utiliser ce plan avec codex

Pour chaque update, donne à codex **une seule** des sections ci-dessus (ex : "applique l'Update γ"), avec instruction :
> Lis `PLAN_REFONTE.md`, applique uniquement l'Update <lettre>, valide tests + app, fais le commit indiqué. N'enchaîne pas sur l'update suivante.

Entre deux updates, lance toi-même `python3 arpes_explorer.py` et fais un test manuel : charger un fichier, regarder un plot, changer un offset. Si ça plante, dis-le moi avant l'update suivante.
