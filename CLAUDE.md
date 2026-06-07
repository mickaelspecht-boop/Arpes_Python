# CLAUDE.md — ARPES Explorer

Contexte projet auto-chargé par Claude Code. Toute modification du code doit respecter ces règles.

## Projet

- ARPES PyQt6 explorer (analyse Angle-Resolved Photoemission Spectroscopy).
- Repo principal: `arpes/` (package). Shims racine `arpes_explorer.py` + `arpes_plots.py` ≤5 LOC, ne pas étendre.
- Branche active: `main`. Mono-dev, pas de PR.
- Lancer tests: `python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py -q`
- État courant: 838 tests OK / 9 skip (env micromamba `peaks`, PyQt6 présent). Sans PyQt6: ~745 OK + ~61 skip Qt.

## Index docs (où vit quoi)

Seul ce fichier est auto-chargé. Lire les autres **à la demande**.

| Fichier | Rôle | Quand le lire |
|---|---|---|
| `CLAUDE.md` (ici) | Règles durables + architecture | Toujours (auto) |
| `docs/ai/BACKLOG.md` | Quoi faire ensuite (1 seule liste priorisée) | Avant de choisir/commencer un travail |
| `docs/ai/DECISIONS.md` | Journal append-only quoi+pourquoi | Pour comprendre un choix passé |
| `docs/ai/COUNCIL.md` | Template conseil + quand spawn quel agent | Avant feature non-triviale |
| `.claude/agents/*.md` | 9 personas conseil | Chargés dans le prompt du sous-agent |
| `docs/ai/archive/` | Vieux audits/plans (froid) | Référence ponctuelle, jamais auto |
| `arpes/docs/*.md` | Help in-app (EN, chargé au runtime) | Si on touche le contenu d'aide |
| `docs/BUILD_EXECUTABLE.md` | Recette packaging PyInstaller | Pour builder un binaire |

**Règle anti-prolifération** : ne **jamais** créer un `*_PLAN.md`/`*_TODO.txt` à
la racine. Tout travail → section dans `BACKLOG.md`. Gros design → un seul fichier
`docs/ai/plans/<slug>.md` lié depuis le backlog, supprimé/archivé une fois fait.
Langue : app en anglais, docs AI-facing + commits-context en français.

## Architecture (NON négociable)

```
arpes/
  core/                dataclasses + session JSON + fit_result_store
    session.py         FileEntry + Session (load/save .arpes_session.json)
    fit_result_store.py  set_fit_result / clear_fit_result (single setter)
    models.py          dataclasses analyse
    undo.py            undo stack générique
  io/                  loaders + logbook + orchestration IO  — AUCUN PyQt
    loaders/           1 fichier par backend (bessy, cls, solaris, common)
  physics/             numpy/scipy pur  — AUCUN PyQt
    fs.py              FSParams + extract_fs_map + cache helpers
    plot_compute.py    BM/MDC compute (display pipeline)
    waterfall_compute.py  WaterfallData + draw_waterfall_axes
    tb_fit.py, kink_analysis.py, gap_extraction.py  band analysis
    distortion.py, gamma.py, ef_calibration.py, fit.py
  ui/
    builders/          construction widgets (panels.py, menus.py)
    controllers/       orchestration Qt — 1 controller = 1 responsabilité
    widgets/           widgets PyQt + dialogs
  app.py               ArpesExplorer (orchestrateur léger) + main()
```

### Règles dures

1. **Plafond 700 LOC par fichier.** Tout dépassement bloque la prochaine feature. Splitter d'abord (sans compter les lignes en commentaires qui expliquent le code).
2. **PyQt interdit dans `physics/` et `io/`.** Logique testable headless.
3. **1 controller = 1 responsabilité.** Pas de fourre-tout. Si feature mélange 2 sujets, 2 méthodes dans 2 controllers.
4. **Aucun global mutable.** `self.ap` (chargé via `_load_ap()`) remplace l'ancien `AP` global.
5. **`__init__` de `ArpesExplorer`**: controllers instanciés AVANT `QTimer.timeout.connect(...)` sinon `__getattr__` lève AttributeError.
6. **Naming**: `*Controller` réservé à `ui/controllers/`. Logique pure → `*Fitter`/`*Manager`/`*Service`/`*Resolver`.
7. **Loaders extensibles.** L'app doit pouvoir intégrer progressivement de nouveaux backends de chargement ARPES. Ajouter un loader = nouveau module dédié dans `arpes/io/loaders/`, détection/orchestration dans l'API IO commune, tests dédiés, aucun PyQt. Ne pas coder de logique beamline spécifique dans `app.py`, `ui/`, ou les widgets.

## PROXY_MAP (`arpes/ui/controllers/proxy_map.py`)

- Mappe handlers `_on_*`/`_draw_*`/`_apply_*` exposés par `ArpesExplorer` vers leur controller via `__getattr__`.
- Plafond 150. **Actuel: 143**.
- Tout nouveau handler → ajouter une entrée + couvrir par `tests/test_ui_smoke.py::test_proxy_dispatch_resolves_every_entry`.
- **Pour ≥3 actions liées: verb-dispatch unique.** Exemple: `fit_zone_action(verb, payload)` couvre add/remove/set_active/toggle/rename/clear_results/list en 1 entrée. Préserve la marge.

## Patterns codifiés

### Free-function module + thin wrapper (split LOC)

Quand un fichier dépasse 700 LOC, extraire un bloc cohérent vers un module de **free functions** prenant la classe publique comme premier argument (`ctrl` ou `p`). La méthode publique reste comme wrapper 1-ligne pour ne casser ni callers ni tests.

Exemples livrés:
- `plot_controller.py` (1056→663) → `fit_overlay_drawer.py` + `kf_drag_handlers.py` + `mdc_edc_drawer.py`
- `band_analysis_panel.py` (873→671) → `band_analysis_summary.py` + `band_analysis_renders.py` + `band_analysis_presets.py`
- `plot_compute.py` (744→509) → `waterfall_compute.py`

### Single-setter pour state partagé

`entry.fit_result` est muté par plusieurs sites. Toute écriture passe par `arpes/core/fit_result_store.py`:

```python
from arpes.core.fit_result_store import set_fit_result, clear_fit_result
set_fit_result(entry, fr, zone_id=...)   # mirror automatique vers zone active
clear_fit_result(entry)                   # reset legacy + tous fit_zones
```

**Aucune écriture directe `entry.fit_result = ...` hors de `fit_result_store.py`**.

### Multi-zone fit

- Zones stockées `entry.fit_zones: list[dict]` UUID-keyed: `{id, label, color_idx, active, fit_params, fit_result}`.
- Sélection: `entry.active_zone_id: str | None`.
- Shim legacy: `entry.fit_result` = mirror zone active (≥6 consumers non-zone-aware: `results.py`, `aggregation.py`, `bootstrap.py`, `band_analysis_*`, `interaction mark-bad`, `plot overlay`).
- Tag axes au fit dans `fr["distorted"]` + `fr["grid_active"]`. `_draw_kf_overlay` refuse si état courant diffère (cf `_axis_state_mismatch`).

## Conseil agents

9 personas dans `.claude/agents/`. Quand spawn quel agent + template + format de
sortie : **`docs/ai/COUNCIL.md`**. Résumé : `architect` + `redteam` toujours,
`arbiter` tranche ; spawn via `general-purpose`, sonnet, caveman.

## Anti-patterns INTERDITS

- God class (`Explorer` 4136 LOC éliminée refonte α→σ, **ne pas réintroduire**).
- Global mutable.
- Lazy circular import `from arpes import app as _ae`.
- Écriture directe `entry.fit_result = ...` hors `fit_result_store`.
- Nouveau Controller fourre-tout (>4 sujets).
- `try/except: pass` silencieux sur persistance (cf HIGH-3 audit: perte zones silencieuse).
- PyQt6 import dans `physics/` ou `io/`.
- Forward `__setattr__` aveugle vers le parent (P3.1): un controller qui forward une écriture doit l'allow-lister via `_PARENT_WRITES` (sinon faute de frappe = attribut fantôme sur le parent, état réel périmé). 8 controllers concernés: distortion, browser, plot, fs, norm, gamma, pocket, pairing. Nouvel attribut parent écrit → l'ajouter au `_PARENT_WRITES` du controller.

## Git workflow

- SSH port 22 timeout réseau connu → workaround:
  ```bash
  git -c url."ssh://git@ssh.github.com:443/".insteadOf="git@github.com:" push
  ```
- Commits Co-Authored-By Claude via heredoc.
- Mono-dev, push direct main.

## Tests env

- Env `peaks` (micromamba) a PyQt6 → tests Qt actifs. Activer: `micromamba activate peaks`.
- Hors `peaks`, PyQt6 absent → ~61 tests skip (UI smoke + Qt-dependent).
- Skip permanents: `test_annotations.py`, `test_local_dft_loaders.py` (deps non installables headless).
- CI à venir: `xvfb-run pytest` pour activer Qt headless.
- Pre-existing failure: `test_yaml_schema_loads_band_axis_labels_and_efermi` (PyYAML manquant), pas une regression.

## Dette technique tracée

Voir **`docs/ai/BACKLOG.md`** (splits LOC à anticiper, architecture à terme,
audit 2e passe). Ne pas dupliquer ici.

## Modèles de données clés

### `FileEntry` (`arpes/core/session.py`)

Champs critiques: `ef_offset`, `view_mode`, `fit_params`, `fit_result`, `meta`, `bm_distortion`, `grid_correction`, `ef_correction`, `theory_overlay`, `band_analysis`, **`fit_zones`** (multi-zone), **`active_zone_id`**, `annotations`.

### `fit_result` schema

```python
{
  "e_fitted": list[float],          # binding E
  "kF_minus": list[list[float]],    # par paire, en π/a
  "kF_plus":  list[list[float]],
  "gamma_corrige": list[list[float]],
  "ensemble": {"kF_minus_std": ..., "kF_plus_std": ..., ...} | None,
  "params_hash": str,               # détection stale
  "distorted": bool,                # axis state tag (HIGH-1 fix)
  "grid_active": bool,
  "zone_id": str | None,            # multi-zone seulement
  "zone_label": str | None,
  "asymmetric_warning": str | None,
}
```

### `fit_zones[i]` schema

```python
{
  "id": str,                # uuid4().hex[:8]
  "label": str,             # "Z1", "Z2", ...
  "color_idx": int,         # index ZONE_PALETTE
  "active": bool,           # inclu dans Run all
  "fit_params": dict,       # asdict(FitParams)
  "fit_result": dict | None,
}
```

## Mémoire externe

User auto-memory disponible dans `~/.claude/projects/-Users-alexandrespecht/memory/`. Voir `MEMORY.md` pour les memos persistants entre sessions (workflow conseil ARPES, état projet).

## Commandes utiles

```bash
# Tests
python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py -q

# Tests UI smoke (besoin Qt headless)
xvfb-run python3 -m pytest tests/test_ui_smoke.py

# Compter PROXY_MAP entries
python3 -c "from arpes.ui.controllers.proxy_map import PROXY_MAP; print(len(PROXY_MAP))"

# Vérifier zones LOC
wc -l arpes/ui/controllers/*.py arpes/ui/widgets/*.py arpes/physics/*.py | sort -rn | head -15

# Chercher violations directes fit_result
rg "entry\\.fit_result\\s*=" arpes tests

# Chercher imports PyQt interdits
rg "PyQt6|QtCore|QtGui|QtWidgets" arpes/physics arpes/io
```

## Re-derivation avant gros changement

Avant feature/refactor non trivial:

1. Relire ce fichier + `git status --short`.
2. Identifier fichiers touchés et compter LOC avant édition.
3. Vérifier si changement touche: physique, IO, UI controller, session schema, fit_result, fit_zones.
4. Si physique: isoler calcul dans `physics/`, ajouter tests numpy/scipy headless.
5. Si UI: garder orchestration dans controller dédié, widget sans logique métier lourde.
6. Si session/model: préserver compatibilité JSON ou bump `Session.VERSION`.
7. Si nouveau handler public: ajouter `PROXY_MAP` entry + smoke test.
8. Finir par tests ciblés, puis suite standard si blast radius large.

## Définition de done

- Aucun fichier modifié >700 LOC sauf dette déjà explicitement connue.
- Aucun PyQt dans `physics/` ou `io/`.
- Aucune mutation directe de `entry.fit_result`.
- Tests pertinents lancés ou impossibilité notée clairement.
- Changement documenté si schema, workflow utilisateur, ou dette technique changent.
