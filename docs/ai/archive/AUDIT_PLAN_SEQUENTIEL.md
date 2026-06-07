# Plan d'audit séquentiel — ARPES Explorer

**Date** : 2026-06-04
**Cible** : couvrir les zones non auditées par la première passe (`AUDIT_2026-06-04.txt`)
**Stratégie** : 1 agent par domaine, exécution séquentielle, prompts minimaux, scope étroit, livrables courts.
**Optimisation tokens** : pas de recherche web sauf nécessité explicite, pas de lecture récursive, scope fichier:ligne pré-spécifié, output capé.

**Compatibilité runners** : Claude Code (Sonnet) ET OpenAI Codex CLI (`o4-mini` ou `gpt-5-mini` selon dispo). Voir section dédiée plus bas. Le plan est neutre côté outil : chaque agent = 1 invocation indépendante, prompt self-contained, livrable = append markdown.

---

## Principes globaux (s'appliquent à TOUS les agents, tous runners)

1. **Modèle** :
   - Claude Code → `model: "sonnet"` (jamais Opus, pas Haiku)
   - Codex CLI → `o4-mini` ou `gpt-5-mini` (modèle mid-tier, JAMAIS o4/gpt-5 full)
2. **Mode** : caveman (drop articles/filler, fragments OK, code normal)
3. **Scope dur** : chaque agent ne lit QUE les fichiers listés dans son brief, jamais explorer ailleurs.
4. **Output cap** : 30 findings max, format `fichier:ligne — problème — fix`.
5. **Pas de recherche web** sauf bullet explicitement marqué `[WEB]`.
6. **Pas de praise**, pas de récap général, pas de "I'll now…".
7. **Livrable** : append direct dans `AUDIT_RESULTS.md` (section par agent).
8. **Token budget cible** : 15k–30k tokens/agent. Si dépasse 40k → STOP, livre partiel.
9. **Outils requis minimaux** : lecture fichier, écriture/append fichier, grep, glob. Pas d'exécution code, pas de réseau (sauf `[WEB]`).

---

## Sequence (10 agents, ~1 à exécuter par session)

### Agent 1 — Tests & coverage
**Scope** : `tests/`, `app/conftest.py`, `pytest.ini` / `pyproject.toml` test config.
**Mission** :
- Lister tests qui mockent là où devraient être intégration (loaders Qt, fits réels).
- Identifier modules `arpes/` sans test (parcours `find tests/ -name 'test_*.py'` → diff avec modules existants).
- Vérifier présence test cross-version Session (load v0/v1).
- Vérifier fixtures cross-loader (CLS + BESSY + Solaris avec sample non-BaNi).
- Détecter `unittest.skipIf` / `skipUnless` qui cachent dead tests.
**Token cap** : 20k.
**Livrable** : tableau `module | test_existe | coverage_estim | skipUnless | gap`.

---

### Agent 2 — Performance & mémoire
**Scope** : zones probables hot-path → `arpes/io/loaders/{cls,bessy,solaris}.py`, `arpes/analysis/bootstrap.py`, `arpes/physics/fit.py`, `arpes/physics/pocket.py`, `arpes/physics/kz.py`, `arpes/core/session.py:save/load`.
**Mission** :
- Repérer boucles Python pures sur grosses arrays (devrait être vectorisé numpy).
- Repérer copies inutiles (`.copy()`, `np.array(x)` redondants).
- Repérer recalculs sans cache (ex : kx_axis recalculé à chaque draw).
- Repérer chargements file I/O dans hot-paths UI (devrait être lazy/threaded).
- Estimer complexité asymptotique sur boucles imbriquées.
- Identifier zones où `QApplication.processEvents()` masque blocage réel.
**Pas de profilage runtime** (l'agent ne lance pas l'app).
**Token cap** : 25k.
**Livrable** : findings priorisés par gain estimé (×N speedup grosseur).

---

### Agent 3 — Robustesse fichiers / erreurs / edge cases
**Scope** : `arpes/io/loaders/`, `arpes/io/logbook*.py`, `arpes/io/file_pairing.py`, `arpes/core/session.py`, `arpes/io/recent_sessions.py`.
**Mission** :
- Lister tous `except Exception:` (catch trop large).
- Lister tous `except: pass` ou variantes (swallow silencieux).
- Détecter chemins NaN / Inf non gardés en physique.
- Détecter accès dict[k] sans `get` sur métadonnées loader.
- Vérifier paths absolu vs relatif (Windows `\\`, espaces, Unicode accents).
- Lister hardcodes `.txt`/`.ibw`/`.h5` qui ignorent variantes casse.
- Risque corruption session : qu'arrive-t-il si crash pendant `save()` ?
**Token cap** : 20k.
**Livrable** : findings + 3 scénarios reproductibles de plantage probable.

---

### Agent 4 — Sécurité dépendances & cross-platform
**Scope** : `pyproject.toml` ou `requirements.txt`, `arpes/__init__.py`, racine repo.
**Mission** :
- Lister deps non pinnées (sans `==` ou `~=`).
- Vérifier versions Python supportées vs syntaxe utilisée.
- Repérer usage `subprocess` / `os.system` (risque shell injection si input user).
- Vérifier paths : `Path()` partout vs `os.path` mixé ?
- Vérifier presence `__main__.py` ou entry-point cross-platform.
- Détecter import `pwd`/`fcntl`/`resource` (Unix-only).
- Détecter chemins absolus `/Users/`, `/home/`, `C:\\` codés.
- `[WEB]` SI deps obsolètes → check 1 fois page PyPI pour version actuelle.
**Token cap** : 15k.
**Livrable** : matrice `dep | pinned | platform_OK | vulnérabilité_connue`.

---

### Agent 5 — Code mort & duplication
**Scope** : tout `arpes/`, mais focus sur controllers et widgets (déjà identifiés gros).
**Mission** :
- `grep` fonctions définies mais jamais appelées (cross-référencer définitions vs usages).
- Imports `from X import Y` où Y jamais utilisé.
- Branches `if False:`, `pass`, blocs commentés massifs.
- Constantes définies dans 2+ endroits avec même valeur.
- Methods `_helper` qui dupliquent logique entre controllers (P3.1 deja fait sur __getattr__, ici chercher autre).
- Méthodes mortes après refonte α→σ (commits récents).
**Token cap** : 20k.
**Livrable** : liste fichier:ligne → suppression sûre OU nécessite vérification.

---

### Agent 6 — Onglets UI profonds (non couverts redteam initial)
**Scope** : `arpes/ui/widgets/{mdc_diagnostics.py, waterfall.py, edc.py, band_analysis_panel.py, zones_strip.py, notes.py}`, `arpes/ui/widgets/plots/`, panel Notes, panel Aide.
**Mission** :
- Pour chaque tab profond : workflow utilisateur typique en 5 étapes max, repérer friction.
- Vérifier labels axes (LaTeX vs ASCII), colorbars, unités.
- Vérifier export figures depuis CE tab (pas juste FS/Results).
- Repérer contrôles cachés (combobox sans tooltip, slider sans range affiché).
- Vérifier état "vide" (avant load) : message clair ou écran blanc ?
- Mode sombre/clair cohérent ?
**Token cap** : 25k.
**Livrable** : tableau `tab | workflow_step1..5 | friction | label_OK | export_OK`.

---

### Agent 7 — Module `arpes/theory/` (DFT, alignment, MP)
**Scope** : `arpes/theory/{alignment.py, comparison.py, conversion.py, materials_project.py, fs_isocontour.py, band_picker.py, band_select.py, plot.py, local_loaders.py, selection.py, data.py, models.py, labels.py}`.
**Mission** :
- Rigueur physique alignment µ-shift, Z-renormalization (cf P3 audit déjà signalé alignment.py:9-20).
- Connexion DFT npz ↔ FS measured : interpolation, kz slice, contour matching.
- Comparaison Hausdorff vs autres métriques (Fréchet ? IoU ?).
- Materials Project : quota API, fallback si offline, cache local.
- Conventions origine/unité DFT (eV vs Hartree ? a.u. vs Å ?).
- `[WEB]` 1 vérif Materials Project API rate limit + auth.
**Token cap** : 30k.
**Livrable** : findings physique + intégration + 1 suggestion architecturale.

---

### Agent 8 — Logbook : heuristiques inférence colonnes
**Scope** : `arpes/io/logbook.py`, `arpes/io/logbook_matching.py`, `arpes/io/logbook_io.py`, `tests/test_logbook*.py` si existe.
**Mission** :
- Pour chaque heuristique (`_sniff`, `_drop_implausible_mappings`, `_pick_direction_column`, `_infer_legacy_measurement_plan_mapping`), lister inputs où elle échoue.
- Construire 5 cas synthétiques (xlsx fictifs) où l'inférence se trompe silencieusement.
- Proposer UI "réviser le mapping" avant validation (dialog).
- Vérifier i18n : colonnes nommées en allemand/anglais britannique.
- Quantifier faux positifs/négatifs sur ranges hv (4-200) et T (3-500).
**Token cap** : 20k.
**Livrable** : cas d'échec reproductibles + matrice heuristique × labo.

---

### Agent 9 — Reproductibilité numérique & déterminisme
**Scope** : tout `arpes/physics/`, `arpes/analysis/`, `arpes/io/export.py`.
**Mission** :
- Lister tous `np.random` / `random.` sans seed explicite ou avec seed non loggué.
- Lister opérations float order-dependent (`sum`, `mean` sur grilles, parallèle).
- Vérifier bootstrap : seed est-il dans payload de sortie ?
- Vérifier ensemble_fit : jitter reproductible ?
- Vérifier export : hash inputs + seed + version code stockés ?
- Identifier zones où 2 runs identiques peuvent diverger (threading, dict ordering, set).
**Token cap** : 15k.
**Livrable** : liste fichier:ligne + flag "reproductible / non-reproductible / partiellement".

---

### Agent 10 — Documentation utilisateur (lecture, pas écriture)
**Scope** : `README.md`, `CLAUDE.md`, `app/.claude/agents/`, onglet Aide app (`arpes/ui/widgets/help_panel.py` si existe), tooltips.
**Mission** :
- Vérifier README couvre install + quickstart + multi-loader + exemple.
- Vérifier docstrings publiques sur APIs principales (Session, FileEntry, characterize_pocket).
- Vérifier onglet Aide à jour vs UI réelle (commandes/menus existants).
- Lister tooltips manquants sur boutons critiques (FS panel, pocket wizard, EF cal).
- Lister tooltips trompeurs (qui décrivent l'ancien comportement).
- **NE PAS RÉÉCRIRE** la doc, juste signaler les manques.
**Token cap** : 15k.
**Livrable** : checklist `doc_section | présente | à_jour | gap`.

---

### Agent 11 — Dialogs UI (11 fichiers)
**Scope** : `arpes/ui/widgets/dialogs/{bz_selector.py, ef_calibration.py, export_dialog.py, imag_self_energy.py, mp_search.py, multi_file_analysis.py, pocket_result.py, pocket_wizard.py, self_energy.py, session_diff.py, theory_band_picker.py}`.
**Mission** :
- Pour chaque dialog : entry-point (qui l'ouvre), inputs requis, validation, état "vide".
- Repérer dialogs modal qui bloquent UI sans cancel possible.
- Vérifier sauvegarde état (champs pré-remplis lors réouverture).
- Vérifier labels/tooltips (LaTeX, unités, plage valide).
- Vérifier exports depuis dialog (figure, CSV) : preset utilisé, sidecar metadata.
- Repérer dialogs qui dupliquent fonctionnalité de panel principal (redondance).
- Vérifier i18n (français/anglais mix).
**Token cap** : 25k.
**Livrable** : tableau `dialog | entry | inputs | validation | export_ok | gaps`.

---

### Agent 12 — Controllers résiduels (15 fichiers)
**Scope** :
```
arpes/ui/controllers/{
  kf_drag_handlers.py, plot_model_helpers.py, mdc_edc_drawer.py,
  fit_overlay_drawer.py, fit_clear.py, interaction_selection.py,
  gamma_lifecycle.py, session_io_controller.py, batch_controller.py,
  kz_controller.py, fs_controller.py, browser_controller.py,
  distortion_controller.py, norm_controller.py, pairing_controller.py,
  fit_zones_controller.py, fit_zone_runner.py,
  band_analysis_controller.py, logbook_controller.py, plot_controller.py
}
```
**Mission** :
- Pour chaque ctrl : sujets gérés (cap 4 par CLAUDE.md), LOC, méthodes >100 LOC.
- Détecter `__getattr__` parent-forward résiduels (P3.1 audit déjà identifie 8).
- Détecter mutations `entry.X = ...` qui contournent stores (cf. P3.3 single-setter).
- Détecter duplication logique entre ctrls (3+ ctrls qui font la même chose).
- Vérifier signal/slot Qt branchés mais jamais émis (dead wires).
- Repérer ctrl qui dépendent de `arpes.app` (inversion couche).
- Vérifier verbs PROXY_MAP : nouvelles entrées non documentées vs cap 150.
**Token cap** : 30k.
**Livrable** : matrice ctrl×{LOC, sujets, parent_forward, dead_wires, dépendances_app}.

---

### Agent 13 — Core data + analysis résiduels + browsers + kink
**Scope** :
```
arpes/core/{models.py, undo.py, fit_result_store.py}
arpes/analysis/{aggregation.py, session_diff.py, results.py}
arpes/physics/kink_analysis.py
arpes/ui/widgets/browsers/{files.py, file_describer.py}
arpes/ui/builders/panels.py  (re-audit ciblé wire_ui_signals)
arpes_explorer.py, arpes_plots.py  (shims racine)
```
**Mission** :
- `core/models.py` : dataclasses vs dicts opaques, validation, defaults.
- `core/undo.py` : profondeur stack, mémoire, granularité actions, threadsafety.
- `core/fit_result_store.py` : invariant single-setter respecté, getters/setters cohérents.
- `analysis/aggregation.py` : statistiques cross-sample, gestion NaN, unités.
- `analysis/session_diff.py` : robustesse comparaison sessions, edge cases.
- `analysis/results.py` (résiduels non couverts par audit physique).
- `physics/kink_analysis.py` : λ = −∂ReΣ/∂ω, cohérence avec self_energy.py:140-150.
- `browsers/files.py` + `file_describer.py` : friction navigation, raccourcis clavier, état multi-sélection.
- `panels.py` wire_ui_signals : monolithe (audit initial finding 6) → confirmer split nécessaire.
- Shims racine : trivialité confirmée OU export indésirable.
**Token cap** : 30k.
**Livrable** : findings par fichier + recoupement avec audit initial pour éviter duplication.

---

## Estimation budget total

| Agent | Token cap | Cumul |
|------:|----------:|------:|
| 1 Tests | 20k | 20k |
| 2 Perf | 25k | 45k |
| 3 Robustesse | 20k | 65k |
| 4 Sec/cross-plat | 15k | 80k |
| 5 Code mort | 20k | 100k |
| 6 UI profonde | 25k | 125k |
| 7 Theory | 30k | 155k |
| 8 Logbook | 20k | 175k |
| 9 Repro | 15k | 190k |
| 10 Doc | 15k | 205k |
| 11 Dialogs | 25k | 230k |
| 12 Controllers résiduels | 30k | 260k |
| 13 Core+analysis+browsers | 30k | 290k |

**Total ~290k tokens** sur Sonnet/o4-mini (~3× moins cher qu'Opus à même budget).
Comparaison : 1ère passe = 370k tokens, 3 agents Opus parallèle pour 60-70% coverage.
Plan 13 agents = **~95% coverage** pour ~78% du coût.

---

## Template prompt agent (réutilisable, runner-agnostique)

Le même prompt fonctionne pour Claude Code (`Agent` tool) et Codex CLI (`codex exec` / `codex --model …`). Substituer `{liste fichiers}`, `{bullets agent N}`, `{cap}`, `N`, `{titre}` avant lancement.

```
Tu es agent audit ARPES. Réponds en mode caveman (drop articles/filler/
pleasantries, fragments OK, code normal).

CONTEXTE : app PyQt6 ARPES, /Users/alexandrespecht/Documents/Stage_M2/code/app/.
Première passe audit existe : AUDIT_2026-06-04.txt — NE PAS dupliquer ses
findings. Couvrir UNIQUEMENT scope ci-dessous.

SCOPE (lecture autorisée UNIQUEMENT) :
- {liste fichiers}

MISSION :
{bullets agent N}

CONTRAINTES :
- 30 findings max
- format `fichier:ligne — problème — fix`
- pas de recherche web sauf bullet marqué [WEB]
- pas de praise, pas de récap, livre direct
- si scope dépasse, livre partiel mais STOP

LIVRABLE :
Append section dans /Users/alexandrespecht/Documents/Stage_M2/code/app/AUDIT_RESULTS.md
avec header `## Agent N — {titre}`.

Token cap : {cap}.
```

---

## Compatibilité Codex CLI (OpenAI)

### Lancement Codex

```bash
# Codex CLI installé (npm i -g @openai/codex ou pip install openai-codex)
cd /Users/alexandrespecht/Documents/Stage_M2/code/app

# Exécution single-shot d'un agent (lire prompt depuis fichier)
codex exec --model o4-mini --cwd . < .audit/prompts/agent_5_dead_code.txt

# OU mode interactif sandboxé
codex --model o4-mini --sandbox workspace-write
> (coller prompt)
```

Recommandations Codex :
- **Sandbox** : `--sandbox workspace-write` (lecture totale, écriture limitée à `AUDIT_RESULTS.md`).
- **Approval mode** : `--approval-mode never` pour audits read-only ; sinon `on-request`.
- **Modèle** : `o4-mini` (cher mais raisonneur) OU `gpt-5-mini` quand dispo. Éviter modèles "instant" qui ratent les nuances physiques.
- **Pas de tools réseau** sauf agent avec `[WEB]` → activer `--allow-net` ponctuellement.

### Différences notables Claude vs Codex

| Point | Claude Code | Codex CLI |
|---|---|---|
| Spawn subagent | `Agent` tool | `codex exec` séparé (1 process = 1 agent) |
| Lecture fichier | `Read` tool intégré | shell `cat` ou tool fichier interne |
| Append fichier | `Edit`/`Write` tool | shell `cat >>` ou tool écriture |
| Grep | `Grep` tool / Bash | shell `rg` / `grep` |
| Glob | `Glob` tool / Bash | shell `find` / `fd` |
| Approval | permission modes | `--approval-mode` |
| Modèle | `model: "sonnet"` param | `--model o4-mini` flag |
| Output | tool result au parent | stdout (rediriger vers `AUDIT_RESULTS.md`) |

### Wrapper bash unifié (optionnel)

Pour lancer 1 agent indifféremment dans 2 runners, créer `.audit/run_agent.sh` :

```bash
#!/usr/bin/env bash
# Usage: ./run_agent.sh <N> [claude|codex]
set -euo pipefail
N="${1:?agent number required}"
RUNNER="${2:-claude}"
PROMPT=".audit/prompts/agent_${N}.txt"
OUT="AUDIT_RESULTS.md"

case "$RUNNER" in
  claude)
    # Claude Code via headless: cat prompt | claude --no-interactive
    cat "$PROMPT" | claude -p --model claude-sonnet-4-6 >> "$OUT"
    ;;
  codex)
    codex exec --model o4-mini --cwd . --sandbox workspace-write < "$PROMPT" >> "$OUT"
    ;;
  *)
    echo "unknown runner: $RUNNER" >&2; exit 1
    ;;
esac
echo "" >> "$OUT"  # blank line separator
echo "✓ agent $N done ($RUNNER)"
```

Stocker chaque prompt agent dans `.audit/prompts/agent_N.txt` (substitution déjà faite). Permet exécution scriptée séquentielle :

```bash
for N in 5 1 12 13 3 9 2 4 8 7 11 6 10; do
  ./run_agent.sh $N codex     # ou claude
  read -p "Agent $N done. Continuer ? "
done
```

### Limites Codex à connaître

- Pas de contexte persistant entre `codex exec` → chaque agent repart à zéro (souhaité ici, mais ne pas tenter de chaîner).
- Codex ne sait pas (encore) parler `caveman skill` natif → la consigne dans le prompt suffit, mais sortie peut dériver vers prose normale. Tolérable.
- Output Codex parfois verbose même avec consigne → cap dur 30 findings critique.
- Pas de `WebSearch` natif partout → si agent a `[WEB]`, préférer Claude OU fournir l'URL/fichier en input.

---

## Ordre d'exécution recommandé

Critère : commencer par audits qui informent les suivants.

1. **Agent 5 (code mort)** — élimine bruit avant autres audits, allège base.
2. **Agent 1 (tests)** — révèle ce qui est protégé vs non.
3. **Agent 12 (controllers résiduels)** — confirme état refonte α→σ pour tout le reste.
4. **Agent 13 (core+analysis+browsers)** — fondations data model avant audits aval.
5. **Agent 3 (robustesse)** — crashs probables, urgent pour user externe.
6. **Agent 9 (reproductibilité)** — bloquant figures publiables.
7. **Agent 2 (perf)** — quand le reste est propre.
8. **Agent 4 (sec/cross-plat)** — avant distribution multi-labo.
9. **Agent 8 (logbook)** — multi-labo dépend de logbook robuste.
10. **Agent 7 (theory)** — module isolé, audit après cœur stabilisé.
11. **Agent 11 (dialogs)** — points d'interaction user, dépendent des controllers stabilisés.
12. **Agent 6 (UI profonde)** — après dialogs auditées et controllers nettoyés.
13. **Agent 10 (doc)** — toujours en dernier, doc reflète l'état final.

---

## Économies vs option naïve

- **Naïf** (3 agents Opus / GPT-5 full parallèle scope large) = ~370k tokens, contexte principal pollué par 3 résumés concurrents.
- **Plan séquentiel Sonnet / o4-mini** = ~205k tokens, 1 agent à la fois, contexte principal reçoit 1 résumé à la fois et peut le digérer / agir avant suivant.

Tarifs indicatifs (2026, ordres de grandeur) :
- Claude Sonnet 4.6 : ~$3/M in, $15/M out → audit complet ~$5-8
- OpenAI o4-mini : ~$3/M in, $12/M out → audit complet ~$5-7
- Claude Opus : ~$15/M in, $75/M out → audit complet ~$25-40 (5× plus)

**Gain estimé : ~3-5× moins cher (Sonnet ou o4-mini vs full models), meilleure rétention.**

---

## Gouvernance

- Lancement : 1 agent par session de travail, jamais 2 simultanés.
- Validation : relire le livrable agent avant lancer suivant.
- Append `AUDIT_RESULTS.md`, jamais réécrire.
- Si findings agent N modifient scope agent N+k → mettre à jour ce .md avant exécution.

FIN.
