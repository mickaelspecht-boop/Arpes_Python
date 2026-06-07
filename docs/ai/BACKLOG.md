# BACKLOG — ARPES Explorer

Source unique du « quoi faire ensuite ». **Une seule** liste, priorisée. Pas de
nouveau fichier `*_PLAN.md` à la racine : tout passe ici.

**Cycle de vie d'un item**
1. Nouveau travail → ajouter une section ci-dessous (titre + pourquoi + done-when).
2. Besoin d'un gros design → UN fichier `docs/ai/plans/<slug>.md`, lié depuis l'item. Pas à la racine.
3. Terminé → supprimer la section + ajouter 3 lignes dans `DECISIONS.md`. Le code est la vérité, ne pas re-raconter.

Priorités : **P0** bloquant en cours · **P1** prochain · **P2** dette planifiée · **P3** idée.

---

## EN COURS

### Traduction app FR→EN (P0)
Tout l'app en anglais (code/UI/commentaires/docstrings/help in-app), **sauf**
mots-clés de matching logbook (`logbook*.py` : listes de noms de colonnes FR =
clés, pas du texte UI). Parallélisé via sous-agents, partiellement fait avant
limite de session.
- **Reste à faire** : vérifier/finir controllers (ex. `fit_runner_controller.py`
  messages « Fit déjà en cours » encore FR), widgets, dialogs/plots/browsers,
  messages physics/io user-facing, docstrings publiques.
- Mettre à jour les tests qui assertent du FR (`test_fit_reentrancy_p37.py`
  « déjà en cours » ; tests pocket « rejetée »/« lissée » ; ~13 fichiers repérés).
- Vérifier que les listes mots-clés logbook sont **intactes**.
- **Done-when** : `grep` FR ne trouve plus de string user-facing FR ; suite verte
  sous `peaks` ; launch réel OK.

---

## DETTE PLANIFIÉE (P2)

### Splits LOC à anticiper
- `fit_runner_controller.py` 700 LOC, 5 sujets → split `_fit_*` au prochain ajout.
- `band_analysis_controller.py` ~510 LOC, 6 sujets → split en 4 ctrls.
- 6 fichiers zone jaune 660-700 LOC à surveiller (`wc -l` cf CLAUDE.md commandes).

### Architecture à terme
- Itération `fit_zones` explicite dans 6 consumers → tuer le shim `entry.fit_result`.
- `FitZone` (P3.4) : dataclass + normalize posés, mais runtime reste `dict`
  (93 sites d'accès par clé). Conversion complète des consumers = à terme.
- `QThreadPool` pour fits longs : garde de ré-entrance `_fit_busy` posée (P3.7) ;
  vrai threading à faire (gain = réactivité UI + annulation, pas vitesse, numpy GIL-bound).
- Cache LRU `_get_work_data` distortion-warped (ensemble fit recompute 30×).
- Consolidation verb-dispatch sur `_band_analysis_ctrl` (−5 entrées PROXY_MAP).
- Hoist imports lazy dans wrappers une fois cycles vérifiés.

### Audit 2e passe (non démarré)
Plan 13-agents séquentiel des zones non couvertes par la 1re passe :
`docs/ai/archive/AUDIT_PLAN_SEQUENTIEL.md`. À dérouler agent par agent ; chaque
résultat → item ici, puis `DECISIONS.md`.

---

## IDÉES (P3)
- CI `xvfb-run pytest` pour activer Qt headless (cf CLAUDE.md tests env).
