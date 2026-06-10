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

_(rien)_

---

## DETTE PLANIFIÉE (P2)

### Splits LOC à anticiper
- `fs_panel.py` 713 brut / 640 hors commentaires → extraire `FermiSurfaceCanvas`
  (~290 LOC) vers `widgets/fs_canvas.py` au prochain ajout (plan architecte 2026-06-10).
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
- Modes **1D-MDC curvature** (le long de k) et **1D-EDC curvature** (le long de E) :
  features distinctes (nouveau mode combo + compute), DEFER conseil 2026-06-10.
  Le physicien préfère MDC-1D pour suivre une bande métallique traversant EF.
- Avertissement UI si grille k non-uniforme (Δk varie >5%) en modes dérivés
  (gaussian_filter en px suppose grille uniforme) — conseil 2026-06-10, non bloquant.
- Regroupement panneau FS (proposition UX 2026-06-10, DEFER arbitre) :
  "FS Extraction" (EF window, norm, smoothing, cmap, distortion, redraw) open /
  "Lattice & Units" collapsed / Γ open / "Brillouin Zone" fusionné collapsed /
  "BM Cuts" collapsed / Pockets open. Cosmétique, après stabilité des features viz.
- Preview sous-échantillonnée pendant drag (NO-GO arbitre 2026-06-10 : debounce
  150 ms suffit, goulot = intégration numpy pas pcolormesh, stride aliase les
  poches fines). Rouvrir SEULEMENT avec un benchmark mesuré qui prouve le besoin.
- Colonne n_Luttinger dans Results (NO-GO conseil 2026-06-10 tant que la
  dimensionnalité du compte n'est pas définie — kF²/π 2D vs autre; lié au
  Luttinger des poches FS qui lui existe déjà).
