# Spec — Sélecteur visuel de bandes DFT (remplace le tableau)

État : **proposition / à implémenter**. Aucun code écrit. Document de
conception pour reprise ultérieure (économie de tokens).

## 1. Objectif

Remplacer la table cochable (`tbl_theory_bands`, peu lisible : on coche
des index sans voir à quoi ils correspondent) par un **bouton** qui
ouvre une **fenêtre montrant le diagramme de bandes DFT**. L'utilisateur
**clique directement sur les bandes** voulues → elles passent en
**surbrillance**. Choix du **chemin/segment** dans la même fenêtre.
Objectif : ergonomique, visuel, accès direct.

## 2. Flux utilisateur

1. Onglet DFT → après import MP/local, bouton **« Choisir les bandes… »**
   (remplace la table, qui peut rester repliée en mode avancé).
2. Clic → ouverture `TheoryBandPickerDialog` (modale non bloquante ou
   modale simple).
3. La fenêtre affiche le **diagramme E(k)** complet du chemin DFT
   (toutes bandes, axe k = labels Setyawan Γ, X, M…).
4. Interactions :
   - **Clic sur une courbe de bande** → bande sélectionnée, repassée en
     **surbrillance** (couleur vive + épaisseur), les autres atténuées.
   - **Clic à nouveau** → désélection.
   - **Glisser un rectangle** (lasso/box) → sélectionne toutes les
     bandes traversant la boîte (sélection multiple rapide).
   - **Menu déroulant « Chemin »** : liste les branches réelles MP
     (`branch_display_names`, ex `Γ-X`, `X-M`). Choisir un segment
     restreint l'affichage à cette portion + le servira à l'overlay.
   - **Spinbox « Fenêtre E_F ± (eV) »** : grise/atténue les bandes ne
     croisant pas la fenêtre (réutilise `bands_crossing_ef`).
   - Optionnel : **clic droit sur bande** → infos (index, E min/max,
     caractère orbital si projections, `band_meta`/`band_character`).
5. Boutons : **Appliquer** (réinjecte la sélection), **Annuler**,
   **Tout désélectionner**, **Inverser**.
6. À *Appliquer* : la sélection est convertie en `band_indices`
   (`format_band_indices`, ex `1,3,5-8`) + segment choisi → mêmes
   chemins de données que la table actuelle (rétrocompat totale).

## 3. Maquette (ASCII)

```
┌─ Bandes DFT — mp-568280 BaNi₂As₂ ───────────────────────────┐
│ Chemin: [ Γ-X ▾ ]   Fenêtre E_F ±[0.50] eV  ☑ proj. couleur │
│                                                              │
│  E−E_F                                                       │
│  (eV)   ╱‾‾╲      ___                                         │
│   1 ─  ╱    ╲    ╱   ╲      ← bandes atténuées (non sélect.)  │
│   0 ──━━━━━━━━━━━━━━━━━━━  ← E_F (pointillé cyan)             │
│  -1 ─  ▓▓▓▓▓▓ bande 24 (SURBRILLANCE, cliquée)               │
│  -2 ─    ╲__╱   ╲___╱                                        │
│        Γ        X          (labels du segment)               │
│                                                              │
│ Sélection: 23, 24, 26   [Tout désélec.] [Inverser]           │
│                              [Annuler]  [Appliquer]          │
└──────────────────────────────────────────────────────────────┘
```

## 4. Réutilisation de l'existant (peu de code neuf)

- Données : `TheoryBandData` (`bands`, `k_distance`, `k_distance_abs`,
  `branches`, `band_meta`, `band_character`, `labels`). Tout est déjà
  stocké dans l'overlay courant.
- Sélection : **réutiliser** `parse_band_indices` /
  `format_band_indices` (sync avec le champ legacy `band_indices` et la
  config) → zéro divergence avec le pipeline overlay actuel.
- Filtrage : `bands_crossing_ef`, `branch_display_names`,
  `_branch_index_for_segment`.
- Couleur/caractère : `_band_color` (tab20) +
  `aggregate_projection_character` déjà en place.
- Le dialog **ne calcule rien de physique** : il lit `TheoryBandData`
  et écrit `band_indices` + `segment`. Logique pure inchangée.

## 5. Architecture proposée (God-class-free)

- `arpes/ui/widgets/dialogs/theory_band_picker.py` (NOUVEAU) :
  `TheoryBandPickerDialog(QDialog)`. Contient un `MplCanvas` (réutilise
  le wrapper existant + toolbar/reset). Pur Qt + matplotlib, **aucune**
  logique métier (reçoit `TheoryBandData` + sélection initiale, émet
  `selection_applied(list[int], str segment)`).
- `params_theory.py` : remplacer la table par
  `btn_theory_pick_bands` (la table peut devenir un repli « avancé »).
  Signal `theory_band_picker_requested`.
- `theory_overlay_controller.py` : `_open_theory_band_picker()` —
  construit le dialog depuis l'overlay courant, branche
  `selection_applied` → écrit `band_indices`/`segment` dans la config,
  `_on_theory_overlay_changed()` (fast path overlays-only).
- `app.py` `_PROXY_MAP` + `panels.py` wiring (1 ligne chacun).
- Picking matplotlib : `Line2D` par bande avec `picker=True` +
  `mpl_connect('pick_event')`. Box-select via `RectangleSelector`.

## 6. Détails ergonomie

- **Surbrillance** : sélectionnées lw≈2.0 couleur tab20 par index ;
  non sélectionnées lw≈0.5 alpha≈0.25 gris. Contraste fort.
- Survol (`motion_notify`) : surligner légèrement la bande sous le
  curseur + tooltip `b{idx} · E[min,max] · caractère`.
- Le segment choisi grise hors-branche (réutilise la logique de masque).
- Persistance : sélection rouverte = état courant (depuis
  `band_indices`), pas remise à zéro.
- Accessibilité : double-clic sur une bande = sélection exclusive
  (juste celle-là). Échap = Annuler.
- Fenêtre redimensionnable, zoom via toolbar + bouton « Vue init »
  (déjà fiable côté BM, réutiliser `MplCanvas`).

## 7. Phasage

- **P1** : dialog + clic-sélection + surbrillance + Appliquer/Annuler,
  sync `band_indices`. (cœur, suffisant pour usage)
- **P2** : menu Chemin (segment) + fenêtre E_F dans le dialog.
- **P3** : box-select, survol/tooltip, couleur par caractère orbital,
  double-clic exclusif.

## 8. Risques / points conseil (à valider avant code)

- arpes-pyqt-dev : `pick_event` sur ~40 lignes = OK perf ; éviter
  redraw complet à chaque pick (utiliser `set_lw/set_alpha` + `draw_idle`,
  pas de replot).
- arpes-ux : ne pas dupliquer les contrôles (E_F, proj) entre panneau
  et dialog → le dialog est la source pendant l'édition, applique en
  sortie.
- arpes-redteam : sélection vide = comportement actuel (auto top-N) ;
  ne pas casser `band_indices` legacy ni la session sauvegardée ;
  dialog non bloquant ne doit pas désynchroniser si l'overlay change
  dessous (recharger si `material_id` change → fermer/avertir).
- arpes-architect : aucune logique physique dans le dialog ; tout passe
  par `band_select`/`models` existants.
- DFT local sans `branches` : menu Chemin masqué, reste fonctionnel.

## 9. Tests prévus

- Unit (logique, déjà couverte) : `format_band_indices` ↔
  `parse_band_indices` round-trip (existe).
- UI smoke : instancier `TheoryBandPickerDialog` avec un
  `TheoryBandData` factice, simuler sélection → vérifier signal émet
  bons index + segment ; bouton présent dans `params_theory`.
- Non-régression : table legacy (mode avancé) toujours synchro.

## 10. Décisions à confirmer (prochaine session)

1. Dialog **modal** (bloque jusqu'à Appliquer/Annuler) vs **non
   bloquant** (live preview sur la BM pendant la sélection) ?
2. La table actuelle : **supprimée** ou **conservée en repli avancé** ?
3. Box-select (P3) souhaité ou clic seul suffisant ?

> Reprise : implémenter P1 d'abord, passer par le conseil
> (pyqt-dev + ux + redteam + architect → arbiter), commit thématique
> séparé, suite de tests verte avant push.
