# DECISIONS — ARPES Explorer

Journal **append-only**. Une entrée = ce qui a changé + **pourquoi**, 3-5 lignes
max. Le *comment* est dans le code/git ; ici seulement le *quoi* et le *pourquoi*
non-déductibles. Plus récent en haut.

Historique détaillé pré-2026-06-06 archivé :
`docs/ai/archive/AUDIT_UPDATES_HISTORY.md` (ne pas charger sauf besoin précis).

---

## 2026-06-10 — Poches FS : lasso human-in-the-loop + wizard dégraissé
Plainte user : trop d'options. Conseil (architect+redteam+arbiter, 1 spawn) :
lasso rectangle = seul workflow viable (s'accroche au preview existant ; points
de contrainte/spline = refusé, pocket.py 670 LOC déboderait ; drag contour =
refonte). Livré : bouton toolbar « ▭ Pocket » → drag boîte autour d'UNE poche →
`physics/pocket_lasso.py` (pur numpy) dérive seed = centre boîte + level =
percentile(40) intérieur → enchaîne sur le preview existant (slider + clic
droit Validate). Gardes-fous redteam TOUS bruyants : sélection <16 px, zone
NaN, zéro contraste, pas de contour fermé au level (= 2 poches sélectionnées),
convexité >1.4 (= double poche, warning), contour au bord scan (suggère Arc,
PAS de switch auto). RectangleSelector recréé à chaque toggle (survit ax.cla()).
Wizard dégraissé : sigma_y/x, mdc_n_directions, mdc_r2_min retirés (auto/
internes — defaults panneau) ; restent ef_window, algo MDC/Iso, iso level,
mode Auto/Arc, orientation HS. Verbe « lasso » dans _pocket_action (0 entrée
PROXY_MAP). 861 OK / 9 skip.

## 2026-06-10 — SecDev/Curvature réparés (conseil physicist+numerics+arbiter)
Sur BNA réel : SecDev = bruit pur, Curvature = seuls les bords du masque
trapèze ressortent, bande lavée. 3 causes empilées trouvées : (1) formule
« Zhang 2D » INCOMPLÈTE — manquaient le terme `(C0+I_k²)·I_EE` ET le facteur
2 du terme croisé → n'affûtait que selon k ; (2) `C0 = 0.05·max(|∇I|)` : le
max = falaise du bord masque → C0 explose → bande/C0→0 ; (3) `sigma=2px` figé +
nan-median fill qui fait fuiter le fond dans la falaise. Fix : vraie formule 2D
complète ; `_smooth_masked` (convolution normalisée `Gauss(I·M)/Gauss(M)`,
NaN jamais propagés) ; C0 = α·percentile(|∇|,95) sur intérieur **érodé 5px**
(exclut la falaise) ; σ exposés en **unités physiques** (eV, π/a) → px via
median(diff(axe)) ; masquage à **EF+margin** (0.05 eV) au lieu de EF sec (coupe
sèche = artefact dérivée). UI : barre σ_E/σ_k/C0α visible si SecDev/Curvature
(C0α curvature-only), `_on_deriv_params_changed` (PROXY 146/150). Test
tautologique remplacé par référence indépendante + test « bande>bord » +
« C0 non lavé ». Copie morte `common.py::secdev_curvature` supprimée. DEFER
(backlog) : modes 1D-MDC et 1D-EDC curvature. 854 OK / 9 skip.

## 2026-06-10 — Viz production-grade : conseil 6 agents, hover/export/badge/cache
Demande « production-grade viz ». Conseil (architect/redteam/pyqt/numerics/ux/
arbiter) : la perf demandée EXISTAIT déjà (debounce 150 ms `_fs_redraw_timer`,
réuse mesh `set_array`+signature, NavToolbar FS, cmaps perceptually-uniform) —
vérifié avant d'agir. GO livrés : (1) hover readout (k, E/ky, I) via
`ax.format_coord` (toolbar mpl native, zéro throttle/signal — lit le VRAI array,
NaN → « — ») sur FS et BM ; (2) export toolbar PNG 300 dpi / SVG avec `draw()`
sync avant savefig ; (3) badge « Updating… » pendant le debounce (frame
périmée ≠ silencieuse) ; (4) fix cache : `id(array)` → digest contenu sparse
(collision GC possible = stale map servie, redteam §7). NO-GO : preview
sous-échantillonnée (stride aliase les poches fines ; goulot = intégration
numpy, pas pcolormesh) et cumsum O(1) (+784 MB RAM, gain nul). DEFER :
regroupement panneau (backlog). Crash pré-existant test_fs.py standalone
(canvas sans QApplication) fixé au passage. 849 OK / 9 skip.

## 2026-06-10 — Spinbox locale + rotation BM par direction + logbook b/c/φ
Trois choses livrées (2 commits). (1) **Locale** : sur OS FR, `QDoubleSpinBox`
attend la virgule et rejetait silencieusement le point (work function "4.5" →
"4"). Fix global `QLocale.setDefault(C)` dans `main()` (couvre tous les spinboxes
+ dialogs + futurs ; UI EN partout). `setLocale` par-helper gardés en filet.
(2) **Rotation BM** : quand `azi` moteur absent (BESSY), on tourne le cut depuis
le label direction du logbook (Γ-X/Γ-M…) au lieu de supposer 0° ; cut tagué
"rotated". (3) **Logbook** : mappe désormais b, c et work function (match exact +
flou + garde plausibilité numérique). `b_angstrom` ajouté à SampleConfig/FileMeta
(additif, pas de bump VERSION). Cuts dupliqués décalés perpendiculairement pour
rester lisibles. 844 OK / 9 skip.

## 2026-06-07 — FS↔BM CLS2026 : LE vrai bug = double key_for_path
Malgré les fixes découverte+φ, toujours 0 lien sur CLS2026. Cause racine :
`key_for_path` **n'est pas idempotent** sur clé nichée (`"BNA_S1/FS3"→"FS3"`).
`_active_fs_path` renvoie déjà une clé, puis `_bound_bms_for_active_fs` /
`_collect_bm_cuts_for_active_fs` refaisaient `key_for_path` dessus → `"FS3"` →
`files.get` = None → `[]`. Ba122 (plat) marchait car `key_for_path` idempotent
sur clé sans dossier. Fix : `_fs_entry_for_key` (lookup direct par clé, fallback
key_for_path), plus de double conversion. Aussi : la **liste linked-BMs** était
couplée aux cuts (φ-dépendants) → découplée via `refresh_matches` (montre les
matches sans φ ; l'overlay seul garde φ). Pas de bouton refresh : `build_pseudo`
tourne à chaque draw (live). Reste : régler φ pour dessiner les cuts.

## 2026-06-07 — MDC : fenêtre d'intégration ΔE (anti-serpentage)
kF(E) serpentait car chaque MDC = **une seule ligne d'énergie** (bruit max), la
méthode (fit séquentiel paires + seed prev_popt + rejet saut) étant par ailleurs
saine. Ajout `mdc_energy_window` (eV) à `fit_mdc_peak_pairs` : intègre ±window/2
en énergie (nanmean des lignes) avant le fit. Bruit ↓∝√N, **ne biaise ni kF ni
Γ** (varient lentement en E) — contrairement au lissage k qui gonfle Γ. 0 =
comportement actuel. Plumbé `FitParams.mdc_energy_window` → `fit.fit_kwargs` →
spinbox `sp_mdc_ewin` (params_fit). Distinct de `dE_eV` (résolution instrumentale
pour correction Γ).

## 2026-06-06 — Direction des cuts : normalisation + registre ZDB + filtre + azi
Constat : pour CLS la direction cristalline **n'est pas dans le raw** (sidecar a
polar/tilt/X/Y/Z, pas d'azimut) → vient du **logbook** (colonne `direction`).
Nouveau module pur `arpes/physics/hs_directions.py` :
- `normalize_direction_label` : raccourcis `GS→Γ-Σ`, `GX→Γ-X`, segments `XM→X-M`,
  variantes (`GtoX`, `Gamma-X`, `Γ→Σ`, espaces/slash). `_format_direction_label`
  (logbook) délègue dessus → forme canonique `A-B`. **S=Σ en entrée** (le coin
  rect `S` reste littéral dans le registre, non normalisé).
- `BZ_DIRECTIONS` registre par forme ZDB (square/rect/hex/centered_rect).
- `direction_from_azimuth` (P3, data-driven freezable) : ref None → UNCALIBRATED,
  **jamais inventer** (cf [[project_arpes_au_calibration]]) ; angles dérivés de
  `bz_high_symmetry_points`. UI de référence non construite (inerte pour CLS sans
  azi → backlog).
- Filtre direction dans le FS panel (`PairingCriteria.direction_filter`,
  `_filter_by_direction` normalise des 2 côtés → marche sur les "GS"/"SX" déjà
  stockés bruts). Combo branché sur `params_changed` (debouncé, pas le bug lag).

## 2026-06-06 — FS↔BM CLS2026 : 2 fixes (découverte + φ)
Diagnostic sur vraie session BaNi2As2-CLS2026 (logbook scopé par sous-dossier
BNA_S1/BNA_S2). Cause 1 : `build_pseudo_entries_from_logbook` scannait
`session.folder` à depth=1 → quand la session est ouverte au **parent** (data
nichée 2 niveaux), 0 pseudo-entry → aucune auto-découverte des BM non chargés.
Fix : descendre aussi dans les sous-dossiers scopés (0→6 pseudo-entries).
Cause 2 : `_collect_bm_cuts_for_active_fs` retournait `[]` **muet** si φ=0 (cas
fréquent sans logbook φ) → « Show BM cuts » semblait cassé. Fix : raise fort
→ le draw l'affiche en statusbar. (Le fix « dédup » envisagé était un faux
diagnostic : les doublons venaient d'un test à depth=1 croisant BNA_S1/S2.)
**Pourquoi** : l'appariement marchait sur Ba122 (dossier plat + logbook local)
mais pas CLS2026 (parent + scoped). Voir [[project_arpes_kz_bna_v0]].

## 2026-06-06 — KZ analyse : fit V0 (Lomb-Scargle) + profil I(kz)
Ajout `fit_inner_potential` (bouton « Fit V0 ») + `kz_profile_at_normal_emission`
(overlay I(kz)@k//0 + c via FFT). Méthode V0 = **Lomb-Scargle** de I(kz0(V0)) à
ω=c (période 2π/c) ; max = V0 où la modulation E_F est sinusoïdale pure.
Essayé d'abord paramètre d'ordre circulaire → **biaisé** (V0 haut groupe les
phases artificiellement, raile) ; puis concentration spectrale à fc → échoue.
LS récupère le V0 synthétique (5z ET 2z) avec max interne, et donne `power∈[0,1]`
= significativité. Confiance "low" si power<0.5 / railing / <1.5 zone → dans ce
cas **on n'écrase pas V0**, on dit pourquoi. Voir [[project_arpes_kz_bna_v0]].

## 2026-06-06 — Refonte onglet KZ
5 modes confus (`interpolated/binned/points/hv map/MDC waterfall`, mélangeaient
système de coordonnées × style de rendu) → **2 vues** (`Raw hν` · `kz`) +
overlays cases à cocher (`sample points`, `plans Γ/Z`). Supprimé `binned`,
`points`, `MDC waterfall` (`compute_mdc_waterfall`/`MdcWaterfallResult` retirés ;
`compute_kz_map` interpolé seul). Ajouté l'aide calibration V0 : lignes Γ/Z
(`kz_high_symmetry_planes`) + readout périodicité/hν (`kz_coverage_summary`,
`hv_for_kz`) ; autofill a/c depuis `sample_for_entry` ; fallback Raw si a/c
manquants. Conseil ARPES fait inline.
**Pourquoi** : l'onglet ne servait à rien (pas d'aide pour trouver V0, but réel
d'un scan kz) ; `fold_kz_to_1bz` physique existait mais n'était pas branché.

**Bug racine trouvé en testant sur vraies données (BNA S1/PS1·PS2, S2/PS1)** :
`load_kz_stack` ne passait pas `a_lattice` à `_load_cls_photon_scan_folder` →
défaut 0 → `_cls_angle_to_k_pi_over_a` renvoyait k//≡0 → nuage (k//,kz)
colinéaire → `compute_kz_map` plantait (`QhullError`). Fix : thread `a_lattice`
(load + controller via `_lattice_a` depuis l'échantillon) + garde anti-crash
(nuage dégénéré → carte binnée, flag `degenerate_kpar`, warn). Les 3 dossiers
rendent désormais une vraie carte kz (Γ@hν≈47,64,83,104 · Z@hν≈55,73,93 eV).

## 2026-06-06 — Traduction app FR→EN terminée (Codex)
Tout l'app en anglais (code/UI/commentaires/docstrings/help in-app). Listes
mots-clés de matching logbook **gardées bilingues** (`logbook*.py` : clés FR+EN
type `["energie","ev"]`, `"polarisation"`, `"dossier"` = clés de matching, pas
du texte UI). 189 fichiers, suite 816 OK / 9 skip sous `peaks`.
**Pourquoi** : app destinée à publication anglophone multi-labo ; les clés
logbook restent FR car les vrais logbooks sont en français.

## 2026-06-06 — Refonte système docs AI
Racine encombrée de `*_PLAN.md`/`*_TODO.txt` (un par feature, jamais nettoyés) +
historique 44KB. Adopté « 4 fichiers chauds + archive froide » : `CLAUDE.md`
(règles+index), `docs/ai/{BACKLOG,DECISIONS,COUNCIL}.md`, `docs/ai/archive/`.
**Pourquoi** : 1 fichier par rôle (pas par feature), seul `CLAUDE.md` auto-chargé
→ moins de tokens, plus de fiabilité. Vieux audits → archive (git garde tout).

## 2026-06-05 — P3 résiduel (P3.4 / P3.6 / P3.7)
`FitZone` dataclass + normalize au load (loud sur clé inconnue, tue pertes
silencieuses) ; controllers instanciés dans `_install_controllers()` ; garde
ré-entrance `_fit_busy`. Détail : `archive/AUDIT_UPDATES_HISTORY.md`.
