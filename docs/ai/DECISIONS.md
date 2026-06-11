# DECISIONS — ARPES Explorer

Journal **append-only**. Une entrée = ce qui a changé + **pourquoi**, 3-5 lignes
max. Le *comment* est dans le code/git ; ici seulement le *quoi* et le *pourquoi*
non-déductibles. Plus récent en haut.

Historique détaillé pré-2026-06-06 archivé :
`docs/ai/archive/AUDIT_UPDATES_HISTORY.md` (ne pas charger sauf besoin précis).

---

## 2026-06-11 — Materials Project : raw mode + fallback chemins DFT
Import MP cassait avant conversion car `mp_api` validait des réponses sans
`equivalent_labels` via Pydantic. Fix : `MPRester(..., use_document_model=False)`
et helpers compatibles dict/objet ; bandstructure passe par route directe et
tente `setyawan_curtarolo`, `hinuma`, `latimer_munro`.
**Pourquoi** : MP peut changer son schéma serveur avant le client installé.
Pour `mp-568280`, MP répond maintenant que les 3 objets S3 bandstructure manquent.

## 2026-06-11 — Loader ALLS SpecsLab Prodigy ITX
Ajout backend `alls_itx` pour exports Igor Text SpecsLab Prodigy (`.itx`) :
détection stricte, parsing `WAVES/S/N` + `SetScale`, FS en `fs_data`
`(scan,kx,E)` et 2D affichée = moyenne scan. Axes `ShiftX a.u.`, `delay`,
`Loop` restent bruts/non calibrés ; scans temps sans `theta_y` refusés.
**Pourquoi** : importer les données ALLS sans logbook tout en évitant une fausse
conversion silencieuse vers momentum ou binding energy.

## 2026-06-11 — Packaging : exécutables Win/macOS/Linux via GitHub Actions
Demande : binaires 3 OS, robustes, patchables. Pas de cross-compilation
possible (PyInstaller) → CI = la seule voie robuste sans 3 machines. Livré :
`arpes.spec` (recette PyInstaller : hiddenimports scipy + pandas/openpyxl
lazy ; datas = arpes/docs (Help runtime) + arpes_plots.py ; **erlab EXCLU**
= backend optionnel gardé, en le bundlant on tirerait un arbre énorme),
`requirements.txt` (majors épinglés depuis l'env peaks : PyQt6 6.10, numpy
2.3, scipy 1.17, mpl 3.10, pandas, openpyxl, PyYAML), CI
`.github/workflows/build.yml` : tests headless à chaque push ; sur tag `v*`
→ 4 binaires (Linux x86_64 sur Ubuntu 22.04 pour glibc large, Win x86_64,
macOS arm64 + Intel) attachés à une GitHub Release. Patcher = committer puis
`git tag v1.0.x && git push origin v1.0.x` — tout est rebuildé tout seul.
macOS non signé → premier lancement clic-droit→Open (doc). dist/build
gitignorés. Spec validée par build local macOS réel.

## 2026-06-11 — Band Analysis : layout 2 colonnes, courbe TB persistée, branch lisible
Retour user après la passe tooltips : « toujours pareil » — les vrais problèmes
étaient (1) courbe TB INVISIBLE : `restore_all` rappelait `show_tb_result`
sans k/E/E_fit → plot vide après tout refresh/changement de fichier. Fix :
courbe persistée dans `ba["tb"]["curve"]` au fit, fallback dans le render.
(2) Canvases écrasés : les 3 tabs TB/Kink/Gap empilaient form+boutons+summary
AU-DESSUS du canvas → refonte 2 colonnes (rail params 340 px à gauche, canvas
toute la largeur restante). (3) Sélection de bande cryptique : combo branch
affiche « kF− (left branch, k<0) » mais garde la VALEUR `kF_minus` via
currentData (le texte sert de clé dans fit_result → prereq passé de findText
à findData, sinon autofill silencieusement cassé). (4) MDC Results : cap dur
350 px droite remplacé par QSplitter draggable (plots prennent tout l'écran,
défaut 1100/380). 868 OK / 9 skip, app réelle vérifiée.

## 2026-06-11 — Results compréhensibles : tooltips physiques + seuils honnêtes
Plainte : résultats « difficiles à comprendre et utiliser ». Conseil 4 voix,
plan 6 étapes indépendantes, livrées 1-5 (6 = polish marginal, skip) :
(1) tooltips sur CHAQUE colonne des 2 tables Results — 1 phrase physique
vérifiée dans le code (xg = offset commun du fit de paires ; Corr. Γ = après
correction résolution ; Γ₀ = intercept du fit FL Γ(E)=Γ₀+aE²…) ; χ²_red en
feu tricolore (≲1.5 vert / 1.5-4 orange / >4 rouge, seuils dans le tooltip).
(2) labels clairs : « Per-slice diagnostics » vs « Physical results — the
quantities to report ». (3) Band Analysis : Summary passé en 1er onglet (on
LIT la synthèse, on AGIT dans TB/Kink/Gap) ; chips ○ MDC→Gap cliquables
(naviguent vers le sous-onglet) + tooltips workflow Step 0-3. (4) interpré-
tation λ honnête (weak <0.3 / typical 0.3-1.5 / strong / unphysical <0) et
Δ vs résolution (« not resolved » si Δ<2×res — affiché, jamais silencieux),
dans les renders ET le Summary. n_Luttinger toujours NO-GO (backlog). 868 OK.

## 2026-06-10 — Results = hub d'analyse : Band Analysis déplacé, tables lisibles
Plainte : Results illisible + Band analysis caché dans MDC Fit. Conseil (ux+
architect+redteam+arbiter) : (A) GO — Results devient sous-onglets « MDC
Results » + « Band Analysis » (panneau déplacé ; placeholder « → Results tab »
laissé dans MDC Fit pour les habitudes, redteam R2). `_band_panel` reste créé
dans _build_mdc_tab (ordre de build : MDC avant Results) mais affiché dans
Results — signaux panels.py inchangés. (B) GO — tables 10→11px, headers gras,
lignes alternées, table physique stretch=2 vs per-slice 1 (le résultat prime
sur le diagnostic), boutons 5 empilés → 2 lignes. (C) NO-GO — colonne
n_Luttinger PAS calculée dans results.py ; l'ajouter = choisir une
dimensionnalité (kF²/π vs autre) → feature séparée, backlog. 865 OK / 9 skip.

## 2026-06-10 — Poches FS : barre d'action inline pour le preview
User : « ajuster Level dans le panneau + clic droit Validate = pas ergonomique »
(actions invisibles, aller-retour panneau↔canvas). Conseil (ux+architect+
redteam+arbiter) GO : barre inline sous le canvas FS, visible SEULEMENT
pendant un preview : `[Level slider+spin] [✓ Validate (MDC fit)] [✗ Cancel]`.
Slider live avec debounce 80 ms (sinon recompute contour à chaque tick) ;
range calibré sur min/max réels de la carte previewée (pas 0–1 figé) ; source
de vérité unique = sp_pocket_level du panneau (sync blockSignals). Cancel via
menu cache AUSSI la barre (état zombi sinon, redteam cas 1) ; échec MDC → la
barre RESTE (preview gardée). Molette souris pour level REFUSÉE (conflit
scroll-zoom mpl). Menu clic-droit conservé comme chemin alternatif. Code dans
fs_panel_pockets.py (free function, pattern lasso) — fs_panel quasi au cap.
864 OK / 9 skip.

## 2026-06-10 — Poches FS : flux unique, wizard SUPPRIMÉ, panneau dégraissé
Suite plainte « 4 portes d'entrée incompréhensibles ». Conseil (architect+ux+
redteam+arbiter, 1 spawn) : flux UNIQUE « lasso/clic → preview ISO (visuel) →
Validate = fit MDC radial (chiffres kF±σ) ». ISO = aperçu, MDC = mesure.
**Échec MDC = poche NON validée, preview gardée, message + auto-expand
Advanced — JAMAIS de fallback ISO silencieux** (un contour iso sans kF±σ
empoisonnerait Luttinger/bootstrap en aval, redteam cas 1). « Quick ISO (no
fit) » reste le chemin rapide EXPLICITE au menu. Wizard 3 pages SUPPRIMÉ
(fichier + verbe + signal, ~250 LOC). Menu clic-droit : 4 items au lieu de 6.
Panneau FS Pockets : visibles = count/Quality/Manual level/Level/n bands/
Spin/Export/Clear ; les 12 réglages algorithmiques → sous-groupe « Advanced
settings » replié (auto-déplié sur échec MDC, redteam cas 3). QThread pour le
MDC = DEFER (synchrone qq s, double-validate impossible sans thread). Test
manual_level adapté au nouveau contrat. 861 OK / 9 skip.

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

## 2026-06-11 — Import DFT Materials Project cassé = client mp-api périmé, pas le code

Symptôme : tout import MP échoue avec `No object found: s3://materialsproject-parsed/bandstructures/<id>.json.gz`, même mp-149 (Si). Cause : migration du stockage côté serveur MP ; le client mp-api 0.46.1 résolvait des clés S3 obsolètes. Fix : `micromamba run -n peaks pip install -U mp-api emmet-core` (0.46.3 / 0.87.0). Réflexe futur : si l'import MP casse d'un coup sans changement du code, tester `mp-149` en direct puis upgrader mp-api AVANT de chercher dans `arpes/theory/`.

## 2026-06-11 — Orientation BM↔FS par direction logbook + conventions labels ZDB

Décisions user : (1) direction logbook > azi moteur en cas de conflit (warning visible) ; (2) presets + renommage libre pour les labels HS ; (3) coupes verticales identifiées par la colonne direction. Implémentation data-driven : angles des directions depuis bz_high_symmetry_points (géométrie du panneau FS), remap labels unique dans bz.py (overlay + poches + matching direction cohérents). Bug de signe historique corrigé (fallback direction tournait fs−bm : Γ-M à 135° au lieu de 45°). Alias « Γ-Y » → X vertical sur zone carrée (usage logbook standard, documenté). Persistance par entrée : fs_bz_label_overrides/fs_bz_label_preset. Conseil GO (architect+redteam+arbiter) ; commit 827d745.
