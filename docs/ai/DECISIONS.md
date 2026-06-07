# DECISIONS — ARPES Explorer

Journal **append-only**. Une entrée = ce qui a changé + **pourquoi**, 3-5 lignes
max. Le *comment* est dans le code/git ; ici seulement le *quoi* et le *pourquoi*
non-déductibles. Plus récent en haut.

Historique détaillé pré-2026-06-06 archivé :
`docs/ai/archive/AUDIT_UPDATES_HISTORY.md` (ne pas charger sauf besoin précis).

---

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
