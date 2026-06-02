# Plan fixes detection poches FS

Date : 2026-06-02

Objectif : fiabiliser la detection/caracterisation des poches FS apres audit. Priorite aux bugs qui peuvent produire une poche fausse, mal placee, ou mal comparee a la DFT.

## Diagnostic court

Le workflow actuel marche pour le cas simple `kx_center=0`, `ky_center=0`, contour ferme bien dans la carte, BZ manuelle non tournee. Les risques apparaissent surtout quand :

- le centre Gamma est non nul ;
- une poche touche le bord de la carte ;
- on utilise l'overlay cristal/MP ;
- on compare a une grille DFT ;
- on utilise le chemin MDC radial au lieu de l'iso-contour.

## Convention cible

Adopter une seule convention interne :

- Toutes les donnees stockees en session doivent rester en coordonnees raw FS, donc meme repere que `kx`, `ky`, `extract_fs_map`, DFT, BZ raw.
- Le rendu canvas applique seul le shift display : `display = raw - (kx_center, ky_center)`.
- Les champs `centroid_kx`, `centroid_ky`, `contour`, `hs_distance`, `dft_compare` restent raw.
- Les exports CSV restent raw, avec colonnes supplementaires optionnelles `display_centroid_kx`, `display_centroid_ky` seulement si utile.

Cette convention evite les doubles corrections et rend les comparaisons physiques coherentes.

## Fix P0

### 1. Ne plus accepter les contours ouverts comme poches fermees

Fichiers :

- `arpes/physics/pocket.py`
- `tests/test_pocket.py`

Probleme : `extract_fs_contour()` ferme chaque contour avec `_close_contour()` avant de verifier qu'il etait deja ferme. Un arc ouvert qui sort du bord peut devenir un faux contour ferme.

Implementation :

1. Recuperer les lignes brutes de `contourpy`.
2. Verifier fermeture naturelle avant `_close_contour()` :
   - `raw.shape[0] >= 4`
   - `norm(raw[0] - raw[-1]) <= tol`
3. Fermer seulement apres cette verification.
4. Ajouter test avec iso-ligne qui traverse le bord de map et doit lever `ValueError`.

Critere OK :

- `extract_fs_contour()` rejette une courbe ouverte.
- Les tests existants de contour ferme restent verts.

### 2. Corriger les coordonnees stockage/rendu

Fichiers :

- `arpes/ui/controllers/pocket_controller.py`
- `arpes/ui/controllers/pocket_controller_mdc.py`
- `arpes/ui/widgets/fs_panel_pockets.py`
- `tests/test_pocket_controller.py`

Probleme : contour stocke en display coords, centroid en raw coords. Apres changement Gamma, contour et label peuvent diverger.

Implementation :

1. `_contour_for_storage()` retourne contour raw, sans soustraire `kx_center/ky_center`.
2. Chemin MDC radial stocke `contour_raw` aussi en raw.
3. `draw_pockets()` soustrait `canvas._bm_cut_center` pour contour et centroid au rendu.
4. Ajouter compat lecture anciens fichiers :
   - si `pocket["coord_frame"]` absent, traiter comme ancien format display ;
   - pour nouveau format, stocker `pocket["coord_frame"] = "raw"`.
5. Ne pas migrer toute la session automatiquement au premier patch ; garder fallback local dans renderer/export.

Critere OK :

- Avec `kx_center != 0`, contour et label restent alignes.
- Changer le centre Gamma puis redessiner ne decale pas contour et label de facon differente.

### 3. Corriger preview double-shift

Fichiers :

- `arpes/ui/controllers/pocket_controller.py`
- `arpes/ui/widgets/fs_panel_pockets.py`
- `tests/test_pocket_controller.py`

Probleme : `_draw_preview_at()` shift le contour, puis `draw_pocket_preview()` shift encore.

Implementation :

- Option recommandee : preview passe contour raw au canvas.
- `draw_pocket_preview()` applique le shift display.
- Ajouter test nonzero center qui inspecte les offsets du `PathCollection`.

Critere OK :

- Apercu et poche validee tombent au meme endroit visuel.

### 4. Corriger DFT compare

Fichiers :

- `arpes/ui/controllers/pocket_controller.py`
- `tests/test_pocket_controller.py`

Probleme : `compare_pocket_contours()` exige deux contours dans le meme repere, mais le contour experimental est actuellement stocke en display coords.

Implementation :

- Apres fix stockage raw, `_attach_dft_compare()` lit directement `pocket["contour"]`.
- Pendant compat ancien format, convertir ancien contour display vers raw avant comparaison :
  `contour_exp_raw = contour + (kx_center, ky_center)` si `coord_frame` absent/display.
- Ajouter test DFT compare avec centre Gamma non nul.

Critere OK :

- DFT compare ne depend pas du recentrage display.

## Fix P1

### 5. Appliquer `min_area_pct_bz` au chemin MDC radial

Fichiers :

- `arpes/ui/controllers/pocket_controller_mdc.py`
- `tests/test_pocket_controller.py`

Probleme : iso-contour rejette les petites poches, MDC radial les sauvegarde sans appliquer `min_area_pct_bz`.

Implementation :

- Si `closed=True`, rejeter quand `area_pct_bz < min_area_pct_bz`.
- Si `closed=False` arc mode, ne pas appliquer l'aire BZ ; afficher aire `NaN`.

Critere OK :

- Petite poche MDC fermee rejetee.
- Arc MDC reste possible sans faux Luttinger count.

### 6. Clarifier labels HS : BZ manuelle vs BZ cristal MP

Fichiers :

- `arpes/ui/controllers/pocket_controller.py`
- `arpes/ui/widgets/dialogs/pocket_result.py`
- tests a definir.

Probleme : `hs_label_nearest` vient de la BZ manuelle, alors que l'utilisateur peut regarder l'overlay cristal MP.

Implementation minimale :

- Renommer/ajouter champs :
  - `hs_label_nearest_manual`
  - `hs_label_nearest_crystal`
  - garder `hs_label_nearest` comme label principal pour compat.
- Si overlay cristal actif et lattice disponible, calculer label cristal depuis `project_hs_points()` avec meme `plane`, `phi_c`, Gamma.
- Dans dialog/export, indiquer la source du label : `manual_bz` ou `mp_crystal`.

Critere OK :

- Le label sauvegarde ne contredit plus silencieusement le label visible.

### 7. Deriver directions Gamma-X / Gamma-M depuis BZ active

Fichiers :

- `arpes/ui/controllers/pocket_controller.py`
- `arpes/ui/widgets/fs_panel.py`
- tests pure/unit.

Probleme : `kF_gamma_x` et `kF_gamma_m` utilisent par defaut `0 deg` et `45 deg`, meme si BZ tournee/hexagonale/cristal.

Implementation :

- Garder overrides UI manuels.
- Ajouter mode `Auto` :
  - BZ carree/rectangle : X depuis point X le plus proche dans direction +kx, M depuis diagonale la plus proche.
  - BZ cristal : directions depuis points HS projetes.
- Sauvegarder `hs_dir_source`.

Critere OK :

- Pour BZ tournee, `kF_gamma_x/m` suivent la BZ visible.

## Recommandations physiques P2

### 8. Topologie electron/hole

Risque : `pocket_topology()` classe electron/hole par intensite inside/outside. Ce n'est pas une vraie mesure de dispersion.

Recommandation :

- Renommer dans UI : `topologie_intensite` ou afficher warning.
- Garder `electron/hole/unclear`, mais ajouter `topology_method = "inside_outside_intensity"`.
- Ne pas utiliser ce champ seul pour conclusion physique.
- Pour publication : preferer validation par BM/MDC dispersion ou comparaison DFT.

### 9. Iso-contour vs MDC radial

Recommandation :

- Iso-contour = rapide, bon pour poches fermees propres.
- MDC radial = mode publication si signal anisotrope/bruite ou intensite inhomogene.
- Ajouter dans dialog une ligne `algo` + `quality_checks`.

### 10. Arcs et bords de carte

Recommandation :

- Une poche qui touche le bord ne doit pas donner aire/Luttinger.
- Pour arcs, stocker :
  - `closed=False`
  - `arc_coverage_deg`
  - `area_pct_bz=NaN`
  - `n_carriers_2D=NaN`

## Tests obligatoires avant merge

Commandes :

```bash
python3 -m pytest tests/test_pocket.py tests/test_pocket_controller.py tests/test_pocket_mdc_radial.py -q
python3 -m pytest tests/ --ignore=tests/test_annotations.py --ignore=tests/test_local_dft_loaders.py -q
```

Cas a ajouter :

- `extract_fs_contour` rejette contour ouvert au bord.
- Preview nonzero Gamma : pas de double-shift.
- Poche iso nonzero Gamma : contour et centroid alignes au rendu.
- Redraw apres changement Gamma : ancien contour raw reste coherent.
- DFT compare nonzero Gamma : contours dans meme frame.
- MDC radial applique `min_area_pct_bz` sur poche fermee.
- `_hs_points_raw()` avec BZ carree : 4 X + 4 M conserves, centre non nul inclus.
- Bootstrap controller : persiste `uncertainty`, `quality_checks`, `coord_frame`.

## Ordre d'implementation recommande

1. `extract_fs_contour` fermeture naturelle + test.
2. `coord_frame="raw"` pour iso + renderer compat.
3. Preview raw + test nonzero center.
4. MDC radial raw + min area.
5. DFT compare raw/legacy compat.
6. Labels HS cristal et directions auto.
7. UI wording pour topologie et algo.

## Definition of done

- Plus aucun contour ouvert artificiellement ferme.
- Toutes les poches nouvelles stockent `coord_frame="raw"`.
- Les anciennes poches restent affichables.
- Preview et poche validee superposent visuellement.
- DFT compare utilise le meme repere que la poche experimentale.
- MDC radial respecte les memes garde-fous que l'iso-contour.
- Tests cibles + suite filtree passent.
