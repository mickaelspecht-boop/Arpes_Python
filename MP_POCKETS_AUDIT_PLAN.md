# Audit MP + caractérisation poches FS — plan d'update

Date : 2026-06-01. Issu d'une revue 2-agents (architect code + physicist physique)
sur l'intégration Materials Project (MP) actuelle et la caractérisation des
poches de Fermi surface. User signale "ça ne marche pas du tout".

---

## Synthèse de l'audit

### Ce qui existe (et marche)

| Composant | Fichier | État |
|---|---|---|
| Récup lattice MP (a, b, c, bravais, space_group) | `theory/materials_project.py:134 load_lattice` | OK, cache disque |
| Récup bandes DFT 1D le long path Setyawan-Curtarolo | `theory/materials_project.py:30 load_materials_project_band_data` | OK |
| Recherche par formule + dialog | `theory/materials_project.py:89 search_by_formula` + `dialogs/mp_search.py` | OK |
| Projection caractère orbital (opt-in) | idem :37 `with_projections` | OK |
| Overlay bandes DFT sur BM (k, E) | `theory/plot.py` | OK |
| Comparaison fit kF ↔ DFT + fit μ-shift | `theory/comparison.py`, `models.py` | OK |
| Self-energy Re Σ(E) = E_exp − E_DFT | `analysis/self_energy.py` | OK |
| BZ 2D + points haute symétrie (HS) | `physics/bz.py:206` | OK, Voronoi + fallback half-plane |
| Rotation cristal/détecteur (φ_c) + projection HS | `physics/bz_overlay.py:38 project_hs_points` | OK |
| Free electron final state model (kz from hν) | `physics/kz.py` | OK |
| Overlay polygone BZ cristal sur FS | `ui/widgets/fs_panel.py:530 _overlay_bz_crystal` | OK partiel (cf bugs) |
| Auto-fetch theory au load depuis logbook | `load_controller.py:609` | OK |
| Restauration theory_overlay au switch fichier | `load_controller.py:503` | OK |
| DFT locale (vasprun, QE, yaml, json) | `theory/local_loaders.py` | OK |

### Ce qui est cassé (bugs UI bloquants)

1. **Bouton « Récup symétrie MP » fantôme** — `fs_panel.py:184` émet
   `mp_lattice_fetch_requested` mais aucune connexion dans `panels.py` ne
   route vers `_on_mp_lattice_fetch`. Le bouton est cliquable mais ne fait
   rien. **Bug H, fix 5 lignes.**
2. **Settings BZ cristal jamais restaurés au switch fichier** —
   `_restore_fs_crystal_settings_from_entry` (`fs_controller.py:453`) défini
   mais aucune callsite. Toggle BZ xtal, V0, φ_c, mp_id se perdent en
   mémoire entre fichiers (alors qu'ils sont persistés dans la session
   JSON via `fs_bz_crystal_visible`, `fs_v0`, `fs_kz_plane`, `fs_phi_c_deg`).
   **Bug H, fix 1 ligne.**
3. **Fallback heuristique tétragonal silencieux** — `_overlay_bz_crystal`
   utilise `c=10.0` si `entry.fs_lattice` vide. L'utilisateur voit un
   polygone BZ et croit que c'est la BZ MP, alors que c'est une heuristique
   fabriquée. **Bug H, fix : warning explicite + désactivation overlay si
   pas de mp_id valide.**

### Ce qui n'existe pas du tout

- **Détection de poche** sur la FS expérimentale (extraction contour fermé
  à I = I_EF par seuillage / contourpy).
- **Classification électron vs trou** par signe de la dispersion locale
  (∂E/∂k|kF perpendiculaire au contour).
- **Comptage de poches** (blob detection + filtre topologique).
- **Labellisation α / β / γ / δ** ou assignation au point HS le plus proche
  (Γ, X, M, Y, S…).
- **Calcul aire** en %BZ (Luttinger), forme (fit ellipse → kF_a, kF_b,
  anisotropie, angle), masse effective m*.
- **Iso-contours DFT projetés** sur la FS expérimentale. Le module
  `theory/fs_isocontour.py` existe (extracteur 3D → 2D propre) mais **n'est
  jamais appelé**. Cause racine : MP ne fournit qu'une bandstructure 1D
  (chemin Setyawan-Curtarolo), pas de grille 3D `E_n(kx, ky, kz)` requise
  par `fs_isocontour`. **Bloqueur architectural pour cette feature via
  MP seul.**

### Modules orphelins

- `theory/fs_isocontour.py` — extracteur propre, jamais wiré. Bloqué côté
  données (cf supra).
- `physics/bz_overlay.py:88 fit_phi_c_from_clicks` — fit LSQ de φ_c + Γ
  depuis clics user sur HS. Code propre, **aucune UI pour cliquer**.

### Dette technique tracée

- `theory/theory_overlay_controller.py` 491 LOC mélange 6 sujets (import
  MP, import local, picker, alignement, μ-fit, self-energy, search,
  restore). Candidat à split.
- `ui/widgets/fs_panel.py` 605 LOC, zone jaune (proche plafond 700).
- `materials_project.py:_get_bandstructure` / `_get_structure` ont
  plusieurs forks de signature mp-api → fragile aux versions.
- Aucune vérification de cohérence `FSParams.a_lattice` (UI) ↔
  `fs_lattice["a"]` (MP) au moment du draw : warning seulement au fetch.
- Tests : 0 test UI smoke sur `_on_mp_lattice_fetch`, 0 test sur fallback
  heuristique BZ overlay, 0 test wiring `bz_crystal_overlay_changed`.
  `test_fs_isocontour` pur numérique sans branchement app.

---

## Pourquoi "ça ne marche pas du tout"

Trois causes additives :

1. **Bouton « Récup symétrie MP » fantôme** : l'utilisateur clique, rien ne
   se passe. Mauvaise première impression. Bug d'1 connexion manquante.
2. **Settings BZ xtal perdus au switch fichier** : tu actives le BZ
   overlay, tu changes de fichier, il disparait. Bug d'1 appel manquant.
3. **Pipeline pocket inexistant** : aucun outil pour réellement
   « caractériser une poche ». Tout ce qui existe c'est de l'overlay
   géométrique (BZ + points HS) et de l'overlay bandes 1D sur BM. Le pas
   physique majeur (extraction contour FS, aire, topologie e/h, label HS)
   n'a jamais été codé.

Le 1) et 2) sont triviaux à fixer. Le 3) demande une vraie phase de
développement.

---

## Phases d'update

### Phase 0 — Fix bugs UI bloquants (≤ 30 min)

Objectif : que ce qui existe déjà fonctionne réellement.

1. **Wirer `mp_lattice_fetch_requested`** dans `panels.py` vers
   `_on_mp_lattice_fetch` (analogue au pattern `bz_crystal_overlay_changed`).
2. **Appeler `_restore_fs_crystal_settings_from_entry`** dans
   `load_controller._apply_session_after_load`, juste après le bloc
   `_restore_theory_overlay_for_entry` existant.
3. **Garde explicite dans `_overlay_bz_crystal`** : si `entry.fs_lattice`
   vide ou `mp_id` absent → status `« Pas de lattice MP, overlay BZ désactivé.
   Récupère-la avec le bouton 'Récup symétrie MP'. »` + ne dessine rien
   plutôt que dessiner un polygone fantôme.

**Fichiers** : `panels.py`, `load_controller.py`, `fs_panel.py`.
**Tests** : ajouter 2 cas dans `test_ui_smoke.py` (wiring fetch + restore
crystal settings).

### Phase 1 — Module `physics/pocket.py` autonome (~1 jour)

Objectif : caractériser une poche **depuis ARPES seule**, sans dépendance
DFT. Indépendant de MP.

Nouveau module pur, headless, testable :

```python
# arpes/physics/pocket.py (~200 LOC)

@dataclass(frozen=True)
class PocketProperties:
    centroid_kx: float
    centroid_ky: float
    area_inv_a2: float            # aire en (π/a)²
    area_pct_bz: float            # % aire BZ 2D
    kF_mean: float                # rayon moyen
    kF_a: float                   # demi-axe ellipse
    kF_b: float
    ellipse_angle_deg: float
    topology: Literal["electron", "hole", "unclear"]
    topology_confidence: float    # 0..1
    hs_label_nearest: str         # "Γ", "X", "M", ...
    hs_distance: float            # à HS le plus proche

def extract_fs_contour(
    image: np.ndarray, kx: np.ndarray, ky: np.ndarray,
    level: float,                 # seuil iso-intensité
    seed_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Retourne (N, 2) points du contour fermé contenant seed_point.
    Utilise contourpy.contour_generator. Si seed_point None → tous contours.
    """

def pocket_area(contour: np.ndarray) -> float:
    """Aire signée via formule du lacet (Shoelace)."""

def pocket_topology(
    image: np.ndarray, kx: np.ndarray, ky: np.ndarray,
    contour: np.ndarray, n_rays: int = 8,
) -> tuple[Literal["electron", "hole", "unclear"], float]:
    """Tire N rayons depuis centroïde vers extérieur, compare intensité
    intérieure vs extérieure. Si I_inside > I_outside → électron.
    Confidence = consistance des N rayons."""

def fit_pocket_ellipse(contour: np.ndarray) -> tuple[float, float, float]:
    """PCA sur points contour → (kF_a, kF_b, angle_deg)."""

def assign_hs_label(
    centroid: tuple[float, float],
    hs_points: dict[str, tuple[float, float]],
) -> tuple[str, float]:
    """Label HS le plus proche + distance."""

def characterize_pocket(
    image, kx, ky, *,
    seed_point, level,
    bz_polygon, hs_points,
) -> PocketProperties:
    """Pipeline complet : contour → aire → topologie → ellipse → label HS."""
```

**Tests** : `tests/test_pocket.py` (~12 cas) — cercle synthétique e/h, ellipse,
mauvais seed, level trop bas, label HS proche/loin.

### Phase 2 — UI caractérisation poche (~1 jour)

Objectif : rendre `characterize_pocket` accessible à l'utilisateur.

1. **Click-droit sur la FS** → menu contextuel `« Caractériser poche ici »`.
2. Le click fournit `seed_point = (event.xdata, event.ydata)`.
3. Default `level = 0.5` ou détecté auto par dérivée locale (knee point sur
   histogramme MDC à E=EF).
4. Appel `characterize_pocket(...)` → `PocketProperties`.
5. **Dialog résultat** affiche : label HS, aire en %BZ, kF moyen + kF_a/kF_b,
   topologie (icône e⁻ ou h⁺), confidence, suggestion `mp_label`.
6. Persistance : `entry.fs_pockets: list[PocketProperties.asdict()]` (nouveau
   champ FileEntry, défaut `[]`).
7. Re-overlay des poches caractérisées sur la FS au prochain redraw (contours
   colorés + label flottant).

**Fichiers** :
- `arpes/ui/dialogs/pocket_result_dialog.py` (nouveau).
- `arpes/ui/controllers/pocket_controller.py` (nouveau, verb-dispatch
  `_pocket_action(verb, payload)`).
- `arpes/core/session.py` : ajout `FileEntry.fs_pockets`.
- `arpes/ui/widgets/fs_panel.py` : wire context menu.

**PROXY_MAP** : +1 entrée (`_pocket_action`). Reste 149/150.

**Tests** : `tests/test_pocket_controller.py` (~6 cas), `tests/test_session.py`
roundtrip `fs_pockets`.

### Phase 3 — Garde-fous cohérence MP ↔ ARPES (~30 min)

1. **Refuser overlay BZ MP** si `|a_MP − a_ARPES| / a_MP > 2 %`. Status
   message explicite + bouton « Forcer override » dans dialog (pour cas
   conscients).
2. **Vérifier bravais détecté** : si carte FS révèle symétrie C4 (FFT
   ring + 4 pics) mais MP retourne hexagonal → warning rouge.
3. **Document** : ajouter au tooltip du bouton MP la note « DFT GGA
   sous-estime m*, position bandes peu fiable, mais aire FS (Luttinger)
   est conservée ».

### Phase 4 — Activer fs_isocontour.py orphelin (optionnel, plus tard)

`fs_isocontour.py` requiert une grille 3D `E_n(kx, ky, kz)` que MP ne
fournit pas. Pour le rendre utile :

1. **Loader bs_uniform local** : nouveau module
   `theory/bs_uniform_loader.py` qui ingère un vasprun.xml / wannier90 /
   CASTEP en 3D uniforme.
2. UI séparée : bouton « Charger DFT 3D locale » distinct de « MP ».
3. Appel `fs_isocontour.iso_contour_at_ef(grid_3d, kz_plane)` → contour.
4. Overlay sur FS expérimentale en couleur DFT distincte.

**Ne pas étiqueter cette feature « MP »** — c'est strictement DFT locale,
MP n'alimente pas cette voie.

### Phase 5 — Calibration V0 depuis hν-scan (optionnel, nice-to-have)

Bouton « Fit V0 from hν-scan » qui :
1. Charge tous les scans kz disponibles (différents hν).
2. Utilise `compute_kz_map` pour calculer kz par scan.
3. Détecte la périodicité observée en kz (FFT sur intensité intégrée).
4. Compare à 2π/c de la lattice MP.
5. Ajuste V0 pour matcher.

Permet de calibrer V0 expérimentalement plutôt que de prendre le défaut
arbitraire 12 eV.

---

## Anti-features (à NE PAS faire)

- **Overlay FS DFT directement depuis MP** : architecturalement impossible.
  MP fournit `BandStructureSymmLine` (chemin 1D), pas de grille 3D
  uniforme. Tenter de l'interpoler depuis le chemin 1D donnerait des
  résultats faux. Si tu veux ça, il faut une DFT 3D recalculée localement
  (Phase 4).
- **Détecter symétrie cristalline depuis la FS automatiquement et
  écraser le `bravais` MP** : trop risqué, surface peut être
  reconstruite, twin domains, désordre. Toujours laisser l'utilisateur
  trancher.
- **Promettre une caractérisation pocket "exacte"** : c'est toujours une
  approximation (contour à un seuil arbitraire, dispersion locale
  estimée). Document les limites dans les tooltips.

---

## Ordre d'exécution recommandé

1. **Phase 0** maintenant (30 min, débloque ce qui existe).
2. **Phase 3** dans la foulée (30 min, ajoute garde-fous, fiabilise
   l'usage des features existantes).
3. **Phase 1 + Phase 2** ensemble (~2 jours, la vraie nouvelle feature
   pocket characterization).
4. **Phase 4 et 5** plus tard si tu travailles régulièrement avec DFT
   locale ou si tu fais des scans kz systématiques.

---

## Estimation totale

| Phase | Effort | LOC | Risque | Bénéfice |
|---|---|---|---|---|
| 0 — Fix bugs UI | 30 min | ~20 modif | très faible | débloque l'existant |
| 1 — `physics/pocket.py` | ~6 h | ~250 nouveau + 12 tests | faible (module pur) | feature core |
| 2 — UI caractérisation | ~6 h | ~200 nouveau + 6 tests | moyen (UI Qt) | usage user direct |
| 3 — Garde-fous | 30 min | ~30 modif | très faible | évite faux positifs |
| 4 — DFT locale 3D | ~1 jour | ~300 nouveau | moyen | optionnel |
| 5 — Calib V0 | ~4 h | ~150 nouveau | moyen | optionnel |

**Total minimum viable (P0+P1+P2+P3)** : ~13 h, ~500 LOC nouveau, 18
tests. Couvre tout ce que l'utilisateur attend en lisant « outil pour
caractériser les poches sur la FS ».

---

## Définition de done (P0+P1+P2+P3)

- Bouton « Récup symétrie MP » charge la lattice et l'affiche dans le
  status bar.
- Switch entre fichiers conserve l'état BZ overlay + V0 + φ_c + mp_id.
- Click-droit sur une poche FS → caractérisation complète en < 2 s.
- Aire, topologie e/h, label HS suggéré affichés dans un dialog.
- Poche caractérisée persiste en session JSON et se réaffiche au reload.
- Suite tests verte (+18 cas).
- Aucune régression sur les 553 tests actuels.
