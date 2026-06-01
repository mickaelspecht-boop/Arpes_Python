# Plan d'update — organisation BM ↔ FS + overlay BM cuts

Date : 2026-06-01. Issu de la discussion post-audit Γ. Doit être exécutable
même si je n'ai plus de tokens : tout le contexte nécessaire est ici.

## Décisions actées (user)

- **Q1 (modèle de relation)** : **M4 hybride** — auto-discovery par
  métadonnées + override manuel via `entry.parent_fs_path`.
- **Q2 (critères de compatibilité)** : les dossiers chargés sont regroupés
  par échantillon. Pas d'opinion forte sur les autres paramètres → suivre la
  **recommandation par défaut** : `même dossier (ou parent commun) + hv ±5 % +
  azi ±2° + polarization identique`.
- **Q3 (multi-actif)** : recommandation suivie → `_current_fs_path` +
  `_current_bm_path` indépendants, `_current_path` devient une property
  rétrocompatible retournant le dernier modifié.
- **Q4 (file browser)** : **O3 minimaliste** — arbre par dataset (FS parent
  + BMs enfants), sans complexifier l'app. Ne pas réinventer le file browser.

## Périmètre — ce qui change

Trois phases A → B → C. **Phase A est le prérequis** pour B. Phase C est
optionnelle, à n'attaquer qu'après usage réel de A+B.

---

## Phase A — Organisation minimale (~1 à 1,5 jour)

Objectif : permettre à l'app de raisonner sur des paires (FS, [BMs]) sans
casser le flow actuel.

### A.1 — Fiabilité du `scan_kind` (~30 min)

Aujourd'hui `entry.meta.scan_kind` est défini par les loaders (CLS, Bessy)
mais pas universellement. Audit nécessaire :

```bash
grep -rn "scan_kind\|fs_data\b" arpes/io/loaders/ arpes/io/loader_orchestrator.py
```

Garantir que chaque entry après load a :
- `entry.meta.scan_kind ∈ {"BM", "FS", "KZ", "EDC", "other"}`
- Soit déduit du loader, soit du contenu (`metadata.fs_data is not None →
  "FS"`, `metadata.kz_scan is not None → "KZ"`, sinon `"BM"` par défaut).

Si le champ n'existe pas sur `FileMeta`, l'ajouter (déjà présent
probablement — vérifier `arpes/core/session.py:FileMeta`).

**Fichiers touchés** : `arpes/core/session.py` (peut-être), `arpes/io/loaders/*.py`,
ou un helper `arpes/io/scan_kind_inference.py` (nouveau, ~30 LOC).

**Tests** : `tests/test_scan_kind_inference.py` (nouveau, ~5 cas).

### A.2 — Auto-discovery (filtre M1) — module pur (~2 h)

Nouveau module `arpes/io/file_pairing.py` (~120 LOC, pur, sans Qt) :

```python
@dataclass(frozen=True)
class PairingCriteria:
    same_folder: bool = True            # ou parent commun à profondeur N
    folder_depth: int = 1               # 0 = même dossier strict, 1 = parent commun
    hv_tolerance_rel: float = 0.05      # ±5%
    azi_tolerance_deg: float = 2.0
    require_polarization: bool = True
    require_sample: bool = False        # opt-in via sample_name/formula

def find_bms_for_fs(
    fs_entry: FileEntry,
    fs_path: str,
    all_files: dict[str, FileEntry],
    criteria: PairingCriteria | None = None,
) -> list[tuple[str, FileEntry, str]]:
    """Retourne [(path, entry, reason)] pour les BMs compatibles.

    `reason` = "auto" ou "manual" (si entry.parent_fs_path == fs_path).
    Triées : manuels d'abord, puis auto par proximité (Δazi, Δhv).
    """

def find_fs_for_bm(
    bm_entry: FileEntry,
    bm_path: str,
    all_files: dict[str, FileEntry],
    criteria: PairingCriteria | None = None,
) -> list[tuple[str, FileEntry, str]]:
    """Symétrique : retourne FS compatibles pour une BM donnée."""
```

**Logique** :
1. Si `bm_entry.parent_fs_path` existe et matche `fs_path` → tag "manual",
   retourne en tête.
2. Sinon, filtres séquentiels : `scan_kind == "BM"` + `same_folder` + `|Δhv|
   ≤ tol * hv_fs` + `|Δazi| ≤ tol_azi` + (optionnel) polar / sample.
3. Tri par distance composite : `sqrt((Δazi/tol)² + (Δhv/(tol·hv))²)`.

**Tests** : `tests/test_file_pairing.py` (~10 cas, headless).

### A.3 — Champ `entry.parent_fs_path` (M4 override) (~15 min)

Ajout dans `arpes/core/session.py:FileEntry` :

```python
# Override manuel pour pairing BM↔FS (cf BM_FS_ORGANIZATION_PLAN.md).
# Si défini : force le rattachement de cette BM à la FS au path donné,
# court-circuitant l'auto-discovery par métadonnées.
parent_fs_path: Optional[str] = None
```

Roundtrip JSON dans `load_from_payload` :

```python
parent_fs_path=edict.get("parent_fs_path"),
```

**Pas de bump VERSION** : ancien JSON sans le champ → None par défaut, OK.

**Tests** : `tests/test_session.py` étendre roundtrip pour couvrir le nouveau
champ (~1 cas).

### A.4 — Multi-actif `_current_fs_path` + `_current_bm_path` (~3 h)

Le plus délicat. Refactor `ArpesExplorer` (`arpes/app.py`) :

**Étape 1** : ajouter les deux variables.

```python
self._current_fs_path: Optional[str] = None
self._current_bm_path: Optional[str] = None
self._last_modified_path: Optional[str] = None  # tracker pour rétrocompat

@property
def _current_path(self) -> Optional[str]:
    return self._last_modified_path
```

**Étape 2** : audit de TOUTES les écritures de `self._current_path` dans le
code. Les remplacer par :

```python
# Si le fichier chargé est une FS :
self._current_fs_path = path
# Si BM/EDC/KZ :
self._current_bm_path = path  # ou _current_path générique
self._last_modified_path = path
```

```bash
grep -rn "_current_path\s*=" arpes/
```

**Étape 3** : adapter `load_controller`, `interaction_controller`, et
`_on_tab_changed` pour utiliser le bon path selon le contexte.

**Étape 4** : sur switch d'onglet (`_on_tab_changed`), si onglet=FS et
`_current_fs_path` défini → restaurer raw_data FS ; si onglet=BM →
restaurer raw_data BM. Implique que `_raw_data` soit aussi dédoublé
(`_raw_data_fs`, `_raw_data_bm`) ou re-chargé à la demande.

**Risque** : c'est un refactor traversant. Test smoke obligatoire. Si trop
fragile, fallback simplifié : garder `_current_path` unique mais ajouter
juste `_pinned_fs_path: Optional[str]` qui "freeze" la FS quand on switche
sur une BM, et utiliser ça pour l'overlay. Moins propre mais limite le
blast radius.

**Recommandation pragmatique** : commencer par le **fallback simplifié**
(`_pinned_fs_path`) — débloque l'overlay (Phase B) sans toucher au cœur de
l'app. Vrai multi-actif (`_current_fs_path/_current_bm_path`) peut venir
en Phase D ultérieure si l'usage le justifie.

**Fichiers touchés (fallback)** : `arpes/app.py` (1 attribut), `_pin_fs_path()`
+ `_unpin_fs_path()` méthodes, PROXY_MAP +2 entrées (148/150).

**Tests** : `tests/test_ui_smoke.py` étendre ou ajouter cas pinning.

### A.5 — File browser tree O3 minimaliste (~3 h)

Refactor du panneau de gauche (browser fichiers). Localisation actuelle :
`arpes/ui/widgets/browsers/files.py` (à vérifier).

**Approche minimaliste** :
- Aujourd'hui : `QListWidget` plat avec une entrée par fichier.
- Cible : `QTreeWidget` (ou `QListWidget` avec indentation visuelle si
  `QTreeWidget` ouvre une boîte de Pandore) :
  - Niveau 0 : `[Sample bna_s2]` (dossier ou regroupement par
    `meta.sample_name`).
  - Niveau 1 : `[FS_001]` (icône FS).
  - Niveau 2 (children de FS_001) : `[BM_03] [BM_04] [BM_05]` (BMs
    auto-discovered + manual).
  - Niveau 0 (orphelins) : BMs sans FS parente, KZ scans, etc.

**Construction** :
1. Itérer `session.files`.
2. Pour chaque FS, calculer `find_bms_for_fs(...)`.
3. Marquer ces BMs comme "rattachées".
4. Les BMs non rattachées vont dans une catégorie `[Orphelins]` au même
   niveau que les FS.

**Re-render** : sur ajout/suppression de fichier, recalculer l'arbre.

**Interaction** :
- Click sur FS → load FS (active onglet FS).
- Click sur BM → load BM (active onglet BM/MDC Fit) ET pin la FS parente
  via `_pin_fs_path(parent_fs)` pour permettre l'overlay.
- Menu contextuel "Rattacher à une autre FS…" → choisit dans liste des FS
  → set `entry.parent_fs_path`.
- Menu contextuel "Détacher" → clear `entry.parent_fs_path`.

**Fichiers touchés** : `arpes/ui/widgets/browsers/files.py` (refactor
moyen), peut-être un nouveau helper `arpes/ui/widgets/browsers/file_tree_builder.py`
(~80 LOC) pour construire l'arbre sans muter le widget.

**Tests** : `tests/test_file_tree_builder.py` (~5 cas, headless si possible
en testant la fonction pure qui retourne la structure d'arbre).

### Bilan Phase A

- 5 sous-tâches, ~1 à 1,5 jour si pas de blast radius surprise.
- LOC ajoutées : ~250 nouveau + ~150 modifié.
- Risque moyen : A.4 (multi-actif). Fallback `_pinned_fs_path` réduit le
  risque drastiquement.
- Tests : ~3 nouveaux fichiers, ~20 cas.
- Couverture : aucune régression attendue si fallback Phase A.4 choisi.

---

## Phase B — Overlay BM cuts sur FS (~1 jour)

Objectif : la feature demandée — afficher les lignes BM par-dessus la FS.

### B.1 — Helper physique pur (~2 h)

Nouveau module `arpes/physics/bm_cut_overlay.py` (~100 LOC) :

```python
@dataclass(frozen=True)
class BMCutLine:
    label: str
    bm_path: str
    polar: float           # angle moteur BM (deg)
    azi: float | None
    hv: float
    kx_points: np.ndarray   # ligne dans le repère FS
    ky_points: np.ndarray
    quality: Literal["exact", "rotated", "scaled", "incompatible"]
    warning: str            # "Δhv=14 eV, échelle approchée"

def compute_bm_cut_in_fs_frame(
    bm_entry: FileEntry,
    bm_path: str,
    fs_metadata: dict,
    fs_polar_center: float,
    fs_azi: float | None,
    fs_hv: float,
    a_lattice: float,
    work_func: float,
) -> BMCutLine | None:
    """Projette une BM dans le repère (kx, ky) d'une FS donnée.

    Retourne None si scan_kind != BM ou si données manquantes.
    Quality :
      - "exact" : même hv ET même azi (± tolerance)
      - "rotated" : hv OK, azi diffère (rotation rigide 2D appliquée)
      - "scaled" : hv diffère (échelle k extrapolée, à interpréter prudemment)
      - "incompatible" : impossible à projeter, log warning
    """
```

**Formules** :
```
ky_bm_in_fs = scale(hv_fs) · sin(polar_bm - polar_fs_center)
# Puis rotation par Δazi = azi_fs - azi_bm autour de Γ :
kx_line = kx_pts · cos(Δazi) - ky_bm_in_fs · sin(Δazi)
ky_line = kx_pts · sin(Δazi) + ky_bm_in_fs · cos(Δazi)
# Si Δhv ≠ 0, scale ajusté par ratio sqrt(Ek_bm / Ek_fs).
```

Réutilise `k_to_angle_offset_deg` inverse depuis `arpes/physics/gamma.py`
pour la cohérence avec le reste de l'app.

**Tests** : `tests/test_bm_cut_overlay.py` (~8 cas).

### B.2 — Agrégateur côté controller (~30 min)

Méthode dans `gamma_controller` ou nouveau mini-controller :

```python
def collect_bm_cuts_for_current_fs(self) -> list[BMCutLine]:
    """Pour la FS courante (_current_fs_path ou pinned), calcule la
    projection de toutes les BMs auto-discovered + manual."""
    fs_path = self._current_fs_path or getattr(self, "_pinned_fs_path", None)
    if not fs_path:
        return []
    fs_entry = self._session.files.get(fs_path)
    fs_raw = ...  # depuis cache ou _raw_data si actif
    pairs = find_bms_for_fs(fs_entry, fs_path, self._session.files)
    cuts = []
    for bm_path, bm_entry, _reason in pairs:
        cut = compute_bm_cut_in_fs_frame(bm_entry, bm_path, fs_raw, ...)
        if cut:
            cuts.append(cut)
    return cuts
```

### B.3 — Rendering matplotlib (~2 h)

Dans `FSControlPanel` (`arpes/ui/widgets/fs_panel.py`) :
- Nouvelle checkbox `chk_show_bm_cuts` ("Afficher BM cuts").
- Signal `bm_cuts_visibility_changed = pyqtSignal(bool)`.

Dans `FermiSurfaceCanvas.draw_fs(...)` :
- Après le draw normal, si toggle activé, appeler `draw_bm_cuts(cuts)`.

Méthode `draw_bm_cuts(cuts)` :
```python
COLOR_MAP = {"exact": "cyan", "rotated": "orange", "scaled": "red"}
for cut in cuts:
    line, = self.ax.plot(
        cut.kx_points, cut.ky_points,
        color=COLOR_MAP[cut.quality],
        linestyle="--" if cut.quality == "scaled" else "-",
        linewidth=1.2, alpha=0.75,
        picker=5,                     # 5 pixels pour pick
    )
    line.bm_cut_label = cut.label    # tag pour pick handler
    self._bm_cut_lines.append(line)
```

**Interaction pick** : connecter `mpl_connect("pick_event", on_bm_cut_pick)` →
load la BM cliquée + pin la FS.

**Tooltip / annotation** : optionnel — au survol, afficher
`f"{label} — polar={polar:+.2f}° — Δazi={dazi:+.2f}°"`. Si compliqué, simple
text annotation persistante près du début de chaque ligne.

### B.4 — Wiring (~30 min)

Dans `arpes/ui/builders/panels.py` :
```python
if hasattr(window._fs_controls, "bm_cuts_visibility_changed"):
    window._fs_controls.bm_cuts_visibility_changed.connect(
        window._toggle_bm_cuts_overlay
    )
```

Nouveau handler `_toggle_bm_cuts_overlay(visible: bool)` dans le controller
qui flippe un flag + appelle `_draw_fs_tab()` (qui à son tour appellera
`draw_bm_cuts` si flag actif).

PROXY_MAP : +1 entrée (`_toggle_bm_cuts_overlay`).

### Bilan Phase B

- LOC : ~250 nouveau, ~50 modifié.
- Risque : faible (purement additif, toggle off par défaut).
- Couverture : `test_bm_cut_overlay.py` + 1 cas dans `test_ui_smoke.py`.

---

## Phase C — Opt-in avancé (optionnel, plus tard)

À n'attaquer que si l'usage de A+B révèle un besoin :

- **C.1** : panneau dédié "Dataset" affichant FS + BMs + KZ liées avec stats
  (nombre de fits, kF moyens, etc.).
- **C.2** : multi-actif vrai (Phase A.4 version complète) si le pinning
  unique se révèle insuffisant.
- **C.3** : extraction de "BM virtuelle" depuis le volume FS (slice à un ky
  donné, comparable directement à une BM réelle).
- **C.4** : hover sur une ligne BM → mini-plot 1D du spectre cumulé le long
  de la coupe.

Pas de design ici — à faire au moment voulu.

---

## Ordre d'exécution recommandé

1. **A.1** scan_kind reliable (30 min, prérequis tout)
2. **A.3** champ `parent_fs_path` (15 min, trivial)
3. **A.2** auto-discovery module pur (2 h, headless testable)
4. **A.4 fallback** `_pinned_fs_path` (1 h, low risk)
5. **B.1** helper `bm_cut_overlay` pur (2 h)
6. **B.2** agrégateur controller (30 min)
7. **B.3** rendering matplotlib (2 h)
8. **B.4** wiring + toggle (30 min)
9. **A.5** file browser tree (3 h, peut être différé après B si pressé)
10. Tests + suite complète + commit en 2 commits (`feat(orga): A —
    pairing infrastructure`, `feat(fs): B — BM cuts overlay`)

Phase A.4 vrai multi-actif et A.5 file tree peuvent être différés en
Phase D si tu veux d'abord valider B en usage.

---

## Fichiers touchés (récap)

### Nouveaux
- `arpes/io/file_pairing.py` (~120 LOC)
- `arpes/io/scan_kind_inference.py` (~30 LOC, si A.1 nécessite)
- `arpes/physics/bm_cut_overlay.py` (~100 LOC)
- `arpes/ui/widgets/browsers/file_tree_builder.py` (~80 LOC, Phase A.5)
- `tests/test_file_pairing.py`
- `tests/test_bm_cut_overlay.py`
- `tests/test_scan_kind_inference.py`
- `tests/test_file_tree_builder.py`

### Modifiés
- `arpes/core/session.py` (FileEntry + parent_fs_path)
- `arpes/app.py` (_pinned_fs_path attribut + handlers)
- `arpes/ui/widgets/fs_panel.py` (checkbox + rendering)
- `arpes/ui/widgets/browsers/files.py` (tree rendering, Phase A.5)
- `arpes/ui/builders/panels.py` (wiring)
- `arpes/ui/controllers/proxy_map.py` (+3 entries)
- `arpes/ui/controllers/gamma_controller.py` ou nouveau (agrégateur)
- `tests/test_session.py` (roundtrip parent_fs_path)
- `tests/test_ui_smoke.py` (pin/unpin cases)

### Doc
- `BM_FS_ORGANIZATION_PLAN.md` (ce fichier, à archiver une fois fait)
- `CLAUDE.md` (mettre à jour patterns : pairing module pur + pinning)

### Estimation totale
- ~700 LOC ajoutées, ~200 LOC modifiées.
- ~25 nouveaux tests.
- ~1,5 jour pour A+B avec fallback A.4.

---

## Risques + mitigations

| Risque | Mitigation |
|---|---|
| A.4 multi-actif complet casse beaucoup | Fallback `_pinned_fs_path` minimal |
| Auto-discovery rate des cas légitimes | Override manuel via `parent_fs_path` (M4) |
| Rendering matplotlib ralentit la FS | Toggle off par défaut, ne dessine que si activé |
| File tree rendering trop lourd | Reporter Phase A.5, garder browser plat avec tag `[FS]` `[BM]` |
| Critères de compatibilité trop stricts | Exposer `PairingCriteria` via QSettings (Phase ultérieure) |
| Δhv mal géré pour le scaling | Marquer `quality="scaled"` + couleur rouge + warning explicite |

---

## Définition de done

- Suite verte (504+ pass).
- gamma_controller + nouveaux fichiers tous sous 700 LOC.
- PROXY_MAP < 150 (actuellement 147 + ~4 = 151 — attention, peut nécessiter
  consolidation verb-dispatch sur `_bm_cut_action` si on dépasse).
- Toggle "Afficher BM cuts" fonctionne en local après load d'un dossier
  contenant ≥1 FS + ≥1 BM.
- Click sur une ligne BM dans la FS → load cette BM dans l'onglet BM.
- Auto-discovery affiche les BMs compatibles + override manuel via menu
  contextuel.
