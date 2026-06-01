# Audit Γ — Plan d'action

Synthèse du conseil ARPES (architect + physicist + redteam + ux + arbiter) sur le workflow Γ. Date: 2026-06-01.

## Contexte

Le workflow Γ s'est étendu sur 5 chemins utilisateur (Auto Γ BM, Détecter Γ FS, Click FS, Γ FS→BM, auto-apply au load) plus 1 chemin loader (offset angulaire pré-load). 4 audits indépendants convergent sur le même diagnostic : workflow correct sur le fond, mais l'exécution accumule des sources de vérité dupliquées, des mutateurs sans garde, et zéro test bout-en-bout. Plusieurs bugs HIGH sont déjà visibles (drift cumulé, double-shift, race au load).

## Consensus (≥ 3 audits)

1. **Source de vérité multiple = racine du problème.** Γ vit dans 5 endroits non synchronisés : `session.gamma_reference`, `session.angle_offsets`, `entry.fs_center_kx/ky`, `entry.fit_params.center_init`, `meta.bm_gamma_axis_*`. Trois sont dérivables des autres.
2. **Re-clic sur axe déjà shifté = double-shift / drift.** `_estimate_gamma_bm`, `_detect_fs_gamma`, `_on_fs_map_click` n'ont aucune garde idempotence.
3. **Single-setter manquant.** Les attributs UI critiques (`sp_cx`, `entry.fit_params.center_init`, `_sel_k`, `entry.fit_zones[].fit_result`) sont mutés par 4 chemins sans point d'entrée unique. Aucun équivalent du pattern `fit_result_store.set_fit_result`.
4. **Feedback utilisateur sur Γ courant absent.** Pas de badge permanent. Le statut Γ est dispersé entre `lbl_res` (BM), `lbl_info` (FS), `status` bar (éphémère), `mark_action_done` (vert 3s).
5. **Couverture e2e zéro.** Aucun test enchaîne fs1→bm04→reclick→reload session.

## Verdict

**Patcher P1 d'urgence → refondre P2 → UX P3.** Pas de big-bang. Le concept (Γ de session projeté par azimut sur chaque fichier) est correct ; l'implémentation actuelle ne le matérialise pas par un contrat unique.

---

## Phase 1 — Urgent (≤ 1 jour)

Objectif : stopper les drift silencieux et bloquer les bugs HIGH avant tout refactor.

### 1.1 Persister les flags d'état d'axe en session JSON

**Fichier** : `arpes/core/session.py`

Aujourd'hui `meta.bm_gamma_axis_centered`, `meta.bm_gamma_axis_shift`, `meta.fs_gamma_axis_*` vivent dans `raw_data["metadata"]` mais ne sont pas persistés. Au reload session, `_apply_stored_gamma_to_current_file` voit `previous_shift = 0` et réapplique le shift complet → `_remap_fit_results_by_delta` re-décale les `fit_result` déjà décalés.

**Action** : étendre `FileEntry` avec un champ `meta_gamma_state: dict` qui sérialise les 6 flags. Restorer dans `_raw_data["metadata"]` après load.

### 1.2 Gardes d'idempotence sur les 3 détecteurs

**Fichier** : `arpes/ui/controllers/gamma_controller.py`

**Action** : ajouter un helper `_axis_is_locked() -> bool` qui retourne `True` si `meta.bm_gamma_axis_centered` ou `meta.angle_offsets_applied`. Refuser au début de `_estimate_gamma_bm`, `_detect_fs_gamma`, `_on_fs_map_click` avec message statusbar : `"Γ déjà appliqué (axe recentré ou offset loader). Utilise 'Oublier Γ' d'abord."`.

### 1.3 Fix bug : `allow_fs=True` manquant

**Fichier** : `arpes/ui/controllers/gamma_controller.py:260`

`_center_current_bm_axis_on_gamma` appelle `_gamma_apply_bm_axis_shift(self._raw_data, gamma_bm, ref=ref)` sans `allow_fs=True`. Si `_raw_data` est un FS, le shift est rejeté silencieusement → marker FS désynchronisé du recentrage BM.

**Action** : détecter `is_fs = meta.get("fs_data") is not None` et passer `allow_fs=True, gamma_ky=ref.get("ky", 0.0)` quand applicable.

### 1.4 Fix bug : branche FS de `_apply_stored_gamma_to_current_file`

**Fichier** : `arpes/ui/controllers/gamma_controller.py:288`

La branche FS de cette méthode n'appelle jamais `_gamma_apply_bm_axis_shift`. Après reload, `fs_kx` brut + marker placé à `kx_ref` projeté = drift visuel.

**Action** : symétriser : si branche FS et `ref` valide, appeler `_gamma_apply_bm_axis_shift(..., allow_fs=True)` + `_remap_fit_results_by_delta`.

### 1.5 Fix bug : coordonnées affichées vs absolues

**Fichiers** : `arpes/ui/controllers/gamma_controller.py:179` (`_on_fs_map_click`), `:200` (`_detect_fs_gamma`)

Quand l'axe FS a déjà été shifté, `res["kx"]` (détecteur) et `event.xdata` (click) sont en coordonnées affichées. Stocker tel quel comme référence absolue → la prochaine détection donnera kx≈0, ce qui écrasera la vraie réf.

**Action** : avant `_store_fs_center_reference`, ré-ajouter `meta.fs_gamma_axis_shift_kx/ky` pour retrouver les coordonnées brutes. Couvert par garde 1.2 mais à garder pour défense en profondeur.

### 1.6 Rollback sur échec de save

**Fichier** : `arpes/ui/controllers/gamma_controller.py:_remap_fit_results_by_delta`

Aujourd'hui : mute `fit_result` puis tente `session.save()`. Si save échoue, mémoire et disque divergent.

**Action** : snapshot des `fit_result` avant mutation, restore si save échoue (try/except autour de la boucle).

### 1.7 Ordre `_apply` ↔ `_update_display_data` au load

**Fichier** : `arpes/ui/controllers/load_controller.py:504`

`_apply_stored_gamma_to_current_file` est appelé avant `_refresh_ui` → trigger de signal `sp_cx` → potentiel redraw avec `_raw_data` partiellement initialisé.

**Action** : soit déplacer après `_refresh_ui`, soit `blockSignals(True)` sur `sp_cx` pendant l'apply.

### 1.8 Tests e2e

**Fichier nouveau** : `tests/test_gamma_workflow_e2e.py`

Scénarios à couvrir (5 minimum) :

- `test_fs_detect_then_bm_switch_then_back` : fs1 détecte → bm04 hérite → switch retour fs1, axes cohérents.
- `test_double_click_no_drift` : `_detect_fs_gamma` × 2 → kx ne dérive pas.
- `test_estimate_bm_blocked_when_axis_centered` : recentrage axe puis Auto Γ BM → refus + statusbar.
- `test_reload_session_no_overlay_drift` : fit + détecter Γ + save + reload + assert kF identique.
- `test_loader_offset_blocks_estimate` : `angle_offsets_applied=True` → toutes actions de détection refusent ou no-op explicite.

Ces tests sont **headless** (`raw_data` synthétique, pas de PyQt requis). Ils doivent être rouges avant les fix 1.1-1.7, verts après.

---

## Phase 2 — Refonte source de vérité (2-3 jours, après P1 vert)

Objectif : un seul contrat, une seule source de vérité, idempotence prouvable.

### 2.1 Nouveau module `arpes/physics/gamma_resolver.py` (pur, sans PyQt)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ResolvedGamma:
    mode: Literal["loader_baked", "axis_shifted", "fit_center_only", "none"]
    display_center: float       # toujours 0.0 sauf fit_center_only
    fit_center_init: float
    axis_shift_delta: float     # à appliquer en plus du shift courant

def resolve(
    raw_data: dict,
    ref: dict | None,
    hv: float,
    phi: float,
    entry_azi: float | None,
    sample_key: str | None = None,
) -> ResolvedGamma:
    ...
```

Fonction pure : prend l'état, retourne la décision. Aucune mutation, aucun side-effect, testable headless.

### 2.2 Single-setter `apply_gamma(ctrl, resolved)`

**Fichier** : `arpes/ui/controllers/gamma_controller.py` (ou nouveau `gamma_apply.py`)

Seul point qui écrit : `sp_cx`, `entry.fit_params.center_init`, `_sel_k`, déclenche `_remap_fit_results_by_delta`, sauve la session.

Tous les 5+ handlers actuels deviennent :
```python
def _on_some_gamma_action(self, ...):
    ref = self._build_ref_from_user_input(...)
    resolved = gamma_resolver.resolve(self._raw_data, ref, ...)
    apply_gamma(self, resolved)
```

### 2.3 Élaguer l'état dupliqué

- Supprimer `session.angle_offsets` (recomputable par `resolve` à partir de `gamma_reference`).
- Supprimer `entry.fs_center_kx/ky` (dérivable par `project_gamma_by_azi`).
- `session.gamma_reference: dict[sample_key, ref]` indexé sur `(sample_name, azi_ref)`.
- Bump `Session.VERSION` + migration read-only de l'ancien format.

### 2.4 Invariant idempotence prouvable

Test : `resolve(raw, ref, hv, phi, azi)` appliqué via `apply_gamma`, puis `resolve` à nouveau → même `ResolvedGamma`. Si non, c'est un bug.

### 2.5 Déplacer `app_angle_offsets.py`

**De** : `arpes/app_angle_offsets.py` (mauvais emplacement, vit hors `ui/` et `io/`).
**Vers** : `arpes/io/gamma_load_offsets.py`.

Le `win.` couplage devient un dataclass `(ref, geom, hv, phi)` passé explicitement.

### 2.6 Wrappers triviaux à éliminer

5 méthodes `_k_to_angle_offset_deg`, `_angle_offsets_from_k_center`, `_project_gamma_by_azi`, `_stored_gamma_reference`, `_gamma_reference_to_bm_center` n'ajoutent qu'un lookup `self._params.sp_hv.value()`. Remplacer par un `GammaContext` immutable passé en paramètre.

**Cible LOC** : `gamma_controller.py` 468 → ~250. `physics/gamma.py` 465 → ~400 + nouveau `gamma_resolver.py` ~150.

---

## Phase 3 — UX simplifiée (optionnel, après P1 + P2)

Objectif : 3 actions claires, état toujours visible, undo possible.

### 3.1 Badge permanent

**Fichiers** : `arpes/ui/builders/panels.py`, nouvelle widget `GammaStatusBadge`.

Format affiché en haut du panneau central :
```
Γ session: kx=+0.041, ky=−0.002 π/a · source=fs_auto (Au_FS_001) · ce fichier: propagé via Δazi=12° · état: appliqué
```

Mise à jour à chaque switch fichier + chaque action Γ.

### 3.2 Fusion contextuelle des boutons

- `Détecter Γ FS` + `Auto Γ BM` → un seul bouton **"Mesurer Γ ici"** qui dispatche selon l'onglet actif.
- `Γ FS → BM` → supprimé (auto au load post-P2).
- `Viser Γ manuel` → renommé **"Pointer Γ"**.
- Ajouter **"Oublier Γ"** + confirm dialog `"Effacer Γ de référence ? (impact: N fichiers)"`.

### 3.3 Warning loader-offset

Si `angle_offsets_applied=True`, désactiver "Mesurer Γ ici" et afficher inline : `⚠ Offset angulaire loader actif — Γ déjà appliqué par le loader.`

**Couverture** : `tests/test_ui_smoke.py` + entrées PROXY_MAP.

---

## Statut P2.bis (livré)

- Câblage `_detect_fs_gamma`, `_on_fs_map_click`, `_estimate_gamma_bm`,
  `_apply_gamma_reference_to_bm` via `apply_resolved_gamma` → 4 sites
  mutateurs collapsés sur le single-setter.
- Nouveau handler `_forget_gamma` (PROXY_MAP `_gamma_ctrl`) : porte de
  sortie aux gardes `_is_axis_locked`. Inverse le shift d'axe, remap
  `fit_result`, clear session.gamma_reference / session.angle_offsets /
  entry.meta_gamma_state / entry.fs_center_kx/ky, reset sp_cx.
- Test `TestForgetGamma` (Qt) couvre l'inversion + clear complet.

## Statut P3 (livré)

- Bouton « Oublier Γ » dans `FSControlPanel` (signal `forget_gamma_requested`)
  câblé via `_forget_gamma_with_confirm` (dialog confirm, impact N fichiers).
- Badge permanent statusbar `_gamma_status_label` (QLabel via
  `addPermanentWidget`) format `Γ kx=±.XXX ky=±.XXX · source · état`
  ou `Γ ∅` / `Γ loader-offset θ0=±.XXX°`. Mis à jour à chaque
  `apply_resolved_gamma` et `_forget_gamma`.
- Split `gamma_controller.py` (720 → 595 LOC) via `gamma_lifecycle.py`
  (144 LOC) — free functions `forget`, `forget_with_confirm`,
  `format_badge_text`, `update_badge` prenant `ctrl` en paramètre.
- PROXY_MAP +3 entrées (147/150).

## Reste à faire (P3.bis, optionnel)

- Fusion contextuelle « Détecter Γ FS » + « Auto Γ BM » → bouton unique
  « Mesurer Γ ici » qui dispatche selon l'onglet actif.
- Renommer « Viser Γ manuel » → « Pointer Γ ».
- Désactiver visuellement « Auto Γ BM » quand `angle_offsets_applied` est
  actif (au lieu du refus runtime).
- Badge enrichi : tag `(propagé via Δazi=12°)` si même session, fichier
  différent.

## Reste à faire (P2.ter, séparable)

- Supprimer `session.angle_offsets` (recalculable depuis
  `gamma_reference` + meta). Bloquant car `distortion_controller` hash
  ce dict pour invalider le cache distortion.
- Supprimer `entry.fs_center_kx/ky` — état UI persisté du panel FS
  distinct de Γ, **à garder** (révision recommandation Architect : ce
  champ n'est pas dérivable, c'est le centre d'affichage persisté).
- Déplacer `arpes/app_angle_offsets.py` → `arpes/io/gamma_load_offsets.py`.
- Bump `Session.VERSION` → 2 + migration read-only ancien format.
- Éliminer 5 wrappers triviaux (`_k_to_angle_offset_deg`, etc.) via
  `GammaContext` immutable.
- Indexer `gamma_reference` par `sample_key` (multi-échantillon).

## Recommandation finale

| Quand | Action |
|---|---|
| **Maintenant** | Phase 1 — écris les 5 tests e2e (rouges), puis pose les 4 gardes + la persistance. ~1 jour, stoppe le drift. |
| **Semaine prochaine** | Phase 2 quand P1 est vert. Refonte source unique. ~2-3 jours. |
| **Plus tard / optionnel** | Phase 3 UX. Pas avant que P1 + P2 soient mergés. |
| **À ne PAS faire** | Big-bang P1+P2+P3 simultané. Refonte sur du sable (P2 sans P1). Renommer boutons (P3) sur un workflow encore buggé. |

## Dette tracée (à ajouter à CLAUDE.md une fois P1 fait)

- Workflow Γ refondu en P2 — voir `gamma_resolver.py` pour le contrat unique.
- `session.angle_offsets` et `entry.fs_center_kx/ky` supprimés en P2 — ne pas réintroduire.
- Toute écriture de `sp_cx` / `entry.fit_params.center_init` / `_sel_k` doit passer par `apply_gamma(ctrl, resolved)` — pattern jumeau de `set_fit_result`.
