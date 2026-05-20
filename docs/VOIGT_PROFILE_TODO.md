# TODO — Profil Voigt (P3-E vrai)

Statut : **non implémenté**. Spec pour reprise dédiée.

## Pourquoi

Actuel : `_make_peak_pairs_model` (`arpes/ui/widgets/plots/fit_overlay.py`)
hardcodé Lorentzien (`_lor_peak`). Pour bandes corrélées (BaNi₂As₂,
cuprates) la résolution instrumentale finie et l'asymétrie de raie sont
mieux décrites par un pseudo-Voigt (mélange Lorentzien + Gaussien).
`_voigt_pseudo` existe déjà dans `fit_overlay.py` mais n'est pas branché
sur les paires.

## Implémentation prévue

1. **`_make_peak_pairs_model`** : ajouter `shape='lorentzian'|'voigt'`.
   Quand `voigt` : appender un paramètre **`eta_global` ∈ [0,1]** au
   vecteur p (η=0 pur Lorentzien, η=1 pur Gaussien). Remplacer
   `_lor_peak(x, x0, A, w)` par `_voigt_pseudo(x, x0, A, w, eta_global)`.
   `n_extra += 1` quand `voigt`.
2. **`fit_mdc_peak_pairs`** (`mdc_fit.py`) :
   - Paramètre `shape='lorentzian'` (défaut).
   - p0 : append `eta_init=0.5` quand voigt.
   - Bornes : `lo += [0.0]`, `hi += [1.0]` pour η.
   - Récupération du `eta` final → dans `fit_result["eta"]`.
3. **`FitParams`** (`core/session.py`) : `+shape: str = "lorentzian"`.
4. **UI** (`params_fit.py`) : combobox `cmb_lineshape` (lorentzian /
   voigt) dans la section « Fit MDC ». Connecter à `fit_only_changed`.
5. **`MdcFitter.fit_kwargs`** : passer `shape` depuis `fp.shape`.
6. **Hash params** (`compute_fit_params_hash`) : déjà extensible via
   `fp` dict (sera capturé automatiquement).
7. **Tests** :
   - Synthétique : profil Lorentzien pur → fit voigt doit converger
     vers η≈0 (± seuil).
   - Synthétique Gaussien pur → η≈1.
   - Non-régression : mode 'lorentzian' donne mêmes résultats qu'avant.

## Risques (arpes-redteam)

- Layout p0/bounds change selon `width_mode`. Ne pas casser
  `width_mode='global'` (un `w_global` déjà à la fin, η viendrait
  **après**).
- Tests `test_fit_controller` doivent rester verts (FakeAP n'utilise
  pas le vrai modèle, mais le pipeline réel doit l'accepter).
- Performance : pseudo-Voigt légèrement plus lent que Lorentzien (~×1.3),
  acceptable pour fit complet ; ensemble fit (×N) reste tractable.

## Effort estimé

~150-250 lignes dont 80% dans `fit_overlay.py` + `mdc_fit.py`, plus 20 lignes
UI + tests. Une demi-journée focalisée, conseil
(physicist + numerics + redteam + arbiter) avant code.
