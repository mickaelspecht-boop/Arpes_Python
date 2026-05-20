# Cahier des charges — Superposition DFT Materials Project / ARPES pour semi-métaux

## Objectif

Développer un outil Python permettant de comparer rigoureusement des données ARPES expérimentales avec une structure de bandes DFT récupérée via l'API Materials Project.

Le cas d'usage principal est celui de semi-métaux : l'alignement ne doit pas être fait naïvement en imposant l'égalité entre le niveau de Fermi calculé et le niveau de Fermi expérimental. Il faut plutôt référencer chaque jeu de données à son propre zéro, puis appliquer un décalage chimique global et, si nécessaire, une renormalisation globale de bande.

Formule principale :

```math
E_\mathrm{overlay}(n,\mathbf{k}) =
Z \left( \varepsilon^\mathrm{DFT}_{n,\mathbf{k}} - E_F^\mathrm{DFT} \right)
+ \Delta\mu
```

où :

- `E_F^DFT` est le niveau de Fermi calculé par Materials Project ;
- `E_F^exp` est le niveau de Fermi expérimental ARPES, déjà utilisé comme zéro expérimental ;
- `Delta_mu` est un décalage rigide du potentiel chimique ;
- `Z` est un facteur de renormalisation de la dispersion ;
- `Z = 1` signifie aucune renormalisation ;
- `Delta_mu = 0` signifie aucun shift chimique appliqué.

---

## Règles physiques à respecter

### 1. Ne pas identifier automatiquement `E_F^DFT` et `E_F^exp`

La DFT Materials Project décrit un cristal bulk idéal, avec une stœchiométrie et une structure précises.

L'ARPES mesure un échantillon réel, souvent avec :

- dopage involontaire ;
- lacunes ;
- défauts ;
- différence de stœchiométrie ;
- band bending de surface ;
- reconstruction de surface ;
- potentiel chimique différent ;
- possible différence entre bulk et surface.

Donc le niveau de Fermi calculé ne doit pas être considéré comme un niveau absolu directement comparable à l'expérience.

---

### 2. Convention d'énergie

Les bandes DFT doivent être converties en énergie relative :

```python
E_dft_rel = E_dft_raw - E_fermi_dft
```

Les données ARPES doivent être mises en convention :

```text
E = 0 au niveau de Fermi expérimental
E < 0 pour les états occupés
```

Si les données ARPES utilisent une énergie de liaison positive `E_B`, alors convertir avec :

```python
E_arpes = -E_B
```

Exemple :

```text
E_B = 0.25 eV  ->  E_arpes = -0.25 eV
```

---

### 3. Déterminer `Delta_mu` par les croisements de Fermi

Pour un semi-métal, le critère principal doit être l'accord des croisements de Fermi :

```math
k_F^\mathrm{DFT}(\Delta\mu) \approx k_F^\mathrm{ARPES}
```

Il faut varier `Delta_mu` pour que les poches de Fermi calculées correspondent aux poches observées en ARPES.

Le shift ne doit pas être choisi uniquement pour rendre la figure visuellement jolie.

Objectif numérique recommandé :

```math
\Delta\mu^* =
\arg\min_{\Delta\mu}
\sum_i
\left[
k_{F,i}^\mathrm{ARPES}
-
k_{F,i}^\mathrm{DFT}(\Delta\mu)
\right]^2
```

---

### 4. Déterminer `Z` après `Delta_mu`

On détermine d'abord `Delta_mu` avec les croisements de Fermi.

Ensuite seulement, si les pentes ou largeurs de bande DFT ne correspondent pas à l'ARPES, on ajuste une renormalisation globale :

```math
E_\mathrm{overlay} =
Z E_\mathrm{DFT}^{rel}
+
\Delta\mu
```

Interprétation :

```text
Z = 1.0   : pas de renormalisation
Z = 0.5   : bande expérimentale deux fois moins dispersive que la DFT
Z = 0.33  : bande expérimentale trois fois moins dispersive que la DFT
```

Il faut éviter les renormalisations différentes bande par bande, sauf si c'est explicitement documenté et justifié.

---

## Entrées attendues

Le programme doit accepter :

### A. Identifiant Materials Project

```text
mp_id = "mp-..."
```

Exemple :

```text
mp-149
```

### B. Clé API Materials Project

À lire depuis une variable d'environnement :

```bash
MP_API_KEY
```

Ne jamais mettre la clé API en dur dans le code.

### C. Données ARPES

Format recommandé : CSV.

Colonnes possibles :

```text
kx, ky, kz, energy, intensity
```

ou, pour une coupe 1D :

```text
k, energy, intensity
```

ou, si les énergies sont en énergie de liaison :

```text
k, binding_energy, intensity
```

Le code doit permettre à l'utilisateur de préciser :

```python
energy_mode = "relative"       # energy déjà en eV avec EF_exp = 0
energy_mode = "binding_energy" # binding_energy positive, à convertir en -E_B
```

---

## Sorties attendues

Le programme doit produire :

1. une figure ARPES seule ;
2. une figure DFT seule référencée à `E_F^DFT` ;
3. une figure overlay ARPES + DFT avec `Delta_mu` et `Z` ;
4. un fichier texte ou JSON contenant les paramètres utilisés ;
5. idéalement un tableau des croisements `k_F`.

Exemple de fichier de sortie `fit_report.json` :

```json
{
  "mp_id": "mp-...",
  "efermi_dft_eV": 1.234,
  "delta_mu_eV": -0.080,
  "Z": 0.65,
  "energy_convention": "E=0 at experimental EF; occupied states negative",
  "alignment_criterion": "Fermi crossings",
  "notes": [
    "DFT energies referenced to Materials Project calculated Fermi level",
    "Rigid chemical potential shift applied globally",
    "No band-dependent shifts applied"
  ]
}
```

---

## Architecture conseillée du code

Créer un petit module Python avec les fichiers suivants :

```text
arpes_dft_overlay/
    __init__.py
    mp_loader.py
    arpes_loader.py
    energy_alignment.py
    fermi_crossings.py
    plotting.py
    report.py
    cli.py
```

---

## Fonctions à implémenter

### 1. `load_mp_bandstructure`

Fichier : `mp_loader.py`

But : récupérer la band structure Materials Project.

Signature recommandée :

```python
def load_mp_bandstructure(mp_id: str, line_mode: bool = True):
    ...
```

Comportement :

- utiliser `mp_api.client.MPRester` ;
- lire la clé API depuis `MP_API_KEY` ;
- récupérer la band structure ;
- retourner :
  - les énergies brutes ;
  - le chemin k ;
  - `E_F^DFT` ;
  - les labels haute symétrie si disponibles ;
  - spin up / spin down si applicable.

Pseudo-code :

```python
import os
from mp_api.client import MPRester

def load_mp_bandstructure(mp_id: str, line_mode: bool = True):
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY is not set")

    with MPRester(api_key) as mpr:
        bs = mpr.get_bandstructure_by_material_id(mp_id, line_mode=line_mode)

    return bs
```

---

### 2. `load_arpes_csv`

Fichier : `arpes_loader.py`

But : charger les données expérimentales ARPES.

Signature recommandée :

```python
def load_arpes_csv(path: str, energy_mode: str = "relative"):
    ...
```

Comportement :

- lire un CSV avec `pandas` ;
- si `energy_mode == "binding_energy"`, créer une colonne `energy = -binding_energy` ;
- vérifier que l'énergie est en eV ;
- vérifier que le zéro d'énergie correspond à `E_F^exp`.

Pseudo-code :

```python
import pandas as pd

def load_arpes_csv(path: str, energy_mode: str = "relative"):
    df = pd.read_csv(path)

    if energy_mode == "binding_energy":
        if "binding_energy" not in df.columns:
            raise ValueError("Column 'binding_energy' is required")
        df["energy"] = -df["binding_energy"]

    if "energy" not in df.columns:
        raise ValueError("ARPES file must contain 'energy' or 'binding_energy'")

    return df
```

---

### 3. `reference_dft_to_fermi`

Fichier : `energy_alignment.py`

But : convertir les bandes DFT en énergie relative à leur propre niveau de Fermi.

Signature :

```python
def reference_dft_to_fermi(energies, efermi):
    return energies - efermi
```

---

### 4. `apply_overlay_transform`

Fichier : `energy_alignment.py`

But : appliquer shift chimique et renormalisation.

Signature :

```python
def apply_overlay_transform(E_dft_rel, delta_mu: float = 0.0, Z: float = 1.0):
    return Z * E_dft_rel + delta_mu
```

Important :

- `delta_mu` est en eV ;
- `Z` est sans dimension ;
- le même `delta_mu` et le même `Z` doivent être appliqués globalement à toutes les bandes.

---

### 5. `extract_dft_fermi_crossings`

Fichier : `fermi_crossings.py`

But : extraire les positions `k_F` DFT pour un shift donné.

Signature recommandée :

```python
def extract_dft_fermi_crossings(k_path, band_energies_rel, delta_mu=0.0, Z=1.0):
    ...
```

Méthode :

- appliquer `E_overlay = Z * E_rel + delta_mu` ;
- pour chaque bande, chercher les changements de signe autour de `E = 0` ;
- interpoler linéairement la position du croisement ;
- retourner une liste de `k_F`.

Pseudo-code :

```python
def extract_dft_fermi_crossings(k_path, band_energies_rel, delta_mu=0.0, Z=1.0):
    crossings = []

    for band in band_energies_rel:
        E = Z * band + delta_mu

        for i in range(len(k_path) - 1):
            e1, e2 = E[i], E[i + 1]

            if e1 == 0:
                crossings.append(k_path[i])
            elif e1 * e2 < 0:
                k1, k2 = k_path[i], k_path[i + 1]
                kf = k1 + (0 - e1) * (k2 - k1) / (e2 - e1)
                crossings.append(kf)

    return crossings
```

---

### 6. `fit_delta_mu_from_kf`

Fichier : `fermi_crossings.py`

But : trouver `Delta_mu` qui minimise l'écart entre `k_F^ARPES` et `k_F^DFT`.

Signature recommandée :

```python
def fit_delta_mu_from_kf(
    kf_arpes,
    k_path,
    band_energies_rel,
    delta_mu_bounds=(-0.5, 0.5),
    Z=1.0,
):
    ...
```

Méthode :

- tester `Delta_mu` dans une plage raisonnable, par exemple `[-0.5, 0.5] eV` ;
- extraire les `k_F` DFT ;
- associer les croisements DFT aux croisements ARPES les plus proches ;
- minimiser la somme des écarts quadratiques ;
- retourner le meilleur `Delta_mu`.

Important :

- si le nombre de croisements change brutalement, signaler un warning ;
- si plusieurs solutions donnent un score proche, signaler l'ambiguïté ;
- ne pas cacher le fait qu'un fit est mal contraint.

---

### 7. `fit_Z_from_dispersion`

Fichier : `energy_alignment.py`

But : ajuster `Z` sur les dispersions après avoir fixé `Delta_mu`.

À faire seulement si l'utilisateur fournit des points de dispersion expérimentaux extraits de l'ARPES, par exemple :

```text
k, energy_exp, band_label
```

Méthode simple :

```math
E^\mathrm{exp}(k) \approx Z E^\mathrm{DFT,rel}(k) + \Delta\mu
```

avec `Delta_mu` fixé.

Ne pas ajuster simultanément `Z` et `Delta_mu` sans contrainte forte, car les deux peuvent être corrélés.

---

### 8. `plot_overlay`

Fichier : `plotting.py`

But : produire une figure superposant ARPES et DFT.

Signature recommandée :

```python
def plot_overlay(
    arpes_df,
    k_path,
    band_energies_rel,
    delta_mu=0.0,
    Z=1.0,
    energy_window=(-1.0, 0.5),
    output_path="overlay.png",
):
    ...
```

Exigences :

- ARPES en carte d'intensité ;
- DFT en lignes superposées ;
- ligne horizontale à `E = 0` ;
- titre indiquant `Delta_mu` et `Z` ;
- axes :
  - `k` ou chemin de haute symétrie ;
  - `E - E_F^exp` en eV ;
- sauvegarde PNG et PDF.

---

## Interface CLI souhaitée

Créer une commande :

```bash
python -m arpes_dft_overlay.cli \
    --mp-id mp-... \
    --arpes-csv data/arpes.csv \
    --energy-mode binding_energy \
    --delta-mu -0.08 \
    --Z 0.65 \
    --output overlay.png
```

Avec option de fit automatique :

```bash
python -m arpes_dft_overlay.cli \
    --mp-id mp-... \
    --arpes-csv data/arpes.csv \
    --energy-mode relative \
    --kf-arpes data/kf_arpes.csv \
    --fit-delta-mu \
    --output overlay.png
```

---

## Fichier `kf_arpes.csv`

Format recommandé :

```csv
label,kf
pocket_1_left,-0.085
pocket_1_right,0.092
pocket_2_left,0.310
pocket_2_right,0.365
```

Unités :

```text
k en Å^-1
```

---

## Contrôles de rigueur à inclure

Le programme doit afficher ou sauvegarder les informations suivantes :

```text
E_F^DFT utilisé
Delta_mu appliqué
Z appliqué
Méthode d'alignement
Nombre de croisements k_F ARPES
Nombre de croisements k_F DFT
Écart RMS entre k_F ARPES et k_F DFT
Fenêtre d'énergie utilisée
```

Le programme doit aussi avertir si :

- `abs(Delta_mu) > 0.3 eV`, car le shift est grand pour un semi-métal ;
- `Z < 0.2` ou `Z > 1.5`, car la renormalisation est suspecte ;
- les croisements `k_F` ne peuvent pas être associés clairement ;
- l'utilisateur essaie de décaler chaque bande séparément ;
- l'ARPES est en énergie de liaison mais n'a pas été convertie en énergie relative négative ;
- la comparaison est faite sans tenir compte du possible `k_z`.

---

## Notes sur `k_z`

Pour un matériau 3D, l'ARPES mesure une coupe dépendante de l'énergie photon.

Ne pas conclure trop vite qu'un shift en énergie est nécessaire si le `k_z` expérimental n'est pas identifié.

Formule usuelle du modèle d'état final libre :

```math
k_z \approx
\frac{1}{\hbar}
\sqrt{
2m
\left(
E_\mathrm{kin}\cos^2\theta + V_0
\right)
}
```

Le code peut inclure une fonction optionnelle :

```python
def estimate_kz(E_kin_eV, theta_deg, V0_eV):
    ...
```

Mais cette estimation doit être documentée comme approximative.

---

## Points à vérifier pour Materials Project

Avant de faire une conclusion physique, le script ou le rapport doit rappeler à l'utilisateur de vérifier :

1. le bon polymorphe ;
2. la bonne structure cristalline ;
3. présence ou absence de SOC explicite ;
4. magnétisme ;
5. DFT+U ou non ;
6. proximité de bandes très plates autour de `E_F` ;
7. présence de surface states non décrits par une DFT bulk ;
8. existence d'une reconstruction de surface ;
9. validité du chemin haute symétrie par rapport à la coupe ARPES.

---

## Ce qu'il ne faut pas faire

Ne pas faire :

```text
- aligner visuellement chaque bande séparément ;
- changer Delta_mu pour chaque bande ;
- oublier de dire le shift appliqué ;
- oublier de dire si Z a été appliqué ;
- comparer une coupe ARPES 3D à une mauvaise coupe kz DFT ;
- utiliser EF Materials Project comme vérité expérimentale absolue ;
- interpréter un bon overlay comme preuve définitive sans vérifier surface, SOC et corrélations.
```

---

## Phrase standard à inclure dans les figures ou rapports

Version française :

```text
Les bandes DFT issues de Materials Project ont d'abord été référencées à leur niveau de Fermi calculé. Un décalage rigide du potentiel chimique Delta_mu a ensuite été appliqué pour reproduire les croisements de Fermi observés en ARPES. Lorsque nécessaire, une renormalisation globale de bande Z a été appliquée. Aucun décalage indépendant bande par bande n'a été utilisé.
```

Version anglaise :

```text
The Materials Project DFT band structure was first referenced to its calculated Fermi level. A rigid chemical-potential shift Delta_mu was then applied to match the experimentally observed Fermi crossings. When necessary, a global bandwidth renormalization factor Z was applied. No band-dependent shifts were used.
```

---

## Exemple minimal d'utilisation

```python
from arpes_dft_overlay.mp_loader import load_mp_bandstructure
from arpes_dft_overlay.arpes_loader import load_arpes_csv
from arpes_dft_overlay.energy_alignment import reference_dft_to_fermi, apply_overlay_transform
from arpes_dft_overlay.plotting import plot_overlay

mp_id = "mp-..."
arpes_path = "data/arpes.csv"

bs = load_mp_bandstructure(mp_id, line_mode=True)
arpes_df = load_arpes_csv(arpes_path, energy_mode="binding_energy")

efermi = bs.efermi

# Pseudocode: extract energies and k-path from pymatgen BandStructure object.
# The exact extraction depends on the object structure.
k_path = extract_k_path_from_bs(bs)
energies_raw = extract_band_energies_from_bs(bs)

energies_rel = reference_dft_to_fermi(energies_raw, efermi)

delta_mu = -0.08
Z = 0.65

plot_overlay(
    arpes_df=arpes_df,
    k_path=k_path,
    band_energies_rel=energies_rel,
    delta_mu=delta_mu,
    Z=Z,
    energy_window=(-1.0, 0.5),
    output_path="overlay.png",
)
```

---

## Critère de réussite

L'outil est considéré satisfaisant si :

1. il récupère une band structure Materials Project ;
2. il référence correctement les énergies DFT à `E_F^DFT` ;
3. il lit les données ARPES avec la bonne convention d'énergie ;
4. il applique un seul `Delta_mu` global ;
5. il applique un seul `Z` global, si demandé ;
6. il peut ajuster `Delta_mu` à partir des croisements de Fermi ;
7. il produit une figure overlay claire ;
8. il produit un rapport contenant tous les paramètres ;
9. il signale les cas où la comparaison n'est pas physiquement bien contrainte.
