# Construire ARPES Explorer en exécutable (Linux / macOS / Windows)

But : livrer un binaire double-cliquable. Plus besoin d'installer
`micromamba`, l'env `peaks`, ni de lancer un terminal.

> **Aucune modification du code applicatif n'est nécessaire.** Ce
> document n'ajoute qu'un fichier de recette (`arpes.spec`) et des
> commandes. L'app reste lançable comme avant via
> `python arpes_explorer.py`.

---

## 1. Principe & contraintes

- Outil : **PyInstaller**. Il fige Python + PyQt6 + numpy/scipy/
  matplotlib + le code `arpes/` dans un livrable autonome.
- **Pas de cross-compilation.** Un build ne tourne que sur l'OS
  (et l'archi) qui l'a produit :
  - Windows → builder sur Windows → `.exe`
  - macOS → builder sur macOS → `.app` (archi Intel `x86_64` ≠
    Apple Silicon `arm64` : un build par archi, ou un Mac de chaque)
  - Linux → builder sur Linux → binaire ELF (compatible si la glibc
    de la machine de build est ≤ celle des machines cibles → builder
    sur l'OS le plus ancien que vous visez)
- Taille attendue : ~150–350 Mo (scipy/numpy/Qt sont gros). Normal.
- Mode `onedir` (un dossier) recommandé d'abord : démarre plus vite,
  débogage plus simple. `onefile` (1 seul fichier) possible ensuite.

---

## 2. Préparer l'environnement de build (commun aux 3 OS)

Sur **chaque** machine de build, dans l'env du projet :

```bash
# env identique à celui de dev (Python 3.12, PyQt6 6.10, numpy 2.3,
# scipy 1.17, matplotlib 3.10)
micromamba activate peaks         # ou conda/venv équivalent
pip install pyinstaller
```

Vérifier que l'app démarre depuis cet env **avant** de packager :

```bash
cd code/app
python arpes_explorer.py          # doit ouvrir l'UI sans erreur
```

Si ça ne marche pas ici, le binaire ne marchera pas non plus :
corriger l'env d'abord.

---

## 3. Le fichier recette `arpes.spec`

Créer **`code/app/arpes.spec`** (à la racine `app/`, à côté de
`arpes_explorer.py`) avec ce contenu :

```python
# -*- mode: python ; coding: utf-8 -*-
# Recette PyInstaller pour ARPES Explorer.
# Build : pyinstaller arpes.spec   (depuis code/app/)
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

hiddenimports = []
# scipy charge des sous-modules dynamiquement -> tout embarquer
hiddenimports += collect_submodules("scipy")
# backend Qt de matplotlib (importé via matplotlib.use("QtAgg"))
hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]
# contrôleurs chargés statiquement dans arpes/app.py -> déjà suivis,
# mais on sécurise les chargements dynamiques (arpes_plots fallback)
hiddenimports += collect_submodules("arpes")

datas = []
# fallback runtime de arpes/app.py:_load_ap() qui peut charger
# le fichier frère arpes_plots.py par chemin -> le livrer à la racine
datas += [("arpes_plots.py", ".")]
# données matplotlib (polices, mplstyle)
datas += collect_data_files("matplotlib")

a = Analysis(
    ["arpes_explorer.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PySide6", "PySide2"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ARPES_Explorer",
    console=False,            # pas de fenêtre terminal noire
    disable_windowed_traceback=False,
    argv_emulation=True,      # macOS : ouvrir fichiers via Finder
    target_arch=None,         # archi de la machine de build
    icon=None,                # mettre "icon.ico"/"icon.icns" si dispo
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="ARPES_Explorer",
)

# macOS : produit aussi un bundle .app double-cliquable
app = BUNDLE(
    coll,
    name="ARPES_Explorer.app",
    icon=None,
    bundle_identifier="org.labo.arpesexplorer",
)
```

> Notes :
> - Les contrôleurs (`distortion_controller`, `gamma_controller`, …)
>   sont importés **statiquement** dans `arpes/app.py` : PyInstaller
>   les suit automatiquement. `collect_submodules("arpes")` est une
>   ceinture de sécurité supplémentaire.
> - `~/.config/arpes/distortion_calib.json` et autres fichiers
>   runtime restent dans le HOME de l'utilisateur : **non concernés**
>   par le packaging, ça marche identique.
> - Pas de fichier `requirements.txt`/`pyproject.toml` dans le repo :
>   l'env `peaks` fait foi. Garder le même env entre dev et build.

---

## 4. Lancer le build (sur chaque OS)

Depuis `code/app/`, env activé :

```bash
pyinstaller arpes.spec
```

Sortie dans `code/app/dist/` :

| OS      | Résultat                                            |
|---------|-----------------------------------------------------|
| Linux   | `dist/ARPES_Explorer/ARPES_Explorer` (+ dossier)    |
| Windows | `dist/ARPES_Explorer/ARPES_Explorer.exe` (+ dossier)|
| macOS   | `dist/ARPES_Explorer.app`                           |

Pour livrer : zipper **tout le dossier** `dist/ARPES_Explorer/`
(Linux/Win) ou le `.app` (macOS). Ne pas extraire le seul binaire.

---

## 5. Spécificités par OS

### 5.1 Linux

- Builder sur la distrib **la plus ancienne** que vous visez (glibc
  ascendante-compatible, pas l'inverse).
- L'utilisateur final : `chmod +x ARPES_Explorer` puis double-clic
  ou `./ARPES_Explorer`.
- Si erreur Qt `xcb`/plugin : installer côté cible
  `libxcb-cursor0` (Debian/Ubuntu) — dépendance système Qt6 courante.

### 5.2 macOS

- Build par architecture : un `.app` `arm64` (Apple Silicon) et/ou
  `x86_64` (Intel). `target_arch="universal2"` possible si l'env a
  des wheels universelles (rare pour scipy → préférer 1 build/archi).
- **Gatekeeper** : sans signature/notarisation Apple, au 1er
  lancement macOS bloque (« app endommagée / éditeur non
  identifié »). Contournements :
  - Distribution interne labo : clic-droit sur l'app → **Ouvrir** →
    confirmer (à faire une seule fois par poste). Suffisant pour
    quelques collègues.
  - Distribution large : compte Apple Developer (99 $/an) +
    `codesign` + `notarytool`. Hors périmètre de ce doc.
- `argv_emulation=True` (déjà dans le spec) permet d'ouvrir un
  fichier en le glissant sur l'icône.

### 5.3 Windows

- Builder sur Windows 10/11. `.exe` dans le dossier `dist/`.
- **SmartScreen** : sans signature de code, « Windows a protégé
  votre PC » au 1er lancement → `Informations complémentaires` →
  `Exécuter quand même`. Une fois par poste.
- Signature optionnelle : certificat code-signing (payant) +
  `signtool`. Hors périmètre.
- Antivirus : un `.exe` PyInstaller frais lève parfois un
  faux positif. Préférer `onedir` (moins de faux positifs que
  `onefile`).

---

## 6. Tester avant de distribuer (ne pas casser l'app)

Checklist sur une machine **vierge** (sans l'env Python) idéalement :

1. Lancer le binaire → l'UI s'ouvre, pas de crash console.
2. Charger un fichier ARPES (FS + BM).
3. Calibrer EF, détecter Γ (auto **et** clic manuel).
4. Onglet **utilitaires** : ouvrir/fermer chaque section
   (Filtre grille, DFT/Théorie, Distorsion BM).
5. Distorsion : trapèze (sym/antisym/free) + parabole, overlay
   pointillé visible en édition puis disparaît après application,
   recadrage k// effectif.
6. Filtre grille/DFT restent sur l'onglet BM uniquement.
7. Export CSV + sauvegarde/chargement de session.

Si un point casse dans le binaire mais marche en `python
arpes_explorer.py` → c'est un import manqué : l'ajouter dans
`hiddenimports` du spec et rebuilder. Itérer.

---

## 7. (Optionnel) Builds automatiques via CI

`.github/workflows/build.yml` — produit les 3 binaires à chaque tag :

```yaml
name: build-executables
on:
  push:
    tags: ["v*"]
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-22.04, macos-14, windows-2022]
    runs-on: ${{ matrix.os }}
    defaults:
      run:
        working-directory: code/app
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pyinstaller numpy scipy matplotlib PyQt6 h5py pandas
      - name: Build
        run: pyinstaller arpes.spec
      - uses: actions/upload-artifact@v4
        with:
          name: ARPES_Explorer-${{ matrix.os }}
          path: code/app/dist/
```

> Adapter la liste `pip install` à l'ensemble réel des imports de
> l'app si la CI ne reproduit pas l'env `peaks`. Le plus fiable :
> générer un `requirements.txt` depuis l'env `peaks`
> (`pip freeze > requirements.txt`) et l'installer en CI.

---

## 8. Récap rapide

| Étape | Commande |
|-------|----------|
| Installer outil | `pip install pyinstaller` |
| Vérifier l'app  | `python arpes_explorer.py` |
| Builder         | `pyinstaller arpes.spec` |
| Livrable        | `dist/ARPES_Explorer/` ou `.app` |
| 1er lancement   | macOS : clic-droit→Ouvrir / Win : Exécuter quand même |
