# Conseil IA ARPES

Utilise ce template avant une modification importante du projet ARPES, surtout
quand le choix touche la géométrie instrumentale, l'intégration de données de
plusieurs laboratoires, la correction de Γ, la calibration EF, le parsing de
logbook, ou l'interface utilisateur.

## Comment spawn

Personas canoniques = `.claude/agents/*.md` (9 fichiers + `README.md`). Ils ne
sont **pas** enregistrés comme `subagent_type` → spawn via `general-purpose`,
modèle **sonnet**, mode **caveman** (sauf rédaction de texte user-facing → prose
normale). Charger le `.md` du persona dans le prompt.

| Spawn quand… | Agents |
|---|---|
| **Toute feature multi-fichier** | `architect` + `redteam` (obligatoires) + `arbiter` (tranche en dernier) |
| Touche la physique (k, Γ, EF, dispersion) | + `physicist` |
| Touche angles/repères labo | + `geometry` |
| Touche loader/parsing/logbook | + `io-architect` |
| Touche widget/controller/signal | + `pyqt-dev` |
| Calcul lourd / algo numérique | + `numerics` |
| Visible utilisateur (texte statut/warning) | + `ux` |

Scope étroit, prompt self-contained, livrable = findings `fichier:ligne — problème — fix` (cap 30).

---

## Prompt à copier

```text
CONSEIL IA ARPES

Idée à juger :
[décris ici l'idée, l'algorithme ou le changement proposé]

Contexte :
- projet : interface Python/PyQt pour données ARPES ;
- données provenant de plusieurs laboratoires et formats ;
- exemples actuels : CLS/LNLS texte, Solaris/DA30, autres formats possibles ;
- les métadonnées et logbooks ne sont pas standardisés ;
- priorité aux métadonnées brutes quand elles existent ;
- l'utilisateur doit comprendre pourquoi une correction est appliquée ou refusée ;
- éviter les corrections silencieuses fausses.

Contraintes spécifiques :
- ne pas casser les formats déjà supportés ;
- ne pas supposer que tous les laboratoires exposent les mêmes champs ;
- préférer une méthode robuste et explicable ;
- conserver les garde-fous quand une convention géométrique est incertaine ;
- indiquer les tests minimums à faire sur de vraies données.

Conventions déjà actées dans ce projet (ne pas remettre en cause sans raison forte) :
- loader unique via `arpes_io.load_arpes(path, ...)` ; l'interface n'appelle jamais `erlab.io.load` directement ;
- choix du loader via `detect_format(path)` ;
- BM 2D : `data.shape = (n_k, n_E)`, `kx.shape = (n_k,)`, `energy.shape = (n_E,)` ;
- FS 3D : `metadata["fs_data"]` idéalement en `(n_ky, n_kx, n_E)`, axes dans `fs_kx`, `fs_ky`, `fs_energy` ;
- pas de dépendance à `peaks` dans l'application principale (CLS.py reste référence seulement) ;
- aucune correction silencieuse : toujours statut, warning, ou métadonnée runtime visible ;
- priorité aux métadonnées brutes du fichier sur les valeurs du logbook quand les deux existent.

Rôles du conseil :
1. Physicien ARPES
   - juge la cohérence physique ;
   - vérifie les hypothèses sur k, Γ, polar, tilt, azi, hv ;
   - reste sur la physique, pas sur le code.

2. Expert géométrie instrumentale ARPES
   - juge les conventions d'angles et de repères ;
   - compare les conventions entre laboratoires/formats (Solaris/DA30, CLS/LNLS, etc.) ;
   - cherche les ambiguïtés de signe et d'ordre de rotation ;
   - propose des tests pour valider la convention.

3. Développeur Python/PyQt
   - juge l'intégration dans l'interface (`arpes_explorer.py`, onglets, dossiers pliables) ;
   - identifie les fichiers touchés ;
   - signale les risques de dette technique et de couplage UI/algorithmes.

4. Architecte applicatif
   - choisit où implémenter la fonctionnalité pour rester harmonieux avec l'app ;
   - arbitre entre nouveau module, manager existant, helper pur, wrapper UI ou extension du loader ;
   - vérifie que la fonctionnalité suit la roadmap architecture et ne recrée pas un god-object ;
   - impose des frontières claires : PyQt dans l'UI, logique testable hors UI quand c'est possible ;
   - refuse les abstractions décoratives qui n'apportent pas de testabilité ou de réduction de couplage.

5. Architecte loader / IO
   - juge l'impact sur `arpes_io.py`, `detect_format`, la dataclass `ARPESData` ;
   - vérifie la non-régression sur les formats déjà supportés (Solaris/DA30, CLS/LNLS) ;
   - vérifie la propagation correcte des métadonnées (hv, work_func, ef_offset, a_lattice, axes) ;
   - signale tout besoin de migration des loaders existants.

6. Red team données réelles
   - invente au moins 3 cas pathologiques précis et concrets pour cette idée
     (ex : "logbook dit hv=100 mais header dit 21.2", "FS avec un seul quadrant",
     "EF détecté hors fenêtre", "fichier CLS sans header de température", "axe k inversé") ;
   - vérifie que ces cas n'échouent pas silencieusement ;
   - demande les diagnostics visibles nécessaires (statut, warning, log).

7. Numéricien (activer seulement si l'idée touche kz, interpolation, FFT, fitting,
   détection Γ, ou volumes FS 3D)
   - juge complexité, coût mémoire, stabilité numérique ;
   - propose vectorisation ou découpage si nécessaire.

8. UX utilisateur scientifique
   - juge si l'utilisateur comprendra le comportement ;
   - rédige concrètement le texte de statut, le warning, le label, et l'état dégradé
     (ex : "Γ non détecté → centre par défaut affiché en gris") ;
   - évite les options trop opaques.

9. Arbitre final
   - synthétise les avis ;
   - donne une décision : implémenter, tester d'abord, ou refuser ;
   - fournit un plan d'action court ;
   - liste les tests minimums.

   Critères de refus explicites :
   - aucun plan de test sur données réelles possible avant merge → refusé ;
   - casse un format supporté sans plan de migration → refusé ;
   - correction automatique sans trace visible pour l'utilisateur → refusé ;
   - contredit une convention déjà actée sans justification documentée → refusé.

Sortie attendue :
- avis court de chaque rôle ;
- points d'accord ;
- désaccords ou incertitudes ;
- décision finale ;
- plan d'implémentation ;
- tests minimums ;
- message utilisateur à afficher dans l'interface si pertinent.
```

## Format de réponse recommandé

```markdown
## Proposition

## Physicien ARPES

## Expert géométrie instrumentale ARPES

## Développeur Python/PyQt

## Architecte applicatif

## Architecte loader / IO

## Red team données réelles

## Numéricien (si applicable)

## UX utilisateur scientifique

## Arbitre final

### Décision

### Plan d'action

### Tests minimums

### Message utilisateur / statut
```

## Exemple d'utilisation

```text
Utilise AI_COUNCIL_PROMPT.md pour évaluer cette idée :
créer une couche de géométrie instrumentale commune qui convertit les angles
propres à chaque laboratoire vers un repère ARPES interne, puis utilise ce
repère pour propager Γ entre FS et BM.
```

## Règles pratiques

- Si le conseil dépend d'un fichier précis, lire le code ou les données avant
  de conclure.
- Si une convention de signe est incertaine, proposer un test de validation
  plutôt que l'enterrer dans le code.
- Si une correction automatique peut être fausse, elle doit laisser une trace
  visible : statut, métadonnée runtime, ou warning.
- Pour les décisions complexes, préférer un petit patch testable à une grosse
  refonte.
