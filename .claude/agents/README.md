# Conseil IA ARPES — agents custom

9 agents spécialisés appelés en parallèle avant toute feature ou modification non-triviale de l'app ARPES. Pendant chaque conseil, **arpes-architect est obligatoire** : il garantit que la structure post-refonte (`arpes/{core,io,physics,ui/{builders,controllers,widgets}}`) est préservée et qu'aucune God class ne réapparaît.

## Liste des agents

| Agent | Rôle | Quand l'appeler |
|-------|------|------------------|
| `arpes-architect` | Architecte applicatif. Décide où placer le code, refuse les God class. | **Toujours** pour toute feature multi-fichier. |
| `arpes-physicist` | Cohérence physique (k, Γ, polar, EF, dispersion). | Quand l'idée touche la physique. |
| `arpes-geometry` | Conventions angles/repères labos. | Quand polar/tilt/azi/theta0 sont touchés. |
| `arpes-pyqt-dev` | Faisabilité UI (signals, debouncers, lifecycle). | Toute modif widget/controller/signal. |
| `arpes-io-architect` | Impact loaders + métadonnées + `detect_format`. | Toute modif loader/parsing/logbook. |
| `arpes-redteam` | Cas pathologiques concrets. | **Toujours** avant merge. |
| `arpes-numerics` | Complexité, mémoire, stabilité numérique. | Si calcul lourd ou algo numérique. |
| `arpes-ux` | Texte exact statusbar/warning/dialogs. | Si feature visible utilisateur. |
| `arpes-arbiter` | Synthèse + décision tranchée. | **Toujours en dernier**, jamais seul. |

## Workflow standard

1. User soumet une idée de feature/modification.
2. Claude Code (main thread) lance en parallèle, dans un seul message, les agents pertinents :
   - **Toujours** : `arpes-architect`, `arpes-redteam`.
   - **Selon contexte** : physicist, geometry, pyqt-dev, io-architect, numerics, ux.
3. Une fois tous les avis reçus, Claude Code lance `arpes-arbiter` avec les avis en input.
4. Arbiter rend une décision : **IMPLÉMENTER**, **TESTER D'ABORD**, ou **REFUSER**.
5. Si IMPLÉMENTER → Claude exécute le plan d'action.
6. Si REFUSER → Claude explique le motif et attend ré-écriture de la demande.

## Critères de refus non négociables (vérifiés par arbiter)

1. Aucun plan de test sur données réelles possible avant merge.
2. Casse un format supporté sans plan de migration.
3. Correction automatique sans trace visible utilisateur.
4. Contredit une convention déjà actée sans justification documentée.
5. Fait dépasser un fichier au-delà de 700 LOC sans split (architecte gate).
6. Introduit un global mutable ou un lazy import circulaire.
7. Red team trouve ≥1 cas pathologique non géré → "tester d'abord".

## Échappatoires

- Bug fix trivial 1-2 lignes : conseil non requis. Caveman builder direct.
- Renommage purement cosmétique : conseil non requis.
- Toute autre modif : conseil obligatoire.
