# DECISIONS — ARPES Explorer

Journal **append-only**. Une entrée = ce qui a changé + **pourquoi**, 3-5 lignes
max. Le *comment* est dans le code/git ; ici seulement le *quoi* et le *pourquoi*
non-déductibles. Plus récent en haut.

Historique détaillé pré-2026-06-06 archivé :
`docs/ai/archive/AUDIT_UPDATES_HISTORY.md` (ne pas charger sauf besoin précis).

---

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
