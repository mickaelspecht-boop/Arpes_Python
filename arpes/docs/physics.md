# Formules Physiques

## Axe energie

Les cartes sont affichees en energie relative a EF. L'offset EF sauvegarde corrige le zero d'energie du fichier courant.

## MDC

Un fit MDC ajuste des profils en k a energie fixe. Les points kF extraits forment la dispersion experimentale utilisee pour les resultats de branche.

## Largeur et resolution

La largeur brute mesuree par fit contient la contribution instrumentale. Les resultats physiques utilisent les resolutions energie et k disponibles dans la session lorsque c'est possible.

## Masse effective

La masse effective est estimee a partir de la pente locale de dispersion. Le parametre de maille a sert a convertir les unites reduites vers A^-1.

## Self-energy

L'outil Re Sigma compare la dispersion experimentale a une bande DFT interpolee au meme k. Le resultat est un diagnostic visuel: il depend du choix de bande, du segment et de l'alignement manuel.

## KZ

Le calcul kz utilise l'energie photon, la fonction de travail, le potentiel interne et la geometrie. Les cartes kz sont comparatives et doivent etre interpretees avec les hypotheses de modele electron libre final.
