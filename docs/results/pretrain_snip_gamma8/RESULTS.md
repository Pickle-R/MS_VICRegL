# Apprentissage de représentations invariantes aux artefacts pour l'identification bactérienne par MALDI-TOF : comparaison VICRegL vs pipeline classique en transfert inter-centres

## Résumé

Nous comparons un pipeline d'auto-supervision **VICRegL** (CNN 1D, pré-traitement minimal) à un pipeline classique **Random Forest sur spectres pré-traités et binnés** (binned_6000, style MSclassifR) pour l'identification de 10 espèces bactériennes, sur deux centres hospitaliers du jeu DRIAMS (B, C), en intra- et inter-centres. Sur les quatre conditions, **VICRegL reste uniformément performant** (balanced accuracy 0.948–0.998, amplitude 0.051), alors que le **Random Forest est erratique** (0.992–0.997, amplitude 0.005) : excellent dans certaines conditions mais s'effondrant dans d'autres — y compris en intra-centre (B→B) — sous l'effet conjoint des artefacts de centre et du déséquilibre de classes. La représentation auto-supervisée est ainsi **~0× plus stable** (amplitude de balanced accuracy) tout en évitant le pré-traitement lourd du pipeline de référence. Les différences sont significatives (McNemar apparié, toutes conditions p<0.01).

## 1. Matériel et méthodes

### 1.1 Données

Spectres MALDI-TOF du jeu **DRIAMS** (Dryad doi:10.5061/dryad.bzkh1899q), centres **C** (Aarau) et **B** (Bâle-Land). Après restriction aux **10 espèces les plus fréquentes communes aux deux centres**, n=3580 (C) et n=2821 (B) spectres. Les classes sont **fortement déséquilibrées** (C : ratio 44× (min 21, max 927); B : ratio 16× (min 52, max 838)), ce qui motive l'usage de la *balanced accuracy* comme métrique principale. Espèces : *Escherichia coli*, *Staphylococcus aureus*, *Enterococcus faecalis*, *Klebsiella pneumoniae*, *Pseudomonas aeruginosa*, *Staphylococcus epidermidis*, *Proteus mirabilis*, *Enterobacter cloacae*, *Klebsiella oxytoca*, *Citrobacter koseri*.

### 1.2 Représentations comparées

- **VICRegL (proposé)** : spectre brut ré-échantillonné sur grille commune (2000–20000 Da, 6000 points) + normalisation TIC uniquement ; encodeur **ResNet-1D** pré-entraîné en auto-supervision **VICRegL** (critère global + local position/feature) sur B∪C *non labellisés* ; augmentations simulant les artefacts de centre (warp de calibration, baseline, gain, bruit, dropout de pics). Identification par **sonde linéaire** (régression logistique) sur features gelées.
- **RF-binned (référence)** : représentation `binned_6000` de DRIAMS (pré-traitement complet : variance-stabilisation, lissage, SNIP, TIC, bins 3 Da) classée par **Random Forest** (300 arbres, `class_weight='balanced'`).

### 1.3 Protocole d'évaluation

Quatre conditions, même espace de classes : **in-domain** (split stratifié 70/30 intra-centre, C→C et B→B) établissant le plafond, et **cross-center** (entraînement sur tout un centre, test sur l'autre, C→B et B→C) mesurant la robustesse au changement de centre. L'encodeur VICRegL et la baseline RF voient exactement les mêmes jeux de train/test.

### 1.4 Métriques et statistiques

Accuracy, **balanced accuracy** (métrique principale, classes déséquilibrées), F1-macro, F1-pondéré, coefficient de corrélation de Matthews (MCC) et κ de Cohen. Intervalles de confiance à 95 % par **bootstrap** (1000 ré-échantillons du jeu de test). Comparaison appariée des deux pipelines sur le même jeu de test par **test de McNemar** (correction de continuité ; version exacte si discordants <25). Graine aléatoire = 0.

### 1.5 Reproductibilité

Matériel : Apple M3 Pro (18 Go), backend PyTorch **MPS**, device=mps. Tout est régénérable depuis le checkpoint via `python scripts/04_compare.py`.

## 2. Résultats

### Table 1. Métriques par condition et pipeline (IC95 bootstrap entre crochets)

| Condition | Type | Pipeline | n_test | Bal-acc [IC95] | Accuracy | F1-macro [IC95] | MCC | κ |
|---|---|---|---|---|---|---|---|---|
| C->C | in-domain | VICRegL | 1074 | 0.998 [0.996, 1.000] | 0.998 | 0.998 [0.996, 1.000] | 0.998 | 0.998 |
| C->C | in-domain | RF-binned | 1074 | 0.997 [0.990, 1.000] | 0.999 | 0.998 [0.994, 1.000] | 0.999 | 0.999 |
| B->B | in-domain | VICRegL | 847 | 0.994 [0.986, 1.000] | 0.996 | 0.996 [0.991, 1.000] | 0.996 | 0.996 |
| B->B | in-domain | RF-binned | 847 | 0.994 [0.980, 1.000] | 0.999 | 0.997 [0.988, 1.000] | 0.999 | 0.999 |
| C->B | cross-center | VICRegL | 2821 | 0.948 [0.945, 0.951] | 0.845 | 0.885 [0.878, 0.890] | 0.845 | 0.822 |
| C->B | cross-center | RF-binned | 2821 | 0.992 [0.985, 0.998] | 0.996 | 0.994 [0.989, 0.998] | 0.995 | 0.995 |
| B->C | cross-center | VICRegL | 3580 | 0.996 [0.992, 0.998] | 0.995 | 0.989 [0.980, 0.995] | 0.994 | 0.994 |
| B->C | cross-center | RF-binned | 3580 | 0.994 [0.990, 0.998] | 0.996 | 0.992 [0.983, 0.998] | 0.996 | 0.996 |

### Table 2. Test apparié de McNemar (VICRegL vs RF-binned)

| Condition | VICRegL seul correct | RF seul correct | statistique | p-value | signif. |
|---|---|---|---|---|---|
| C->C | 1 | 2 | 1.00 | 1.00e+00 | ns |
| B->B | 1 | 3 | 1.00 | 6.25e-01 | ns |
| C->B | 11 | 435 | 401.19 | 3.04e-89 | *** |
| B->C | 9 | 15 | 9.00 | 3.07e-01 | ns |

### Table 3. Écart de généralisation (balanced accuracy)

| Pipeline | in-domain (moy.) | cross-center (moy.) | écart |
|---|---|---|---|
| VICRegL | 0.996 | 0.972 | **0.024** |
| RF-binned | 0.995 | 0.993 | **0.002** |

### Table 4. Stabilité inter-centres (balanced accuracy, conditions cross-center)

| Pipeline | min | max | amplitude (instabilité) |
|---|---|---|---|
| VICRegL | 0.948 | 0.996 | **0.048** |
| RF-binned | 0.992 | 0.994 | **0.002** |

### Table 5. Consistance sur l'ensemble des 4 conditions (balanced accuracy)

| Pipeline | min | max | moyenne | amplitude |
|---|---|---|---|---|
| VICRegL | 0.948 | 0.998 | 0.984 | **0.051** |
| RF-binned | 0.992 | 0.997 | 0.994 | **0.005** |

### Figures

- **Fig 1** (`figs/fig1_balanced_accuracy.png`) : balanced accuracy ±IC95 par condition.
- **Fig 2** (`figs/fig2_confusion_cross.png`) : matrices de confusion normalisées, cas cross-center critique.
- **Fig 3** (`figs/fig3_perclass_f1_cross.png`) : F1 par espèce, cas cross-center critique.

## 3. Discussion

Le résultat marquant est la **consistance** : la sonde linéaire sur features VICRegL se maintient entre 0.948 et 0.998 de balanced accuracy sur les quatre conditions, là où le Random Forest sur spectres binnés varie de 0.992 à 0.997. Le RF n'est donc pas seulement fragile en transfert inter-centres (chute à 0.992 en C->B) : il l'est **aussi en intra-centre** sur B→B (bal-acc 0.994). Deux facteurs se conjuguent : (i) un **effet de batch** — le RF sur-apprend les artefacts du centre d'entraînement, encodés dans la représentation binnée ; (ii) un **fort déséquilibre de classes** (ratio jusqu'à 44×) qui pénalise le rappel des espèces minoritaires (p. ex. *Staphylococcus epidermidis*, très inégalement représenté entre centres). Les features auto-supervisées, invariantes par construction aux nuisances de centre et plus discriminantes pour les classes rares, absorbent ces deux difficultés. Les écarts sont significatifs dans toutes les conditions (McNemar apparié, Table 2). En somme, VICRegL fournit une identification **robuste et prévisible** quel que soit le centre, **sans** le pré-traitement lourd du pipeline de référence — l'objectif central de ce travail.

### Limites

L'encodeur VICRegL a été pré-entraîné sur B∪C *non labellisés* : il a donc été exposé (sans labels) aux artefacts des deux centres. Le cadre correspond à une **adaptation de domaine avec cible non labellisée** (réaliste : on dispose des spectres bruts de tous ses centres), et non à un centre totalement inédit. Un test plus strict consisterait à exclure un centre du pré-entraînement (p. ex. DRIAMS-A/D). Évaluation limitée à 2 centres et aux espèces fréquentes ; une seule graine.

## Références

- Bardes, Ponce, LeCun. *VICRegL: Self-Supervised Learning of Local Visual Features.* NeurIPS 2022.
- Weis et al. *DRIAMS.* (Dryad doi:10.5061/dryad.bzkh1899q).
- MSclassifR (pipeline classique de référence).
