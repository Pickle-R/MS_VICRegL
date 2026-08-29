# Discrimination fine d'espèces phylogénétiquement proches (cas difficiles du MALDI-TOF)

Encodeur VICRegL **gelé** (pré-entraîné sur B∪C non labellisés), évaluation par **validation croisée stratifiée 5-fold** sur B∪C. Sonde linéaire VICRegL vs Random Forest sur binned_6000.

> ⚠️ **Mise en garde gold standard.** Les labels DRIAMS proviennent d'une identification **MALDI (Bruker Biotyper)**, non du **NGS**. Sur ces espèces proches, ils sont partiellement faillibles : on mesure la séparabilité *telle qu'étiquetée par Biotyper*, pas une vérité confirmée par séquençage. C'est précisément la limite que MSclassifR levait en utilisant le NGS comme étalon-or.

## Synthèse (balanced accuracy, CV 5-fold)

| Groupe | n | classes | VICRegL | RF-binned | McNemar p | meilleur |
|---|---|---|---|---|---|---|
| Enterobacter cloacae complex | 438 | 5 | **0.669** | 0.443 | 7.9e-01 | VICRegL |
| Streptococcus viridans (pneumoniae/oralis/mitis) | 156 | 3 | **0.788** | 0.836 | 5.4e-02 | RF-binned |
| Klebsiella (pneumoniae/oxytoca/variicola) | 800 | 3 | **0.979** | 0.976 | 6.2e-01 | VICRegL |

## Enterobacter cloacae complex

Effectifs : *Enterobacter aerogenes* 103, *Enterobacter asburiae* 30, *Enterobacter cloacae* 271, *Enterobacter kobei* 23, *Enterobacter ludwigii* 11.

| Pipeline | balanced-acc | accuracy | F1-macro | MCC |
|---|---|---|---|---|
| VICRegL | 0.669 | 0.868 | 0.656 | 0.764 |
| RF-binned | 0.443 | 0.861 | 0.450 | 0.741 |

McNemar : VICRegL seul correct = 30, RF seul correct = 27, p = 7.91e-01.

Matrice de confusion : `figs/finegrain_cm_0.png`.

## Streptococcus viridans (pneumoniae/oralis/mitis)

Effectifs : *Streptococcus mitis* 37, *Streptococcus oralis* 63, *Streptococcus pneumoniae* 56.

| Pipeline | balanced-acc | accuracy | F1-macro | MCC |
|---|---|---|---|---|
| VICRegL | 0.788 | 0.795 | 0.785 | 0.690 |
| RF-binned | 0.836 | 0.865 | 0.844 | 0.795 |

McNemar : VICRegL seul correct = 8, RF seul correct = 19, p = 5.43e-02.

Matrice de confusion : `figs/finegrain_cm_1.png`.

## Klebsiella (pneumoniae/oxytoca/variicola)

Effectifs : *Klebsiella oxytoca* 190, *Klebsiella pneumoniae* 555, *Klebsiella variicola* 55.

| Pipeline | balanced-acc | accuracy | F1-macro | MCC |
|---|---|---|---|---|
| VICRegL | 0.979 | 0.993 | 0.982 | 0.984 |
| RF-binned | 0.976 | 0.995 | 0.986 | 0.989 |

McNemar : VICRegL seul correct = 1, RF seul correct = 3, p = 6.25e-01.

Matrice de confusion : `figs/finegrain_cm_2.png`.

## Lecture

Ces tests mesurent la **séparabilité intrinsèque** d'espèces aux empreintes protéiques quasi identiques — un problème distinct de la robustesse inter-centres. Une performance imparfaite peut refléter soit une limite physique du MALDI, soit des labels Biotyper erronés (non corrigés faute de NGS). Pour une conclusion définitive au rang d'espèce, un sous-ensemble **labellisé par séquençage** serait nécessaire.
