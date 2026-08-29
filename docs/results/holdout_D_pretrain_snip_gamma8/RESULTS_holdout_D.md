# Test de généralisation à un centre inédit : DRIAMS-D tenu à l'écart du pré-entraînement

## Résumé

L'encodeur VICRegL a été pré-entraîné en auto-supervision sur **B∪C uniquement** ; **DRIAMS-D n'a jamais été vu**, même pas en non-labellisé. On réutilise ce checkpoint gelé et on entraîne la sonde linéaire (et la baseline RF-binned) sur **B∪C poolés**, puis on teste sur **D**. C'est le test de centre inédit le plus strict du protocole (contrairement à B↔C où les deux centres étaient vus en non-labellisé).

Espace de classes : 10 espèces communes à B, C et D — *Escherichia coli*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Proteus mirabilis*, *Enterococcus faecalis*, *Pseudomonas aeruginosa*, *Staphylococcus epidermidis*, *Enterobacter cloacae*, *Citrobacter koseri*, *Enterobacter aerogenes*.

## Résultats — balanced accuracy (IC95 bootstrap)

| Condition | Type | Pipeline | n_train | n_test | Bal-acc [IC95] | Accuracy | F1-macro | MCC |
|---|---|---|---|---|---|---|---|---|
| (B+C)->D | hold-out | VICRegL | 6314 | 8797 | 0.942 [0.936, 0.948] | 0.923 | 0.872 | 0.909 |
| (B+C)->D | hold-out | RF-binned | 6314 | 8797 | 0.932 [0.925, 0.938] | 0.919 | 0.897 | 0.904 |
| B->D | hold-out | VICRegL | 2803 | 8797 | 0.933 [0.927, 0.939] | 0.907 | 0.846 | 0.892 |
| B->D | hold-out | RF-binned | 2803 | 8797 | 0.923 [0.916, 0.930] | 0.909 | 0.890 | 0.892 |
| C->D | hold-out | VICRegL | 3511 | 8797 | 0.934 [0.928, 0.940] | 0.913 | 0.862 | 0.897 |
| C->D | hold-out | RF-binned | 3511 | 8797 | 0.915 [0.908, 0.922] | 0.908 | 0.885 | 0.892 |
| (B+C) in-dom | in-domain | VICRegL | 4419 | 1895 | 0.994 [0.986, 0.999] | 0.997 | 0.994 | 0.996 |
| (B+C) in-dom | in-domain | RF-binned | 4419 | 1895 | 0.997 [0.992, 1.000] | 0.998 | 0.998 | 0.997 |

## Test apparié de McNemar (VICRegL vs RF-binned)

| Condition | VICRegL seul correct | RF seul correct | p-value | signif. |
|---|---|---|---|---|
| (B+C)->D | 150 | 114 | 3.12e-02 | * |
| B->D | 59 | 73 | 2.58e-01 | ns |
| C->D | 139 | 103 | 2.45e-02 | * |
| (B+C) in-dom | 2 | 4 | 6.88e-01 | ns |

## Lecture

Condition phare **(B+C)→D** (centre jamais vu) : VICRegL bal-acc **0.942** vs RF-binned **0.932**. À comparer aux conditions B↔C de `runs/pretrain/RESULTS.md`, où l'encodeur avait été exposé (non labellisé) aux deux centres : l'écart mesure la perte de généralisation à un centre réellement inédit.

Figure : `figs/holdout_D_balacc.png`.
