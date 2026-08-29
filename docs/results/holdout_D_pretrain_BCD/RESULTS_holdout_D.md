# Test de généralisation à un centre inédit : DRIAMS-D tenu à l'écart du pré-entraînement

## Résumé

L'encodeur VICRegL a été pré-entraîné en auto-supervision sur **B∪C uniquement** ; **DRIAMS-D n'a jamais été vu**, même pas en non-labellisé. On réutilise ce checkpoint gelé et on entraîne la sonde linéaire (et la baseline RF-binned) sur **B∪C poolés**, puis on teste sur **D**. C'est le test de centre inédit le plus strict du protocole (contrairement à B↔C où les deux centres étaient vus en non-labellisé).

Espace de classes : 10 espèces communes à B, C et D — *Escherichia coli*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Proteus mirabilis*, *Enterococcus faecalis*, *Pseudomonas aeruginosa*, *Staphylococcus epidermidis*, *Enterobacter cloacae*, *Citrobacter koseri*, *Enterobacter aerogenes*.

## Résultats — balanced accuracy (IC95 bootstrap)

| Condition | Type | Pipeline | n_train | n_test | Bal-acc [IC95] | Accuracy | F1-macro | MCC |
|---|---|---|---|---|---|---|---|---|
| (B+C)->D | hold-out | VICRegL | 6314 | 8797 | 0.935 [0.929, 0.941] | 0.914 | 0.866 | 0.899 |
| (B+C)->D | hold-out | RF-binned | 6314 | 8797 | 0.932 [0.925, 0.938] | 0.919 | 0.897 | 0.904 |
| B->D | hold-out | VICRegL | 2803 | 8797 | 0.930 [0.924, 0.936] | 0.904 | 0.858 | 0.889 |
| B->D | hold-out | RF-binned | 2803 | 8797 | 0.923 [0.916, 0.930] | 0.909 | 0.890 | 0.892 |
| C->D | hold-out | VICRegL | 3511 | 8797 | 0.935 [0.928, 0.941] | 0.917 | 0.868 | 0.902 |
| C->D | hold-out | RF-binned | 3511 | 8797 | 0.915 [0.908, 0.922] | 0.908 | 0.885 | 0.892 |
| (B+C) in-dom | in-domain | VICRegL | 4419 | 1895 | 0.994 [0.985, 0.999] | 0.996 | 0.995 | 0.996 |
| (B+C) in-dom | in-domain | RF-binned | 4419 | 1895 | 0.997 [0.992, 1.000] | 0.998 | 0.998 | 0.997 |

## Test apparié de McNemar (VICRegL vs RF-binned)

| Condition | VICRegL seul correct | RF seul correct | p-value | signif. |
|---|---|---|---|---|
| (B+C)->D | 90 | 138 | 1.85e-03 | ** |
| B->D | 59 | 105 | 4.42e-04 | *** |
| C->D | 184 | 109 | 1.54e-05 | *** |
| (B+C) in-dom | 2 | 5 | 4.53e-01 | ns |

## Lecture

Condition phare **(B+C)→D** (centre jamais vu) : VICRegL bal-acc **0.935** vs RF-binned **0.932**. À comparer aux conditions B↔C de `runs/pretrain/RESULTS.md`, où l'encodeur avait été exposé (non labellisé) aux deux centres : l'écart mesure la perte de généralisation à un centre réellement inédit.

Figure : `figs/holdout_D_balacc.png`.
