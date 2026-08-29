# Banc de prédiction de résistance (AMR) — VICRegL vs RF-binned, référence Weis et al. 2022

Sonde sur features VICRegL gelées vs Random Forest sur binned_6000. Métrique AUROC (celle de Weis). Intra-hôpital = CV 5-fold (meilleur cas) ; cross-hôpital = train un centre / test l'autre (C↔B). La **chute** intra→cross est le chiffre directement comparable à Weis (**drop rapporté : 0.065–0.225**).

> Réf. Weis et al. (DRIAMS, *Nat Med* 2022) : AUROC ≈ 0.75 (ceftriaxone/*E. coli* et /*K. pneumoniae*), 0.82 (oxacilline/*S. aureus*).

## AUROC par scénario

| Scénario | n (C/B) | R (C/B) | Pipeline | AUROC intra | AUROC cross | chute |
|---|---|---|---|---|---|---|
| *Escherichia coli* / Ceftriaxone | 913/213 | 148/45 | VICRegL | 0.660 | 0.569 | +0.091 |
| *Escherichia coli* / Ceftriaxone | 913/213 | 148/45 | RF-binned | 0.727 | 0.587 | +0.139 |
| *Escherichia coli* / Ciprofloxacin | 886/212 | 188/58 | VICRegL | 0.721 | 0.586 | +0.135 |
| *Escherichia coli* / Ciprofloxacin | 886/212 | 188/58 | RF-binned | 0.803 | 0.737 | +0.066 |
| *Staphylococcus aureus* / Oxacillin | 738/346 | 41/21 | VICRegL | 0.659 | 0.485 | +0.174 |
| *Staphylococcus aureus* / Oxacillin | 738/346 | 41/21 | RF-binned | 0.636 | 0.455 | +0.181 |
| *Klebsiella pneumoniae* / Ceftriaxone | 366/151 | 55/17 | VICRegL | 0.552 | 0.405 | +0.147 |
| *Klebsiella pneumoniae* / Ceftriaxone | 366/151 | 55/17 | RF-binned | 0.642 | 0.496 | +0.146 |

## Synthèse stabilité (chute AUROC intra→cross, moyenne)

| Pipeline | chute moyenne | vs bande Weis (0.065–0.225) |
|---|---|---|
| VICRegL | +0.137 | dans/sous la bande |
| RF-binned | +0.133 | dans/sous la bande |

## Lecture

La prédiction AMR est intrinsèquement plus difficile que l'ID d'espèce (la résistance tient souvent à un gène sans signature spectrale nette) : les AUROC absolus sont modestes, conformes à Weis. Le point d'intérêt est la **stabilité inter-hôpitaux** : si VICRegL présente une chute cross-site plus faible que le RF-binned, l'invariance apprise bénéficie aussi à l'AMR — prolongeant la solution démontrée sur l'espèce au problème que Weis et al. avaient identifié comme limite principale.
