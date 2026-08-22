# Résultats

Toutes les évaluations : sonde linéaire gelée sur l'encodeur VICRegL (`StandardScaler` +
`LogisticRegression`) vs Random Forest (300 arbres, `class_weight="balanced"`, non
finement tuné) sur spectres `binned_6000`, mêmes splits, mêmes labels espèce (Biotyper/
MALDI, DRIAMS). Centres : B (Basel-Land, 5708 spectres/337 espèces), C (Aarau, 4696/121),
D (Viollier, 10436/54 — sous-ensemble labellisé de ~74k spectres bruts).

## RESULT 1 — Stabilité intra/cross-centre

Sonde entraînée/testée sur 4 conditions (intra-centre et cross-centre B↔C) : VICRegL
uniformément **0.95–0.99** (écart 0.04) contre RF-binned erratique **0.685–0.997**
(écart 0.31), qui s'effondre même en intra-centre (B→B = 0.765). VICRegL ~8× plus stable.

## RESULT 2 — Discrimination fine de sous-espèces proches

5-fold CV sur B∪C, encodeur gelé, 3 groupes difficiles (McNemar p<0.01) :

- *Enterobacter cloacae* complex (5 classes) : VICRegL 0.669 vs RF 0.438 (RF fusionne
  tout dans une seule classe).
- *Streptococcus viridans* (3 classes) : VICRegL 0.788 vs RF 0.688 (RF confond
  *S. oralis*→*S. mitis*).
- *Klebsiella* (3 classes) : VICRegL 0.979 vs RF 0.932.

**Limite de données à connaître :** les labels DRIAMS proviennent du Biotyper (MALDI),
pas d'un séquençage. 0 *Shigella* dans B+C — le Biotyper ne peut pas la séparer d'*E.
coli*, ce qui est la preuve même de la circularité : on ne peut pas revendiquer "battre
le MALDI" avec des labels produits par le MALDI. Il faudrait des données appariées
WGS (design MSclassifR/Godmer) pour trancher cette question spécifique.

## RESULT 3 — AMR (négatif, informatif)

Comparaison à Weis et al. 2022 (*Nat Med*), *K. pneumoniae*, encodeur gelé, AUROC, 4
scénarios C↔B :

- E. coli/ceftriaxone : RF 0.739 ≈ Weis 0.75 (valide notre pipeline RF) ; VICRegL 0.660.
- Chute cross-site identique pour les deux pipelines (+0.137 en moyenne), dans la
  fourchette documentée par Weis (0.065–0.225) → pas de gain de stabilité sur l'AMR.

**Interprétation :** l'invariance apprise (notamment le dropout de pics simulé) enseigne
à l'encodeur à ignorer exactement le signal faible/local dont l'AMR a besoin. L'invariance
de représentation est **spécifique à la tâche**, pas un bien universel. Pistes pour
adapter VICRegL à l'AMR : ne pas geler l'encodeur (fine-tuning), augmentations
préservant l'AMR, apprentissage multi-tâche.

## RESULT 4 — Généralisation stricte à un centre jamais vu (DRIAMS-D)

Encodeur pré-entraîné sur B∪C **uniquement** (D n'existait pas encore au moment de ce
pré-entraînement) ; sonde + RF entraînés sur B∪C poolés, testés sur D — le test de centre
inédit le plus strict.

- (B+C)→D : VICRegL **0.886** vs RF-binned **0.935** (RF gagne, McNemar p≈3e-81) —
  inversion du résultat B↔C.
- Décomposition (pas du sur-apprentissage classique — sonde linéaire, IC serrés) :
  plafond D→D en intra-domaine = 0.914 (−0.074 vs B+C intra-domaine, ~73% de l'écart =
  plafond de représentation, D est intrinsèquement plus dur pour cet encodeur) ;
  0.914→0.886 (−0.028, ~27% = transfert de frontière de la sonde).
- Classifieur de domaine entraîné sur les features gelées, (B+C) vs D : AUROC = **0.997**
  → l'invariance a échoué, les features encodent encore presque parfaitement le centre.

## RESULT 5 — L'exposition non-labellisée à D suffit-elle ?

Pré-entraînement SSL sur B∪C∪D **non-labellisé** (D vu, mais sans aucun label).

- (B+C)→D : 0.886 → **0.915** (+0.029, McNemar p=1.6e-50).
- Plafond D→D : 0.914 → 0.931 (+0.017). Écart de transfert (plafond−transfert) :
  0.028 → 0.016.
- AUROC domaine (B+C) vs D : 0.997 → **0.999** (INCHANGÉ).

**Interprétation :** l'exposition non-labellisée est un vrai gain, gratuit (pas besoin de
labels), mais (i) n'atteint pas la parité (plafond D toujours < B/C ~0.99 et < RF 0.935),
et (ii) l'AUROC domaine reste ~1.0 → le gain vient de features D plus riches, **pas**
d'invariance de centre. Le mécanisme d'invariance reste cassé.

## RESULT 6 — DANN (domaine-adversarial)

Tête de classification de centre derrière un gradient-reversal layer, ajoutée à la perte
sur `repr_` (le même tenseur évalué en aval).

- (B+C)→D : **0.904** — pire que l'exposition seule (0.915).
- AUROC domaine hors-ligne : 0.9979 (sans DANN) → **0.9976** (quasi inchangé).

**Piège méthodologique découvert :** en cours d'entraînement, `domain_acc` (précision du
classifieur en ligne) chutait de 0.76 à ~0.50, signal en apparence encourageant. Mais une
sonde fraîche entraînée *après coup* sur les features gelées finales retrouve le centre
presque parfaitement. Explication : le classifieur en ligne est à la fois la cible
adversariale de l'encodeur *et* une cible mouvante — il n'a jamais vraiment convergé.
**`domain_acc` qui baisse en cours d'entraînement n'est pas une preuve fiable
d'invariance ; seule la sonde hors-ligne sur features gelées (`scripts/10_domain_auroc.py`)
compte.**

## RESULT 7 — CORAL

Alignement direct moyenne+covariance entre centres présents dans le batch, sans réseau
auxiliaire ni jeu adversarial — évite structurellement le piège de RESULT 6.

- CORAL converge proprement sur son propre objectif (perte → ~0 dès l'époque 20-30/120).
- (B+C)→D : **0.914** — statistiquement identique à l'exposition seule (0.915).
- AUROC domaine : 0.9979 → **0.9983** (inchangé, voire marginalement plus séparable).

**Deux mécanismes structurellement différents (adversarial vs alignement déterministe de
moments) échouent au même endroit.** Ce n'est plus "un défaut de méthode" — l'information
qui permet à une sonde de séparer D survit dans une structure que ni l'un ni l'autre ne
touche (probablement des différences systématiques de bas niveau — calibration
d'instrument — encodées dès les premières couches convolutives).

## RESULT 8 — Prior par espèce (inspiré DALMA)

[DALMA](https://arxiv.org/abs/2608.08182) (Garcia-Navarro et al., août 2026) montre par
ablation que son levier principal de transférabilité est un **prior latent conditionné
par espèce** (supervisé), pas ses décodeurs par centre. On a porté ce levier — pas les
décodeurs (incompatibles architecturalement : VICRegL n'a pas de décodeur/reconstruction)
— sous forme d'un prototype appris par espèce vers lequel `repr_` est tiré (MSE,
adaptation déterministe : notre encodeur n'est pas variationnel, pas de vraie KL
possible).

- Dynamique d'entraînement différente et encourageante : perte du prior → ~0.01 ;
  `var`/`inv` **meilleurs** que la référence tout du long (contrairement à DANN/CORAL).
- (B+C)→D : **0.909** — toujours pas d'amélioration sur l'exposition seule (0.915).
- Note méthodologique : ce test inclut les vrais labels d'espèce de D pendant le
  pré-entraînement — plus favorable que le protocole strict de DALMA (leur centre cible
  ne reçoit aucune supervision). Un avantage donné au terme, qui n'a quand même pas
  suffi.
- Raison probable : notre prior est une traction dure vers un point unique, alors que
  celui de DALMA est une KL molle vers une gaussienne à variance **apprise** par espèce —
  bien plus permissive pour des espèces à forte hétérogénéité biologique (sous-espèces,
  souches multiples). Reproduire fidèlement demanderait un encodeur variationnel
  (têtes moyenne+variance), non tenté ici.

## Score-card final (B+C)→D

| Méthode | Bal-acc |
|---|---|
| Sonde gelée, D jamais exposé | 0.886 |
| Exposition SSL non-labellisée (meilleure) | **0.915** |
| DANN | 0.904 |
| CORAL | 0.914 |
| SpeciesPrior | 0.909 |
| RF-binned (référence) | **0.935** |

Trois mécanismes structurellement différents (adversarial, alignement déterministe de
moments, traction supervisée vers un prototype) échouent tous à battre la simple
exposition non-labellisée. Le verrou ressemble à quelque chose de structurel au transfert
(B+C)→D plutôt qu'à un mauvais choix de régulateur.

## Comparaison à DALMA

Sur la configuration la plus proche de la nôtre dans DALMA (sources DRIAMS poolées →
DRIAMS-D, leur tableau 2, ligne "All→DRIAMS-D") : baseline brute = DALMA = **0.911**.
Notre meilleure variante (0.915) est dans la même fourchette — mais :

- **Pas la même tâche** : DALMA poole 5 sources hétérogènes sur 3 pays (DRIAMS-A/B/C +
  MARISMa Espagne + RKI Allemagne) sur 6 groupes d'espèces ; nous poolons B+C (2 sites
  suisses du même écosystème DRIAMS que D) sur 10 espèces.
- **Pas la même méthode** : notre chiffre vient d'un encodeur auto-supervisé (aucun
  label pendant le pré-entraînement pour la variante "exposition seule") + sonde
  linéaire ; DALMA est entièrement supervisé (prior espèce + décodeurs par centre dès
  l'entraînement).
- **C'est le point le plus facile du benchmark DALMA.** Leurs vrais gains apparaissent
  sur des transferts bien plus durs (RKI→MS-UMG : baseline brute 0.732, DALMA 0.954) —
  scénarios qu'on n'a pas testés. Sur ce point précis, DALMA non plus ne bat pas son
  propre baseline — ce n'est donc pas qu'on égale leur force, c'est que personne ne
  gagne beaucoup sur ce transfert-là spécifiquement.
- Fait notable indépendant : les propres baselines DANN/CORAL de DALMA sont nettement
  dominées sur leurs sources difficiles (RKI→MS-UMG : DANN 0.547, CORAL 0.241 vs DALMA
  0.954) — ça corrobore RESULT 6-7, DANN/CORAL comme mécanisme isolé d'invariance ne sont
  pas juste faibles dans notre implémentation, c'est un pattern indépendamment répliqué.

## Pistes non testées

- Prior par espèce **variationnel** (vraie KL, DALMA-fidèle) plutôt que la version
  déterministe actuelle.
- Fine-tuning de l'encodeur au lieu d'une sonde linéaire gelée.
- Correction test-time (recalibration BN / CORAL à l'inférence sur les statistiques du
  centre cible) — non tenté, bien que le plafond D (0.914-0.931) reste sous RF (0.935).
- Ingestion de DRIAMS-A (jamais faite — corpus SSL actuel limité à B+C(+D), ~10-21k
  spectres, loin des ~145k initialement prévus).
