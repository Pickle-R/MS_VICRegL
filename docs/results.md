# Résultats

Toutes les évaluations : sonde linéaire gelée sur l'encodeur VICRegL (`StandardScaler` +
`LogisticRegression`) vs Random Forest (300 arbres, `class_weight="balanced"`, non
finement tuné) sur spectres `binned_6000`, mêmes splits, mêmes labels espèce (Biotyper/
MALDI, DRIAMS). Centres : B (Basel-Land, 5708 spectres/337 espèces), C (Aarau, 4696/121),
D (Viollier, 10436/54 — sous-ensemble labellisé de ~74k spectres bruts).

## RESULT 1 & 2 — RÉVISÉS le 2026-08-22 (bug de données)

**Ce qui suit dans cette section décrit les résultats ORIGINAUX (calculés sur données
B corrompues). Voir "Correction de données" ci-dessous pour les valeurs actuelles.**
Gardé pour traçabilité, mais **ne pas citer ces chiffres tels quels.**

### RESULT 1 (original, invalidé) — Stabilité intra/cross-centre

Sonde entraînée/testée sur 4 conditions (intra-centre et cross-centre B↔C) : VICRegL
uniformément **0.95–0.99** (écart 0.04) contre RF-binned erratique **0.685–0.997**
(écart 0.31), qui s'effondre même en intra-centre (B→B = 0.765). VICRegL ~8× plus stable.

### RESULT 2 (original, invalidé) — Discrimination fine de sous-espèces proches

5-fold CV sur B∪C, encodeur gelé, 3 groupes difficiles (McNemar p<0.01) :

- *Enterobacter cloacae* complex (5 classes) : VICRegL 0.669 vs RF 0.438 (RF fusionne
  tout dans une seule classe).
- *Streptococcus viridans* (3 classes) : VICRegL 0.788 vs RF 0.688 (RF confond
  *S. oralis*→*S. mitis*).
- *Klebsiella* (3 classes) : VICRegL 0.979 vs RF 0.932.

**Limite de données à connaître (indépendante du bug ci-dessous) :** les labels DRIAMS
proviennent du Biotyper (MALDI), pas d'un séquençage. 0 *Shigella* dans B+C — le
Biotyper ne peut pas la séparer d'*E. coli*, ce qui est la preuve même de la
circularité : on ne peut pas revendiquer "battre le MALDI" avec des labels produits par
le MALDI. Il faudrait des données appariées WGS (design MSclassifR/Godmer) pour
trancher cette question spécifique.

## Correction de données (2026-08-22) — RESULT 1 & 2 recalculés

**Découverte :** le tarball DRIAMS-B publié sur Dryad ne contient `binned_6000` que pour
2386 des 6416 spectres bruts (37% — vérifié directement dans l'archive `.tar.gz`, ce
n'est pas un téléchargement corrompu). `ingest.py` remplaçait silencieusement les
spectres manquants par des vecteurs zéro (`bin_d.get(c, np.zeros(...))`). Résultat :
tous les résultats RF-binned impliquant B (RESULT 1, RESULT 2, une partie de RESULT 3)
ont tourné avec ~58% de B en bruit pur côté RF. **VICRegL n'était pas affecté** — son
entrée (`X.npy`) vient du `/raw/`, toujours complet à 100% pour B.

**Correction :** reconstruction complète de `B_Xbin.npy` (les 5708 spectres, pas
seulement les 3322 manquants — pour une méthodologie homogène à l'intérieur du centre)
à partir du `/raw/` de B via le pipeline DSP déjà validé bit-à-bit contre R de
[MSClassifPy](https://github.com/agodmer/MSclassifR) : `sqrt` → lissage ondelettes →
correction de baseline SNIP (25 itérations) → calibration TIC → binning 3 Da
(2000-20000 Da, mêmes bords que la grille VICRegL). Script :
`scripts/11_rebuild_binned_B.py`. ~1.75 min de calcul pour les 5708 spectres.

### RESULT 1 recalculé

| Condition | VICRegL | RF-binned (avant) | RF-binned (après) |
|---|---|---|---|
| C→C | 0.980 | ~0.997 | 0.997 |
| B→B | 0.989 | 0.765 | **0.994** |
| C→B | 0.988 | erratique | **0.992** |
| B→C | 0.949 | erratique | **0.994** |

**Inversion complète.** RF-binned n'était pas erratique — c'était l'artefact du bug.
Corrigé, RF-binned est stable et légèrement **supérieur** à VICRegL sur les 4
conditions (écart ≤0.01). `docs/figures/fix_before_after_result1.png` montre l'effet
sur B→B, le seul point dont la valeur exacte "avant" a été conservée (le JSON original
a été écrasé par la ré-évaluation avant qu'on pense à le sauvegarder).

### RESULT 2 recalculé

| Groupe | VICRegL | RF (avant) | RF (après) | McNemar p (après) |
|---|---|---|---|---|
| *Enterobacter cloacae* complex | 0.669 | 0.438 | 0.443 | 0.79 (non signif.) |
| *Streptococcus viridans* | 0.788 | 0.688 | **0.836** (RF gagne) | 0.054 |
| *Klebsiella* | 0.979 | 0.932 | 0.976 | 0.625 (non signif.) |

**Plus aucun résultat statistiquement significatif.** RF bat même VICRegL sur
*Streptococcus viridans*. La revendication "VICRegL bat RF sur les 3 groupes, p<0.01"
ne tient plus.

### RESULT 3 — peu affecté

Les sous-ensembles B utilisés pour l'AMR étaient petits (17-58 isolats R par
scénario) et le résultat directionnel (RF > VICRegL) ne change pas ; seuls les
chiffres bougent légèrement (E. coli/ceftriaxone RF 0.739→0.727, toujours ≈ Weis
0.75). Voir tableau mis à jour dans le README.

### Ce que ça ne change pas

RESULT 4-8 (généralisation zéro-shot vers D) sont **inchangés côté VICRegL** (n'utilise
jamais `Xbin`) ; RF-binned de référence bouge marginalement (0.935→0.932, dans le
bruit). La conclusion "RF gagne le zero-shot, aucune des 5 méthodes SSL ne le bat" tient
toujours, et est même légèrement renforcée par la correction (l'écart n'était pas un
artefact de données côté zero-shot).

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
| RF-binned (référence, corrigé) | **0.932** |

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
  centre cible) — non tenté, bien que le plafond D (0.914-0.931) reste sous RF (0.932).
- Appliquer le même pipeline DSP lourd (celui utilisé pour reconstruire B_Xbin) en
  entrée de VICRegL lui-même, au lieu du quasi-brut actuel — untest direct de si le
  préprocessing aide aussi l'encodeur SSL sur le zero-shot, au prix probable d'une perte
  de structure locale fine (RESULT 2).
- Ingestion de DRIAMS-A (jamais faite — corpus SSL actuel limité à B+C(+D), ~10-21k
  spectres, loin des ~145k initialement prévus).
