# MS-VICRegL

Identification bactérienne MALDI-TOF (DRIAMS) par **CNN 1D auto-supervisé (VICRegL)**,
sans le pré-traitement DSP lourd habituel (wavelets, SNIP, ComBat, ...).

Approche volontairement opposée à [MSClassifPy](https://github.com/agodmer/MSclassifR) /
MSclassifR : au lieu de *supprimer* les artefacts de centre/instrument par un pipeline de
signal engineering, on apprend une représentation *invariante* à ces artefacts via
auto-supervision, en préservant la structure pic-à-pic (biomarqueurs) grâce au critère
**local** de VICRegL.

> Document de travail, résultats non publiés — partagé pour discussion.

## Architecture (vue simplifiée)

<img src="docs/figures/architecture_simplified.svg" alt="Architecture simplifiée du pipeline" width="620"/>

Deux vues augmentées d'un même spectre (simulateurs d'artefacts : dérive de calibration,
baseline, gain détecteur, bruit, dropout de pics) traversent le **même** encodeur
ResNet-1D (poids partagés, pas de stop-gradient/momentum — contrairement à BYOL/MoCo).
La sortie se sépare en un critère **global** (VICReg classique sur la représentation
moyennée) et un critère **local** (appariement des positions m/z + des features les plus
proches, top-γ), qui préserve l'information fine perdue par une représentation purement
globale.

Détail complet : [`docs/architecture.md`](docs/architecture.md).

## Comparaison à RF-binned

Sonde linéaire gelée sur l'encodeur VICRegL vs Random Forest sur spectres `binned_6000`
(format DRIAMS standard), mêmes labels, sur DRIAMS-B, -C, -D (Dryad
doi:10.5061/dryad.bzkh1899q). *Le `binned_6000` publié pour B était incomplet à 37% sur
Dryad (trou dans les données source) ; reconstruit intégralement via le pipeline DSP
validé de MSClassifPy avant ces chiffres — voir `scripts/11_rebuild_binned_B.py`.*

**Intra/cross-centre (B↔C) :**

| Condition | VICRegL | RF-binned |
|---|---|---|
| C→C | 0.980 | 0.997 |
| B→B | 0.989 | 0.994 |
| C→B | 0.988 | 0.992 |
| B→C | 0.949 | 0.994 |

![Balanced accuracy cross-centre](docs/figures/fig1_balanced_accuracy.png)

**Généralisation zéro-shot vers un centre jamais vu (DRIAMS-D)**, cinq variantes de
l'encodeur VICRegL (sonde seule, exposition SSL non-labellisée, + DANN, + CORAL,
+ prior par espèce inspiré de DALMA) :

![Score-card zero-shot D](docs/figures/scorecard_zeroshot_D.png)

| Méthode | (B+C)→D bal-acc |
|---|---|
| Sonde gelée, D jamais exposé | 0.886 |
| Exposition SSL non-labellisée de D (meilleure) | **0.915** |
| + DANN (domaine-adversarial) | 0.904 |
| + CORAL (alignement de moments) | 0.914 |
| + prior par espèce (inspiré DALMA) | 0.909 |
| RF-binned (référence, indépendant de l'encodeur) | **0.932** |

RF-binned est stable et devant sur toutes les conditions testées, intra-centre comme
zero-shot ; aucune des variantes VICRegL ne le rattrape.

## Comparaison à DALMA

[DALMA](https://arxiv.org/abs/2608.08182) (Garcia-Navarro et al., août 2026) — VAE à
encodeur partagé + décodeurs par centre + prior latent conditionné par espèce,
zero-shot cross-center sur un benchmark de 7 jeux de données, 3 pays.

![Comparaison DALMA](docs/figures/comparison_dalma.png)

Sur la configuration la plus proche de la nôtre (sources DRIAMS poolées → DRIAMS-D),
notre meilleure variante VICRegL (0.915) est dans la même fourchette que le chiffre
publié par DALMA (0.911), et RF-binned (0.932) reste légèrement au-dessus des deux.

Réserves : pas la même tâche (DALMA poole 5 sources sur 3 pays incl. MARISMa/RKI, 6
groupes d'espèces ; nous poolons B+C, 2 sites suisses du même écosystème DRIAMS que D,
10 espèces) ni la même méthode (notre chiffre vient d'un encodeur auto-supervisé sans
label au pré-entraînement + sonde linéaire ; DALMA est entièrement supervisé dès
l'entraînement). C'est aussi, pour DALMA comme pour nous, le point de comparaison le
plus *facile* de leur benchmark — leurs vrais gains apparaissent sur des transferts
inter-pays/instruments (RKI→MS-UMG : brut 0.732, DALMA 0.954) qu'on n'a pas testés.
Fait notable indépendant : les propres baselines DANN/CORAL de DALMA sont elles aussi
nettement dominées sur leurs sources difficiles — cohérent avec nos résultats DANN/CORAL
ci-dessus.

## Installation

```bash
# l'environnement conda base contient déjà torch (MPS) + sklearn
PY=/opt/miniconda3/bin/python
$PY -m pip install -r requirements.txt
```

## Données

Tarballs DRIAMS par centre dans `data/` (source : Dryad doi:10.5061/dryad.bzkh1899q,
non inclus dans ce dépôt — voir `.gitignore`).

## Utilisation

```bash
PY=/opt/miniconda3/bin/python

# Smoke-test (données synthétiques, ~20 s) — valide tout le pipeline sans DRIAMS
$PY -m pytest tests/ -v

# Ingestion (streaming tar -> data/processed/*.npy + meta.parquet)
$PY scripts/01_ingest.py

# Pré-entraînement SSL (MPS)
$PY scripts/02_pretrain.py                                    # B + C
DOMAIN_COEFF=1.0 DOMAIN_METHOD=coral $PY scripts/02_pretrain.py B C D   # + invariance de centre (CORAL)
SPECIES_COEFF=1.0 $PY scripts/02_pretrain.py B C D            # + prior par espèce

# Évaluation cross-centre (sonde VICRegL vs RF-binned)
$PY scripts/03_evaluate.py
$PY scripts/08_holdout_D.py --run pretrain_BCD                # centre D jamais vu
$PY scripts/10_domain_auroc.py pretrain_BCD pretrain_BCD_coral # diagnostic AUROC domaine
```

## Structure

| Fichier | Rôle |
|---|---|
| `ms_vicregl/config.py` | hyper-paramètres |
| `ms_vicregl/ingest.py` | tar streaming → resample → TIC → memmap + labels |
| `ms_vicregl/augment.py` | simulateurs d'artefacts (vues + coordonnées m/z) |
| `ms_vicregl/model.py` | ResNet-1D + expander global + projecteur local + modules d'invariance |
| `ms_vicregl/loss.py` | VICReg + critère local + DANN/CORAL/SpeciesPrior |
| `ms_vicregl/dataset.py` | datasets torch (vues SSL / labellisé) |
| `ms_vicregl/pretrain.py` | boucle SSL + extraction de features |
| `ms_vicregl/evaluate.py`, `analysis.py` | sondes + comparaisons + figures |
| `scripts/` | points d'entrée numérotés (ingestion → pré-entraînement → évaluations) |
| `docs/architecture.md` | schéma détaillé, fidélité au papier VICReg/VICRegL |

## Notes M3 Pro / MPS

- Tout tourne en `float32` sur MPS (le fp16/bf16 MPS est partiel).
- `num_workers=0` forcé sur MPS (DataLoader plus stable/rapide qu'avec des workers).
- Expander réduit à 2048 (vs 8192 du papier VICReg) pour tenir dans 18 Go de RAM.
- Les `.npy` sont chargés en `mmap_mode='r'` pour ne pas saturer la RAM.
