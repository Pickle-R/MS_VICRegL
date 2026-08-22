#!/usr/bin/env python
"""Ablation « exposition SSL du centre cible » : pré-entraîne l'encodeur sur
B∪C∪D *non-labellisé* (D vu sans labels), avec la MÊME architecture/planning que
runs/pretrain (profil medium, 120 ep) pour une comparaison à iso-config.

Puis évaluation identique au hold-out : sonde + RF entraînés sur B+C poolés,
testés sur D (via scripts/08_holdout_D.py --run pretrain_BCD).

But : voir si *exposer D en non-labellisé* fait remonter le plafond de
représentation sur D (0.914 avec l'encodeur B∪C) vers celui de B/C (~0.98),
donc si « ajouter un centre » ne coûte que du SSL sans labels.

RESULT 5 a montré que l'exposition seule ne répare PAS l'invariance : AUROC
classifieur de domaine (B+C)vs D restait ~0.998 sur les features gelées après
pretrain_BCD. RESULT 6 a testé DANN (domain_coeff=1.0) : AUROC hors-ligne
quasiment inchangé (0.9976) et (B+C)->D bal-acc même LÉGÈREMENT pire (0.904 vs
0.915 sans DANN) — le `domain_acc` en ligne descendait bien vers le hasard
pendant l'entraînement, mais c'était un faux signal (discriminateur en ligne
co-adapté avec l'encodeur sous le GRL, jamais vraiment convergé ; cf.
ms_vicregl/loss.py::domain_adversarial_loss et MEMORY). RESULT 7 a testé CORAL :
même verdict (AUROC 0.9983), malgré une convergence propre de son propre objectif
— l'invariance de centre poussée directement (adversarial ou alignement de moments)
ne suffit pas.

SPECIES_COEFF active une piste différente, inspirée de DALMA (Garcia-Navarro et al.
2026, cf. DALMA.pdf) : au lieu de pousser l'invariance de centre, on structure
directement repr_ par identité biologique — un prototype appris par espèce
(SpeciesPrior, ms_vicregl/model.py) vers lequel repr_ est tiré, commun à tous les
centres. Dans l'ablation de DALMA, c'est ce levier supervisé (pas leurs décodeurs
par centre) qui porte l'essentiel de la transférabilité zero-shot.

Sortie -> runs/pretrain_BCD/ckpt.pt (ou runs/<RUN_NAME> si surchargé)

Usage:
    python scripts/09_ssl_BCD.py                                        # ablation d'origine (sans terme additionnel)
    DOMAIN_COEFF=1.0 DOMAIN_METHOD=coral python scripts/09_ssl_BCD.py    # CORAL (RESULT 7 : négatif)
    DOMAIN_COEFF=1.0 python scripts/09_ssl_BCD.py                       # DANN (RESULT 6 : négatif)
    SPECIES_COEFF=1.0 python scripts/09_ssl_BCD.py                      # prior par espèce (DALMA-inspiré)
    SPECIES_COEFF=1.0 RUN_NAME=pretrain_BCD_species python scripts/09_ssl_BCD.py
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.config import PROCESSED, medium_config
from ms_vicregl.dataset import (load_centers, load_centers_with_domain,
                                load_centers_with_domain_species)
from ms_vicregl.pretrain import pretrain

CENTERS = ["B", "C", "D"]


def main():
    centers = [c for c in CENTERS if (PROCESSED / f"{c}_X.npy").exists()]
    missing = [c for c in CENTERS if c not in centers]
    if missing:
        print(f"[SSL BCD] centres manquants (non ingérés) : {missing} — abandon.")
        return

    domain_coeff = float(os.environ.get("DOMAIN_COEFF", "0.0"))
    domain_method = os.environ.get("DOMAIN_METHOD", "dann")
    species_coeff = float(os.environ.get("SPECIES_COEFF", "0.0"))

    name_bits = []
    if domain_coeff > 0:
        name_bits.append(domain_method)
    if species_coeff > 0:
        name_bits.append("species")
    default_run = f"pretrain_BCD_{'_'.join(name_bits)}" if name_bits else "pretrain_BCD"
    run_name = os.environ.get("RUN_NAME", default_run)

    species = None
    if species_coeff > 0:
        X, _, _, domain, species, _ = load_centers_with_domain_species(centers)
        if domain_coeff == 0:
            domain = None
    elif domain_coeff > 0:
        X, _, _, domain = load_centers_with_domain(centers)
    else:
        X, _, _ = load_centers(centers)
        domain = None

    cfg = medium_config()  # iso-config avec runs/pretrain
    if domain_coeff > 0 or species_coeff > 0:
        cfg = replace(cfg, loss=replace(cfg.loss, domain_coeff=domain_coeff, domain_method=domain_method,
                                        species_coeff=species_coeff))
    tags = []
    if domain is not None:
        tags.append(f"{domain_method.upper()} domain_coeff={domain_coeff}")
    if species is not None:
        tags.append(f"SpeciesPrior species_coeff={species_coeff}")
    print(f"[SSL BCD] pré-entraînement sur {centers} : {X.shape[0]} spectres "
          f"non-labellisés, dim {X.shape[1]} | profil medium ({cfg.train.epochs} ep, "
          f"cooldown {cfg.train.cooldown_s}s/{cfg.train.cooldown_every}ep)"
          + (" | " + " | ".join(tags) if tags else ""))
    pretrain(X, cfg=cfg, run_name=run_name, ckpt_every=15, domain=domain, species=species)
    print(f"\n[SSL BCD] terminé -> runs/{run_name}/ckpt.pt")
    print(f"Évalue ensuite :  python scripts/08_holdout_D.py --run {run_name}")


if __name__ == "__main__":
    main()
