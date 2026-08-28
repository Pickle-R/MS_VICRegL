#!/usr/bin/env python
"""Pré-entraînement SSL VICRegL-1D sur les centres ingérés (B + C par défaut).

Usage:
    python scripts/02_pretrain.py             # B + C
    python scripts/02_pretrain.py B C
    EPOCHS=50 python scripts/02_pretrain.py   # override rapide du nb d'époques
    DOMAIN_COEFF=1.0 python scripts/02_pretrain.py B C D               # + DANN (défaut)
    DOMAIN_COEFF=1.0 DOMAIN_METHOD=coral python scripts/02_pretrain.py B C D   # + CORAL
    SPECIES_COEFF=1.0 python scripts/02_pretrain.py B C D              # + prior par espèce (DALMA-inspiré)
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_vicregl.config import CFG, PROCESSED
from ms_vicregl.dataset import (load_centers, load_centers_with_domain,
                                load_centers_with_domain_species)
from ms_vicregl.pretrain import pretrain


def main(argv):
    centers = [a.upper() for a in argv] or ["C", "B"]
    centers = [c for c in centers if (PROCESSED / f"{c}_X.npy").exists()]
    if not centers:
        print("Aucun centre ingéré. Lance d'abord scripts/01_ingest.py")
        return

    domain_coeff = float(os.environ.get("DOMAIN_COEFF", "0.0"))
    domain_method = os.environ.get("DOMAIN_METHOD", "dann")
    species_coeff = float(os.environ.get("SPECIES_COEFF", "0.0"))
    variant = os.environ.get("VARIANT", "")   # "_snip" pour l'entrée SNIP-corrigée
    species = None
    if species_coeff > 0:
        X, _, meta, domain, species, _ = load_centers_with_domain_species(centers)
        if domain_coeff == 0:
            domain = None
    elif domain_coeff > 0 and len(centers) > 1:
        X, _, meta, domain = load_centers_with_domain(centers)
    else:
        X, _, meta = load_centers(centers, variant=variant)
        domain = None
    tags = []
    if domain is not None:
        tags.append(f"{domain_method.upper()} domain_coeff={domain_coeff}")
    if species is not None:
        tags.append(f"SpeciesPrior species_coeff={species_coeff}")
    print(f"Pré-entraînement sur {centers} : {X.shape[0]} spectres, dim {X.shape[1]}"
          + (" | " + " | ".join(tags) if tags else ""))

    cfg = CFG
    overrides = {}
    if os.environ.get("EPOCHS"):
        overrides["train"] = replace(CFG.train, epochs=int(os.environ["EPOCHS"]))
    if domain_coeff > 0 or species_coeff > 0:
        overrides["loss"] = replace(CFG.loss, domain_coeff=domain_coeff, domain_method=domain_method,
                                    species_coeff=species_coeff)
    if overrides:
        cfg = replace(CFG, **overrides)
    run_name = os.environ.get("RUN_NAME", "pretrain")   # ex: RUN_NAME=loco_holdout_B
    pretrain(X, cfg=cfg, run_name=run_name, domain=domain, species=species)


if __name__ == "__main__":
    main(sys.argv[1:])
