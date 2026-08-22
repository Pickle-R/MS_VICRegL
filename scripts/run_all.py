#!/usr/bin/env python
"""Run complet enchaîné : pré-entraînement SSL (B+C) puis évaluation cross-centre.

Variables d'environnement :
    PROFILE  (def. light)  'light' (modèle réduit + cooldown) ou 'full' (profil complet)
    EPOCHS   (def. 80)     nombre d'époques SSL
    COOLDOWN (def. 8)      pause refroidissement (s), profil light
    COOLDOWN_EVERY (def. 5) ... toutes les N époques
    TOPN     (def. 10)     nb d'espèces (communes B/C, plus fréquentes) pour l'éval

Usage : PROFILE=light EPOCHS=80 python scripts/run_all.py
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.config import CFG, RUNS, light_config, medium_config
from ms_vicregl.dataset import load_centers
from ms_vicregl.pretrain import pretrain
from ms_vicregl.evaluate import run_full


def main():
    profile = os.environ.get("PROFILE", "light")
    default_ep = {"light": 80, "medium": 120, "full": 200}.get(profile, 80)
    epochs = int(os.environ.get("EPOCHS", default_ep))
    top_n = int(os.environ.get("TOPN", 10))
    cooldown = float(os.environ.get("COOLDOWN", 6))
    cooldown_every = int(os.environ.get("COOLDOWN_EVERY", 5))

    if profile == "light":
        cfg = light_config(epochs=epochs, cooldown_s=cooldown, cooldown_every=cooldown_every)
    elif profile == "medium":
        cfg = medium_config(epochs=epochs, cooldown_s=cooldown, cooldown_every=cooldown_every)
    else:  # full
        cfg = replace(CFG, train=replace(CFG.train, epochs=epochs, num_workers=0,
                                         cooldown_s=cooldown, cooldown_every=cooldown_every))

    centers = ["C", "B"]
    X, _, meta = load_centers(centers)
    print(f"=== PRÉ-ENTRAÎNEMENT [profil={profile}] ({epochs} ép) sur {centers} : "
          f"{X.shape[0]} spectres, {meta.species.nunique()} espèces | "
          f"cooldown {cfg.train.cooldown_s}s/{cfg.train.cooldown_every}ép ===", flush=True)

    t0 = time.time()
    pretrain(X, cfg=cfg, run_name="pretrain")
    print(f"[run_all] pré-entraînement: {(time.time()-t0)/60:.1f} min", flush=True)

    print("\n=== ÉVALUATION CROSS-CENTRE ===", flush=True)
    results = run_full(run_name="pretrain", centers=("C", "B"), top_n=top_n)

    out = RUNS / "pretrain" / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[run_all] résultats -> {out}")
    print(f"[run_all] TOTAL: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
