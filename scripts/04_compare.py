#!/usr/bin/env python
"""Comparaison scientifique VICRegL vs RF-binned (intra- & cross-centre).

Régénère depuis le checkpoint existant (pas de ré-entraînement du backbone) :
  runs/pretrain/comparison.json, RESULTS.md, figs/*.png

Usage : python scripts/04_compare.py            # top_n=10
        python scripts/04_compare.py 15          # top_n espèces
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.analysis import run_comparison


def main(argv):
    top_n = int(argv[0]) if argv else 10
    # RUN_NAME = dossier du checkpoint à évaluer (def: pretrain)
    # CENTERS  = "seen,holdout" pour le LOCO (def: C,B)
    run_name = os.environ.get("RUN_NAME", "pretrain")
    centers = tuple(c.strip().upper() for c in os.environ.get("CENTERS", "C,B").split(","))
    out = run_comparison(run_name=run_name, centers=centers,
                         top_n=top_n, seed=0, n_boot=1000)
    print("\n=== Synthèse (balanced accuracy) ===")
    for r in out["results"]:
        v = r["preds"]["VICRegL"]["metrics"]["balanced_accuracy"]
        rf = r["preds"]["RF-binned"]["metrics"]["balanced_accuracy"]
        p = r["mcnemar"]["pvalue"]
        print(f"  {r['setting']:8s} [{r['kind']:12s}] VICRegL {v:.3f} | RF {rf:.3f} "
              f"| McNemar p={p:.1e}")
    print(f"\nRapport  -> runs/{run_name}/RESULTS.md")
    print(f"Figures  -> runs/{run_name}/figs/")
    print(f"JSON     -> runs/{run_name}/comparison.json")


if __name__ == "__main__":
    main(sys.argv[1:])
