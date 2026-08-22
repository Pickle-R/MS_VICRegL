#!/usr/bin/env python
"""Discrimination fine d'espèces proches (Enterobacter complex, Streptococcus viridans,
Klebsiella variicola) avec l'encodeur gelé. VICRegL vs RF-binned, CV 5-fold.

Usage : python scripts/05_finegrain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.finegrain import run_finegrain


def main():
    out = run_finegrain(run_name="pretrain", seed=0, n_splits=5)
    print("\n=== Discrimination fine (balanced accuracy, CV 5-fold) ===")
    for g in out["groups"]:
        v = g["VICRegL"]["balanced_accuracy"]; r = g["RF-binned"]["balanced_accuracy"]
        print(f"  {g['group'][:42]:42s} | n={g['n']:4d} | VICRegL {v:.3f} | RF {r:.3f} "
              f"| p={g['mcnemar']['pvalue']:.1e}")
    print("\nRapport -> runs/pretrain/RESULTS_finegrain.md")
    print("Figures -> runs/pretrain/figs/finegrain_*.png")


if __name__ == "__main__":
    main()
