#!/usr/bin/env python
"""Banc AMR : prédiction de résistance (VICRegL vs RF-binned), comparable à Weis et al.

Usage : python scripts/06_amr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.amr import run_amr


def main():
    out = run_amr()
    print("\n=== Banc AMR (AUROC) — intra-hôpital vs cross-hôpital ===")
    for r in out["results"]:
        print(f"\n  {r['species']} / {r['drug']}  (R: C={r['nR_C']}, B={r['nR_B']})")
        for pl in ("VICRegL", "RF-binned"):
            m = r[pl]
            print(f"    {pl:10s} | intra {m['in_domain']:.3f} | cross {m['cross']:.3f} "
                  f"| chute {m['drop']:+.3f}")
    print("\nRapport -> runs/pretrain/RESULTS_amr.md  |  Figure -> figs/amr_auroc.png")


if __name__ == "__main__":
    main()
