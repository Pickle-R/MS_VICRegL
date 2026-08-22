#!/usr/bin/env python
"""Évaluation cross-centre : sonde VICRegL vs baseline RF binned_6000.

Usage:
    python scripts/03_evaluate.py            # centres C,B  top_n=10
    python scripts/03_evaluate.py C B 15     # centres + top_n espèces
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.evaluate import run_full


def main(argv):
    centers = ("C", "B")
    top_n = 10
    if len(argv) >= 2:
        centers = (argv[0].upper(), argv[1].upper())
    if len(argv) >= 3:
        top_n = int(argv[2])
    run_name = os.environ.get("RUN_NAME", "pretrain")   # ex: RUN_NAME=loco_holdout_B
    run_full(run_name=run_name, centers=centers, top_n=top_n)


if __name__ == "__main__":
    main(sys.argv[1:])
