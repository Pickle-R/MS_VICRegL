#!/usr/bin/env python
"""Leave-one-center-out (LOCO) : exclut un centre du PRÉ-ENTRAÎNEMENT, puis l'évalue.

Pourquoi : dans runs/pretrain/, l'encodeur a vu B ET C (non labellisés) au
pré-entraînement. C'est donc une *adaptation de domaine avec cible disponible*, pas
un centre inédit. Ici on entraîne l'encodeur SSL en EXCLUANT totalement le centre
`--holdout`, puis on l'évalue dessus : test strict de généralisation à un centre
JAMAIS vu (même pas en non-labellisé).

La ligne clé du rapport est la condition cross-center  <seen>->'<holdout>'  :
l'encodeur n'a jamais vu le centre `holdout`. On compare sa balanced accuracy à
celle obtenue dans runs/pretrain/ (où le centre était vu) pour quantifier la perte
de généralisation à un centre nouveau.

Sorties -> runs/loco_holdout_<H>/   (n'écrase PAS runs/pretrain/)

Usage:
    # held-out = B, pré-entraînement sur les autres centres ingérés (ici C seul)
    python scripts/07_loco.py --holdout B

    # held-out = B, pré-entraînement explicite multi-centres (recommandé si D/A ingérés)
    python scripts/07_loco.py --holdout B --pretrain C D

    EPOCHS=80 python scripts/07_loco.py --holdout B --pretrain C D
"""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.analysis import run_comparison
from ms_vicregl.config import CFG, PROCESSED, RUNS
from ms_vicregl.dataset import load_centers
from ms_vicregl.pretrain import pretrain


def _ingested_centers():
    return sorted(p.name[:-6] for p in PROCESSED.glob("*_X.npy"))


def main(argv):
    ap = argparse.ArgumentParser(description="Leave-one-center-out pour MS-VICRegL")
    ap.add_argument("--holdout", required=True,
                    help="centre exclu du pré-entraînement et évalué (ex: B)")
    ap.add_argument("--pretrain", nargs="*", default=None,
                    help="centres de pré-entraînement (def: tous les ingérés sauf holdout)")
    ap.add_argument("--top_n", type=int, default=10, help="nb d'espèces (def: 10)")
    args = ap.parse_args(argv)

    holdout = args.holdout.upper()
    available = _ingested_centers()
    if holdout not in available:
        print(f"[LOCO] centre '{holdout}' non ingéré. Disponibles : {available}\n"
              f"       -> place data/DRIAMS_{holdout}.tar.gz puis lance scripts/01_ingest.py")
        return

    pre = ([c.upper() for c in args.pretrain] if args.pretrain
           else [c for c in available if c != holdout])
    pre = [c for c in pre if c != holdout and c in available]
    if not pre:
        print(f"[LOCO] aucun centre de pré-entraînement disponible (≠ {holdout}). "
              f"Disponibles : {available}")
        return

    run_name = f"loco_holdout_{holdout}"
    print(f"=== LOCO : holdout={holdout} | pré-entraînement={pre} | run={run_name} ===")
    if len(pre) == 1:
        print("⚠️  Un seul centre de pré-entraînement : une éventuelle baisse sur le "
              "centre exclu peut mêler 'centre inédit' ET 'moins de données SSL'. "
              "Pour isoler l'effet centre, ingère un 3e centre (DRIAMS-D ou A) et "
              "passe-le via --pretrain.")

    # --- 1. pré-entraînement SSL en EXCLUANT le centre holdout ---
    X, _, _ = load_centers(pre)
    print(f"[LOCO] pré-entraînement sur {pre} : {X.shape[0]} spectres (centre "
          f"{holdout} totalement exclu)")
    cfg = CFG
    if os.environ.get("EPOCHS"):
        cfg = replace(CFG, train=replace(CFG.train, epochs=int(os.environ["EPOCHS"])))
    pretrain(X, cfg=cfg, run_name=run_name)

    # --- 2. évaluation : probe sur un centre VU (pre[0]) -> test sur holdout (inédit) ---
    seen = pre[0]
    out = run_comparison(run_name=run_name, centers=(seen, holdout),
                         top_n=args.top_n, seed=0, n_boot=1000)

    key = f"{seen}->{holdout}"
    print("\n=== LOCO — synthèse (balanced accuracy) ===")
    loco_row = None
    for r in out["results"]:
        v = r["preds"]["VICRegL"]["metrics"]["balanced_accuracy"]
        rf = r["preds"]["RF-binned"]["metrics"]["balanced_accuracy"]
        flag = "  <-- centre INÉDIT (LOCO)" if r["setting"] == key else ""
        print(f"  {r['setting']:8s} [{r['kind']:12s}] VICRegL {v:.3f} | RF {rf:.3f}{flag}")
        if r["setting"] == key:
            loco_row = r
    if loco_row is not None:
        v = loco_row["preds"]["VICRegL"]["metrics"]["balanced_accuracy"]
        print(f"\n>>> Centre {holdout} JAMAIS vu au pré-entraînement : "
              f"VICRegL bal-acc = {v:.3f} ({key}).")
        print(f"    Compare à runs/pretrain/RESULTS.md (même condition {key}, "
              f"mais {holdout} y était vu non labellisé) pour mesurer la perte.")
    print(f"\nRapport -> runs/{run_name}/RESULTS.md")
    print(f"Figures -> runs/{run_name}/figs/")


if __name__ == "__main__":
    main(sys.argv[1:])
