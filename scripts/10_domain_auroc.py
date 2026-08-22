#!/usr/bin/env python
"""Diagnostic autonome : classifieur de domaine (B∪C vs D) sur features gelées.

Mesure l'AUROC atteignable par une sonde entraînée APRÈS COUP à séparer les
centres à partir de la représentation gelée finale — le test qui décide si
l'invariance de centre a vraiment eu lieu, contrairement à `domain_acc` en
cours d'entraînement (qui vient d'un classifieur qui co-évolue avec
l'encodeur sous le GRL, cf. RESULT 6 dans la mémoire projet).

Usage:
    python scripts/10_domain_auroc.py pretrain_BCD pretrain_BCD_dann
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ms_vicregl.config import get_device
from ms_vicregl.dataset import load_center
from ms_vicregl.evaluate import load_encoder
from ms_vicregl.pretrain import extract_features


def domain_auroc(run_name, device, seed=0):
    enc = load_encoder(run_name)
    Xb, _, _ = load_center("B")
    Xc, _, _ = load_center("C")
    Xd, _, _ = load_center("D")
    Fbc = extract_features(enc, np.concatenate([np.asarray(Xb), np.asarray(Xc)]), device=device)
    Fd = extract_features(enc, np.asarray(Xd), device=device)
    F = np.concatenate([Fbc, Fd])
    y = np.concatenate([np.zeros(len(Fbc), dtype=int), np.ones(len(Fd), dtype=int)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, F, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def main(argv):
    device = get_device()
    for run in argv:
        auroc = domain_auroc(run, device)
        print(f"{run:20s} AUROC domaine (B+C) vs D = {auroc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pretrain_BCD", "pretrain_BCD_dann"])
