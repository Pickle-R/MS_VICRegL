"""Évaluation : sonde linéaire sur features gelées + protocole CROSS-CENTRE,
comparé à une baseline Random Forest sur binned_6000 (format standard DRIAMS).

Métrique principale = robustesse au transfert de centre : on entraîne sur un
centre, on teste sur l'autre (non vu), et on compare la chute de performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CFG, RUNS, get_device
from .dataset import load_center
from .model import ResNet1DEncoder
from .config import ModelConfig
from .pretrain import extract_features


def load_encoder(run_name: str = "pretrain", device=None) -> ResNet1DEncoder:
    ck = torch.load(RUNS / run_name / "ckpt.pt", map_location="cpu")
    mc = ModelConfig(**{**CFG.model.__dict__, **ck.get("cfg_model", {})})
    enc = ResNet1DEncoder(mc)
    enc.load_state_dict(ck["encoder"])
    return enc


def common_topn_species(metas, top_n: int) -> list[str]:
    """Espèces présentes dans TOUS les centres, triées par fréquence totale (top_n)."""
    sets = [set(m["species"]) for m in metas]
    common = set.intersection(*sets) if sets else set()
    allsp = pd.concat([m["species"] for m in metas])
    counts = allsp[allsp.isin(common)].value_counts()
    return list(counts.index[:top_n])


def _filter(X, Xbin, meta, keep, le):
    mask = meta["species"].isin(keep).to_numpy()
    y = le.transform(meta.loc[mask, "species"].to_numpy())
    return np.asarray(X)[mask], np.asarray(Xbin)[mask], y


def _metrics(y_true, y_pred) -> dict:
    return {"acc": accuracy_score(y_true, y_pred),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "kappa": cohen_kappa_score(y_true, y_pred)}


def cross_center(encoder, train_center: str, test_center: str,
                 top_n: int = 10, device=None, seed: int = 0) -> dict:
    """Entraîne sur train_center, teste sur test_center (non vu)."""
    device = device or get_device()
    Xtr, Xbtr, mtr = load_center(train_center)
    Xte, Xbte, mte = load_center(test_center)
    keep = common_topn_species([mtr, mte], top_n)
    le = LabelEncoder().fit(keep)

    Xtr_f, Xbtr_f, ytr = _filter(Xtr, Xbtr, mtr, keep, le)
    Xte_f, Xbte_f, yte = _filter(Xte, Xbte, mte, keep, le)

    # --- sonde linéaire sur features VICRegL gelées ---
    ftr = extract_features(encoder, Xtr_f, device=device)
    fte = extract_features(encoder, Xte_f, device=device)
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced"))
    probe.fit(ftr, ytr)
    m_probe = _metrics(yte, probe.predict(fte))

    # --- baseline RF sur binned_6000 (standard DRIAMS) ---
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                class_weight="balanced", random_state=seed)
    rf.fit(Xbtr_f, ytr)
    m_rf = _metrics(yte, rf.predict(Xbte_f))

    return {"train": train_center, "test": test_center, "n_classes": len(keep),
            "n_train": len(ytr), "n_test": len(yte), "species": keep,
            "probe_vicregl": m_probe, "baseline_rf_binned": m_rf}


def run_full(run_name: str = "pretrain", centers=("C", "B"), top_n: int = 10):
    """Évalue les deux directions de transfert et affiche un rapport."""
    enc = load_encoder(run_name)
    a, b = centers
    results = [cross_center(enc, a, b, top_n), cross_center(enc, b, a, top_n)]
    print("\n===== ÉVALUATION CROSS-CENTRE (sonde VICRegL vs RF binned) =====")
    for r in results:
        print(f"\n  {r['train']} -> {r['test']}  "
              f"({r['n_classes']} espèces, train={r['n_train']}, test={r['n_test']})")
        for name, m in (("VICRegL (sonde lin.)", r["probe_vicregl"]),
                        ("RF binned_6000     ", r["baseline_rf_binned"])):
            print(f"    {name} | acc {m['acc']:.3f}  bal_acc {m['bal_acc']:.3f}  "
                  f"F1m {m['f1_macro']:.3f}  kappa {m['kappa']:.3f}")
    return results
