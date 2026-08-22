"""Datasets torch : vues SSL (non labellisé) et données labellisées (sonde).

Les .npy sont chargés en mmap (mmap_mode='r') pour ne pas saturer la RAM 18 Go.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import make_views
from .config import CFG, PROCESSED


def load_center(center: str, processed: Path | None = None):
    """Charge (X_raw, X_bin, meta) d'un centre depuis data/processed/."""
    processed = processed or PROCESSED
    X = np.load(processed / f"{center}_X.npy", mmap_mode="r")
    Xb = np.load(processed / f"{center}_Xbin.npy", mmap_mode="r")
    meta = pd.read_parquet(processed / f"{center}_meta.parquet")
    return X, Xb, meta


def load_centers(centers):
    """Concatène plusieurs centres. Retourne (X_raw, X_bin, meta concaténée)."""
    Xs, Xbs, metas = [], [], []
    for c in centers:
        X, Xb, meta = load_center(c)
        Xs.append(np.asarray(X)); Xbs.append(np.asarray(Xb)); metas.append(meta)
    return (np.concatenate(Xs), np.concatenate(Xbs),
            pd.concat(metas, ignore_index=True))


def load_centers_with_domain(centers):
    """Comme `load_centers`, mais renvoie en plus l'id de centre (0..len(centers)-1)
    par spectre — nécessaire pour le terme d'invariance de centre (DANN) en
    pré-entraînement SSL. Retourne (X_raw, X_bin, meta concaténée, domain)."""
    Xs, Xbs, metas, doms = [], [], [], []
    for i, c in enumerate(centers):
        X, Xb, meta = load_center(c)
        Xs.append(np.asarray(X)); Xbs.append(np.asarray(Xb)); metas.append(meta)
        doms.append(np.full(len(meta), i, dtype=np.int64))
    return (np.concatenate(Xs), np.concatenate(Xbs),
            pd.concat(metas, ignore_index=True), np.concatenate(doms))


def load_centers_with_domain_species(centers):
    """Comme `load_centers_with_domain`, + id d'espèce (encodage cohérent sur l'union
    des centres) — pour le prior par espèce (SpeciesPrior, inspiré DALMA). Retourne
    (X_raw, X_bin, meta concaténée, domain, species, label_encoder)."""
    from sklearn.preprocessing import LabelEncoder

    Xs, Xbs, metas, doms = [], [], [], []
    for i, c in enumerate(centers):
        X, Xb, meta = load_center(c)
        Xs.append(np.asarray(X)); Xbs.append(np.asarray(Xb)); metas.append(meta)
        doms.append(np.full(len(meta), i, dtype=np.int64))
    meta_all = pd.concat(metas, ignore_index=True)
    le = LabelEncoder().fit(meta_all["species"])
    species = le.transform(meta_all["species"]).astype(np.int64)
    return (np.concatenate(Xs), np.concatenate(Xbs), meta_all,
            np.concatenate(doms), species, le)


class SSLViewDataset(Dataset):
    """Renvoie deux vues augmentées + leurs coordonnées (non labellisé), sous forme
    de dict {"v1","c1","v2","c2", ["domain"], ["species"]}.

    `domain` (id de centre, cf. `load_centers_with_domain`) et `species` (id d'espèce,
    cf. `load_centers_with_domain_species`) sont deux canaux de supervision optionnels
    et indépendants, consommés respectivement par le terme DANN/CORAL et le terme
    SpeciesPrior dans `pretrain`. Chacun n'apparaît dans le dict que s'il est fourni.
    """

    def __init__(self, X: np.ndarray, cfg=CFG, domain: np.ndarray | None = None,
                 species: np.ndarray | None = None):
        self.X = X
        self.cfg = cfg
        self.domain = domain
        self.species = species

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        base = np.asarray(self.X[i], dtype=np.float32)
        v1, c1, v2, c2 = make_views(base, self.cfg.model.feature_len, self.cfg.aug)
        item = {"v1": torch.from_numpy(v1), "c1": torch.from_numpy(c1),
                "v2": torch.from_numpy(v2), "c2": torch.from_numpy(c2)}
        if self.domain is not None:
            item["domain"] = int(self.domain[i])
        if self.species is not None:
            item["species"] = int(self.species[i])
        return item


class LabeledDataset(Dataset):
    """Spectres + labels entiers (pour extraction de features / sonde)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)), int(self.y[i])
