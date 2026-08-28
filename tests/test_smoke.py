"""Smoke-test bout-en-bout sur données SYNTHÉTIQUES (sans DRIAMS).

Valide : augmentations -> modèle -> perte VICRegL -> backward -> extraction de
features -> sonde linéaire. À lancer avant que les vrais datasets soient prêts.

    /opt/miniconda3/bin/python -m pytest tests/ -v
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ms_vicregl.config import CFG
from ms_vicregl.augment import make_views
from ms_vicregl.model import VICRegLModel, ResNet1DEncoder
from ms_vicregl.loss import vicregl_loss
from ms_vicregl.dataset import SSLViewDataset, LabeledDataset


def _synthetic(n, L, n_classes=3, seed=0):
    """Spectres synthétiques : chaque classe a un MOTIF de pics propre (positions
    fixes, partagées entre appels), comme des biomarqueurs d'espèce. Volontairement
    PAS un décalage global de l'axe (que la SSL est entraînée à ignorer)."""
    rng = np.random.default_rng(seed)
    sig_rng = np.random.default_rng(12345)   # motifs identiques train/test
    class_peaks = [sig_rng.integers(int(L * 0.1), int(L * 0.9), 8)
                   for _ in range(n_classes)]
    X = np.zeros((n, L), np.float32)
    y = rng.integers(0, n_classes, n)
    grid = np.arange(L)
    for i in range(n):
        for c in class_peaks[y[i]]:
            cc = c + rng.integers(-3, 3)      # léger jitter de calibration (nuisance)
            X[i] += np.exp(-0.5 * ((grid - cc) / 6.0) ** 2) * rng.uniform(0.5, 1.5)
        X[i] += rng.normal(0, 0.01, L).clip(0)
        s = X[i].sum()
        if s > 0:
            X[i] /= s
    return X, y


def _tiny_cfg():
    # modèle minuscule pour un test rapide
    mc = replace(CFG.model, in_len=600, channels=(16, 32, 64, 64),
                 blocks=(1, 1, 1, 1), feature_len=12, repr_dim=64,
                 expander_dim=128, projector_dim=64)
    tc = replace(CFG.train, batch_size=8, epochs=1, num_workers=0)
    lc = replace(CFG.loss, gamma=4)
    return replace(CFG, model=mc, train=tc, loss=lc)


def test_augment_shapes():
    cfg = _tiny_cfg()
    X, _ = _synthetic(4, cfg.model.in_len)
    v1, c1, v2, c2 = make_views(X[0], cfg.model.feature_len, cfg.aug)
    assert v1.shape == (cfg.model.in_len,)
    assert c1.shape == (cfg.model.feature_len,)
    assert np.isfinite(v1).all() and (v1 >= 0).all()


def test_model_forward():
    cfg = _tiny_cfg()
    model = VICRegLModel(cfg.model)
    x = torch.randn(5, cfg.model.in_len)
    g, z, r = model(x)
    assert g.shape == (5, cfg.model.expander_dim)
    assert z.shape == (5, cfg.model.projector_dim, cfg.model.feature_len)
    assert r.shape == (5, cfg.model.repr_dim)
    assert model.encoder.represent(x).shape == (5, cfg.model.repr_dim)


def test_loss_backward():
    cfg = _tiny_cfg()
    model = VICRegLModel(cfg.model)
    B, L, Lp = 8, cfg.model.in_len, cfg.model.feature_len
    v1, v2 = torch.randn(B, L), torch.randn(B, L)
    c1 = torch.sort(torch.rand(B, Lp) * L, dim=1).values
    c2 = torch.sort(torch.rand(B, Lp) * L, dim=1).values
    g1, z1, _ = model(v1)
    g2, z2, _ = model(v2)
    loss, logs = vicregl_loss(g1, z1, c1, g2, z2, c2, cfg.loss)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_end_to_end_probe():
    """Pré-entraînement minimal + sonde : l'accuracy doit dépasser le hasard."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from ms_vicregl.pretrain import pretrain, extract_features

    cfg = _tiny_cfg()
    cfg = replace(cfg, train=replace(cfg.train, epochs=40, batch_size=16,
                                     warmup_epochs=2, lr=3e-3))
    Xtr, ytr = _synthetic(200, cfg.model.in_len, seed=1)
    Xte, yte = _synthetic(100, cfg.model.in_len, seed=2)

    model, hist = pretrain(Xtr, cfg=cfg, run_name="smoke", device=torch.device("cpu"),
                           save=False)
    assert hist[-1]["total"] < hist[0]["total"], "la perte SSL ne décroît pas"
    ftr = extract_features(model.encoder, Xtr, device=torch.device("cpu"))
    fte = extract_features(model.encoder, Xte, device=torch.device("cpu"))
    assert ftr.shape == (len(Xtr), cfg.model.feature_len * cfg.model.repr_dim), \
        "extract_features doit résoudre au max (feature_len) par défaut"
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(ftr, ytr)
    acc = clf.score(fte, yte)
    # tâche facile (motifs séparables) : une représentation qui apprend dépasse nettement le hasard
    assert acc > 0.6, f"sonde au niveau du hasard (acc={acc:.2f}) -> représentation non discriminante"


def test_represent_segments():
    """n_segments=1 == represent() ; n_segments=k -> (B, k*repr_dim), concat de
    moyennes par tronçon contigu de la carte locale ; "max" == feature_len réel
    de la carte (déterminé depuis la sortie de l'encodeur, pas codé en dur) ;
    n_segments > feature_len lève une erreur explicite."""
    cfg = _tiny_cfg()
    model = VICRegLModel(cfg.model)
    enc = model.encoder
    x = torch.randn(3, cfg.model.in_len)
    base = enc.represent(x)
    assert torch.allclose(enc.represent_segments(x, 1), base)
    k = 4
    seg = enc.represent_segments(x, k)
    assert seg.shape == (3, k * cfg.model.repr_dim)
    fmap = enc.forward(x)
    L = fmap.shape[-1]
    chunks = torch.tensor_split(fmap, k, dim=2)
    expected = torch.cat([c.mean(dim=2) for c in chunks], dim=1)
    assert torch.allclose(seg, expected)

    seg_max = enc.represent_segments(x, "max")
    assert seg_max.shape == (3, L * cfg.model.repr_dim)
    chunks_max = torch.tensor_split(fmap, L, dim=2)
    expected_max = torch.cat([c.mean(dim=2) for c in chunks_max], dim=1)
    assert torch.allclose(seg_max, expected_max)

    import pytest
    with pytest.raises(ValueError):
        enc.represent_segments(x, L + 1)


def test_dann_backward():
    """Terme d'invariance de centre (DANN) : tourne sans erreur, gradients finis
    sur l'encodeur ET la tête de domaine, et la perte totale reste finie."""
    from ms_vicregl.pretrain import pretrain

    cfg = _tiny_cfg()
    cfg = replace(cfg, train=replace(cfg.train, epochs=2, batch_size=8),
                 loss=replace(cfg.loss, domain_coeff=1.0))
    X, _ = _synthetic(32, cfg.model.in_len, seed=3)
    domain = np.array([0] * 16 + [1] * 16)  # 2 "centres" synthétiques

    model, hist = pretrain(X, cfg=cfg, run_name="smoke_dann", device=torch.device("cpu"),
                           save=False, domain=domain)
    assert all(np.isfinite(h["total"]) for h in hist)
    assert "domain_loss" in hist[-1] and "domain_acc" in hist[-1]
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_coral_backward():
    """CORAL : tourne sans erreur, gradients finis, pas de tête auxiliaire."""
    from ms_vicregl.pretrain import pretrain

    cfg = _tiny_cfg()
    cfg = replace(cfg, train=replace(cfg.train, epochs=2, batch_size=8),
                 loss=replace(cfg.loss, domain_method="coral", domain_coeff=1.0))
    X, _ = _synthetic(32, cfg.model.in_len, seed=4)
    domain = np.array([0] * 16 + [1] * 16)

    model, hist = pretrain(X, cfg=cfg, run_name="smoke_coral", device=torch.device("cpu"),
                           save=False, domain=domain)
    assert all(np.isfinite(h["total"]) for h in hist)
    assert "domain_loss" in hist[-1] and "domain_acc" not in hist[-1]
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_species_prior_backward():
    """SpeciesPrior (DALMA-inspired) : tourne sans erreur, gradients finis,
    indépendant du terme de domaine (ici désactivé)."""
    from ms_vicregl.pretrain import pretrain

    cfg = _tiny_cfg()
    cfg = replace(cfg, train=replace(cfg.train, epochs=2, batch_size=8),
                 loss=replace(cfg.loss, species_coeff=1.0))
    X, y = _synthetic(32, cfg.model.in_len, n_classes=3, seed=5)

    model, hist = pretrain(X, cfg=cfg, run_name="smoke_species", device=torch.device("cpu"),
                           save=False, species=y)
    assert all(np.isfinite(h["total"]) for h in hist)
    assert "species_loss" in hist[-1] and "domain_loss" not in hist[-1]
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
