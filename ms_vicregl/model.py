"""Backbone ResNet-1D + têtes VICRegL (expander global, projecteur local).

L'encodeur transforme un spectre (B, 1, L) en une carte de features locales
(B, C, L') via des convolutions 1D résiduelles stridées, puis force L' = feature_len
par AdaptiveAvgPool1d (robuste à l'arithmétique des convolutions).

  - représentation gelée (downstream)  = moyenne de la carte sur L'  -> (B, C)
  - embedding global (VICReg)           = expander(représentation)    -> (B, expander_dim)
  - embedding local (VICRegL)           = projecteur 1x1 sur la carte  -> (B, proj, L')
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, ModelConfig


def adaptive_avg_pool1d_safe(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """Adaptive avg pool 1D robuste sur MPS.

    MPS n'implémente pas l'adaptive pooling quand la longueur d'entrée n'est pas
    divisible par la sortie. On utilise alors avg_pool1d (cas divisible, rapide sur
    MPS), sinon on bascule ce seul op sur CPU.
    """
    L = x.shape[-1]
    if L == out_len:
        return x
    if L % out_len == 0:
        return F.avg_pool1d(x, kernel_size=L // out_len)
    if x.device.type == "mps":
        return F.adaptive_avg_pool1d(x.cpu(), out_len).to(x.device)
    return F.adaptive_avg_pool1d(x, out_len)


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idt)


class ResNet1DEncoder(nn.Module):
    """ResNet-1D -> carte de features (B, repr_dim, feature_len)."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or CFG.model
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv1d(1, cfg.channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(cfg.channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers = []
        c_prev = cfg.channels[0]
        for i, (c, n) in enumerate(zip(cfg.channels, cfg.blocks)):
            stride = 1 if i == 0 else 2
            layers.append(BasicBlock1d(c_prev, c, stride=stride))
            for _ in range(n - 1):
                layers.append(BasicBlock1d(c, c, stride=1))
            c_prev = c
        self.body = nn.Sequential(*layers)
        self.feature_len = cfg.feature_len                 # force L' = feature_len (pool MPS-safe)
        assert c_prev == cfg.repr_dim, "le dernier canal doit valoir repr_dim"

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                              # (B, L) -> (B, 1, L)
        x = self.stem(x)
        x = self.body(x)
        return adaptive_avg_pool1d_safe(x, self.feature_len)   # (B, repr_dim, feature_len)

    def represent(self, x):
        """Représentation gelée pour la sonde linéaire : (B, repr_dim)."""
        return self.forward(x).mean(dim=2)

    def represent_segments(self, x, n_segments: int | str = "max"):
        """Représentation gelée à structure spatiale : la carte locale
        (B, repr_dim, L') est découpée en n_segments tronçons contigus le long de
        l'axe m/z, chacun moyenné séparément puis concaténés sur les canaux ->
        (B, n_segments * repr_dim). n_segments=1 est identique à represent()
        (moyenne globale -- efface toute position, deux biomarqueurs distants ne
        sont alors plus distinguables dans repr_).

        n_segments="max" (défaut) : résolution spatiale MAXIMALE, un segment par
        position de la carte locale -- L' = fmap.shape[-1], déterminé
        automatiquement à partir de la sortie réelle de l'encodeur pour ce
        spectre (donc de son ModelConfig.feature_len), plutôt qu'une valeur codée
        en dur. C'est le nombre de tronçons au-delà duquel il n'y a plus rien à
        gagner : chaque tronçon supplémentaire serait vide.
        """
        fmap = self.forward(x)                     # (B, d, L')
        L = fmap.shape[-1]
        if n_segments == "max":
            n_segments = L
        if n_segments > L:
            raise ValueError(f"n_segments={n_segments} > résolution max de la "
                             f"carte locale (feature_len={L})")
        if n_segments <= 1:
            return fmap.mean(dim=2)
        chunks = torch.tensor_split(fmap, n_segments, dim=2)
        return torch.cat([c.mean(dim=2) for c in chunks], dim=1)


def _mlp(dims, last_bn=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
        elif last_bn:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
    return nn.Sequential(*layers)


class VICRegLModel(nn.Module):
    """Encodeur + expander global + projecteur local."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or CFG.model
        self.cfg = cfg
        self.encoder = ResNet1DEncoder(cfg)
        d, e, p = cfg.repr_dim, cfg.expander_dim, cfg.projector_dim
        self.global_expander = _mlp([d, e, e, e])
        self.local_projector = nn.Sequential(   # 1x1 conv = MLP partagé sur les positions
            nn.Conv1d(d, p, 1), nn.BatchNorm1d(p), nn.ReLU(inplace=True),
            nn.Conv1d(p, p, 1), nn.BatchNorm1d(p), nn.ReLU(inplace=True),
            nn.Conv1d(p, p, 1),
        )

    def forward(self, x):
        fmap = self.encoder(x)                  # (B, d, L')
        repr_ = fmap.mean(dim=2)                 # (B, d) — ce qu'évalue la sonde downstream
        g = self.global_expander(repr_)              # (B, e)
        z = self.local_projector(fmap)               # (B, p, L')
        return g, z, repr_


# --------------------------------------------------------------------------- #
# Invariance de centre (DANN) : gradient reversal + tête de classification
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    """Identité en forward, gradient multiplié par -lambda en backward
    (Ganin & Lempitsky, 2015). Place la tête de domaine en aval de ce point
    pour qu'elle apprenne à prédire le centre tout en poussant l'encodeur, en
    amont, dans la direction opposée (représentation indiscernable par centre)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Planning standard DANN : 0 -> 1 selon l'avancement (0..1) de l'entraînement.
    Démarre doux pour ne pas déstabiliser l'encodeur avant que la représentation
    soit un minimum formée, puis sature à 1."""
    progress = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


class DomainHead(nn.Module):
    """Classifieur de centre (repr_dim -> n_domains) branché derrière un GRL.

    C'est exactement le diagnostic utilisé hors-ligne (classifieur de domaine sur
    features gelées, AUROC ~1.0) mais intégré à l'entraînement : au lieu de
    seulement mesurer la fuite d'information de centre après coup, on
    l'attaque directement à chaque step.
    """

    def __init__(self, repr_dim: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, repr_: torch.Tensor, lam: float) -> torch.Tensor:
        return self.net(grad_reverse(repr_, lam))


class SpeciesPrior(nn.Module):
    """Prototype de représentation par espèce, un par espèce (repr_dim,).

    Adaptation déterministe du prior conditionné par espèce de DALMA
    (Garcia-Navarro et al. 2026) : dans DALMA, un encodeur VARIATIONNEL est tiré vers
    N(mu_s, sigma_s^2) par KL. Notre encodeur VICRegL n'a pas de tête de variance ni
    de reparamétrisation (pas de postérieure à proprement parler) — la vraie KL de
    DALMA n'est donc pas applicable telle quelle. On garde l'effet de premier ordre :
    un prototype appris mu_s vers lequel repr_ est tiré (MSE), sans modéliser
    l'étalement (sigma_s). C'est le levier qui, dans l'ablation de DALMA (Table 3),
    porte l'essentiel du gain de transférabilité — plus que les décodeurs par centre.
    """

    def __init__(self, n_species: int, repr_dim: int):
        super().__init__()
        self.prototypes = nn.Embedding(n_species, repr_dim)

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        return self.prototypes(species_ids)
