"""Perte VICRegL adaptée au 1D : critère global + critère local (position + feature).

L = alpha * VICReg(global) + (1 - alpha) * [ L_s(1->2)+L_s(2->1) + L_d(1->2)+L_d(2->1) ]

  - VICReg(global) sur les embeddings globaux poolés.
  - L_s : appariement par POSITION (m/z) des vecteurs locaux, top-gamma.
  - L_d : appariement par FEATURE (plus proche voisin L2), top-gamma.
Chaque appariement est évalué par les mêmes 3 termes VICReg (var, inv, cov).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import CFG, LossConfig


def vicreg_terms(z1: torch.Tensor, z2: torch.Tensor, cfg: LossConfig, eps: float = 1e-4):
    """Termes VICReg sur deux ensembles appariés (N, D). Retourne (loss, inv, var, cov)."""
    inv = F.mse_loss(z1, z2)

    def var_cov(z):
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + eps)
        var = torch.mean(F.relu(1.0 - std))
        n, d = z.shape
        cov_m = (z.T @ z) / max(n - 1, 1)
        off = cov_m.pow(2).sum() - cov_m.diagonal().pow(2).sum()
        return var, off / d

    v1, c1 = var_cov(z1)
    v2, c2 = var_cov(z2)
    var, cov = v1 + v2, c1 + c2
    loss = cfg.sim_coeff * inv + cfg.std_coeff * var + cfg.cov_coeff * cov
    return loss, inv.detach(), var.detach(), cov.detach()


def _pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distances euclidiennes (B, La, Lb) — implémenté sans torch.cdist (compat MPS)."""
    a2 = (a * a).sum(-1, keepdim=True)              # (B, La, 1)
    b2 = (b * b).sum(-1, keepdim=True).transpose(1, 2)  # (B, 1, Lb)
    d2 = a2 + b2 - 2.0 * torch.bmm(a, b.transpose(1, 2))
    return d2.clamp_min(0.0).sqrt()


def _match_loss(za, zb, dist, cfg: LossConfig):
    """Apparie chaque position de za à son plus proche voisin de zb selon `dist`,
    garde les top-gamma paires les plus proches, puis applique VICReg.

    za, zb : (B, L', D) ; dist : (B, L', L').
    """
    B, Lp, D = za.shape
    nn_idx = dist.argmin(dim=2)                                  # (B, L')
    dmin = dist.gather(2, nn_idx.unsqueeze(2)).squeeze(2)        # (B, L')
    matched = torch.gather(zb, 1, nn_idx.unsqueeze(2).expand(-1, -1, D))  # (B, L', D)
    g = min(cfg.gamma, Lp)
    sel = dmin.topk(g, dim=1, largest=False).indices            # (B, g)
    a_sel = torch.gather(za, 1, sel.unsqueeze(2).expand(-1, -1, D)).reshape(-1, D)
    b_sel = torch.gather(matched, 1, sel.unsqueeze(2).expand(-1, -1, D)).reshape(-1, D)
    return vicreg_terms(a_sel, b_sel, cfg)[0]


def local_loss(z1, z2, coords1, coords2, cfg: LossConfig):
    """Critère local VICRegL : appariement position + feature, symétrisé.

    z1, z2 : (B, D, L') ; coords1, coords2 : (B, L').
    """
    z1 = z1.transpose(1, 2)                                     # (B, L', D)
    z2 = z2.transpose(1, 2)
    # appariement par position (m/z source)
    loc_d = (coords1.unsqueeze(2) - coords2.unsqueeze(1)).abs()  # (B, L', L')
    L_s = _match_loss(z1, z2, loc_d, cfg) + _match_loss(z2, z1, loc_d.transpose(1, 2), cfg)
    # appariement par feature
    feat_d = _pairwise_dist(z1, z2)                             # (B, L', L')
    L_d = _match_loss(z1, z2, feat_d, cfg) + _match_loss(z2, z1, feat_d.transpose(1, 2), cfg)
    return L_s + L_d


def vicregl_loss(g1, z1, coords1, g2, z2, coords2, cfg: LossConfig | None = None):
    """Perte VICRegL complète. Retourne (total, dict de composantes pour log)."""
    cfg = cfg or CFG.loss
    glob, inv, var, cov = vicreg_terms(g1, g2, cfg)
    loc = local_loss(z1, z2, coords1, coords2, cfg)
    total = cfg.alpha * glob + (1.0 - cfg.alpha) * loc
    logs = {"total": float(total.detach()), "global": float(glob.detach()),
            "local": float(loc.detach()), "inv": float(inv),
            "var": float(var), "cov": float(cov)}
    return total, logs


def coral_loss(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    """CORAL : aligne moyenne + covariance entre les centres présents dans le batch.

    features : (N, D) représentation pré-expander (même point d'accroche que DANN).
    domains  : (N,) id de centre.

    Contrairement à DANN, pas de réseau auxiliaire ni de jeu adversarial : chaque
    paire de centres est directement rapprochée en moyenne/covariance. Ça évite le
    piège diagnostiqué avec DANN (RESULT 6, MEMORY) — un discriminateur en ligne
    pas assez convergé donnait un faux signal d'invariance (domain_acc en baisse
    pendant l'entraînement, mais AUROC hors-ligne sur features gelées inchangé).
    """
    uniq = torch.unique(domains)
    if uniq.numel() < 2:
        return features.new_zeros(())
    D = features.shape[1]
    stats = []
    for d in uniq:
        f = features[domains == d]
        if f.shape[0] < 2:
            continue
        mu = f.mean(dim=0)
        fc = f - mu
        cov = (fc.T @ fc) / (f.shape[0] - 1)
        stats.append((mu, cov))
    if len(stats) < 2:
        return features.new_zeros(())
    loss = features.new_zeros(())
    npairs = 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            mu_i, cov_i = stats[i]
            mu_j, cov_j = stats[j]
            loss = loss + F.mse_loss(mu_i, mu_j) + (cov_i - cov_j).pow(2).sum() / (4 * D * D)
            npairs += 1
    return loss / max(npairs, 1)


def species_prior_loss(repr_: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Tire repr_ vers le prototype de son espèce (MSE) — cf. SpeciesPrior dans model.py
    pour le pourquoi de cette adaptation déterministe du prior conditionné de DALMA."""
    return F.mse_loss(repr_, prototypes)


def domain_adversarial_loss(domain_head, repr1, repr2, domains, lam):
    """Perte DANN : cross-entropy (via GRL) sur les DEUX vues contre l'id de centre.

    repr1, repr2 : (B, repr_dim) représentation pré-expander des deux vues (même
    spectre source -> même `domains`). `domain_head(x, lam)` applique déjà le GRL.
    Retourne (loss, accuracy) — l'accuracy est un diagnostic : elle doit décroître
    vers 1/n_domains (hasard) si l'invariance de centre progresse.
    """
    logits1 = domain_head(repr1, lam)
    logits2 = domain_head(repr2, lam)
    loss = 0.5 * (F.cross_entropy(logits1, domains) + F.cross_entropy(logits2, domains))
    with torch.no_grad():
        acc = 0.5 * ((logits1.argmax(dim=1) == domains).float().mean()
                     + (logits2.argmax(dim=1) == domains).float().mean())
    return loss, float(acc)
