"""Pré-entraînement auto-supervisé (VICRegL-1D) sur spectres non labellisés."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG, RUNS, get_device
from .dataset import SSLViewDataset
from .loss import coral_loss, domain_adversarial_loss, species_prior_loss, vicregl_loss
from .model import DomainHead, SpeciesPrior, VICRegLModel, dann_lambda


def _cosine_lr(step, total, base_lr, warmup, final_frac):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _save_ckpt(model, cfg, history, run_name):
    out = RUNS / run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(),
                "model": model.state_dict(),
                "cfg_model": cfg.model.__dict__,
                "history": history}, out / "ckpt.pt")
    return out / "ckpt.pt"


def pretrain(X: np.ndarray, cfg=CFG, run_name: str = "pretrain",
             device=None, save: bool = True, ckpt_every: int = 0,
             domain: np.ndarray | None = None, species: np.ndarray | None = None):
    """Entraîne VICRegLModel en SSL. Retourne (model, history).

    ckpt_every > 0 : sauvegarde périodique toutes les N époques (checkpoint
    intermédiaire écrasant runs/<run>/ckpt.pt), pour survivre à une interruption
    d'un run long. La sauvegarde finale a lieu quoi qu'il arrive si save=True.

    domain : id de centre par spectre (cf. `dataset.load_centers_with_domain`).
    Si fourni ET cfg.loss.domain_coeff > 0, ajoute un terme d'invariance de
    centre sur la représentation pré-expander, pour l'attaquer directement
    pendant l'entraînement plutôt que de seulement la diagnostiquer après coup
    (cf. AUROC classifieur de domaine sur features gelées). Deux méthodes
    (cfg.loss.domain_method) :
      - "dann"  : tête de classification de centre derrière un gradient-reversal
                  layer. Auxiliaire d'entraînement, pas sauvegardé dans le
                  checkpoint (seul `model.encoder` sert en aval).
      - "coral" : alignement direct moyenne+covariance entre centres, sans
                  réseau auxiliaire ni jeu adversarial (évite le piège
                  diagnostiqué avec DANN : discriminateur en ligne pas assez
                  convergé -> faux signal d'invariance, cf. MEMORY RESULT 6).

    species : id d'espèce par spectre (cf. `dataset.load_centers_with_domain_species`).
    Si fourni ET cfg.loss.species_coeff > 0, ajoute un terme SpeciesPrior qui tire
    repr_ vers un prototype appris par espèce, commun à tous les centres — adaptation
    déterministe du prior conditionné par espèce de DALMA (le levier principal de sa
    transférabilité), indépendant du terme domain/DANN/CORAL ci-dessus (les deux
    peuvent être actifs simultanément).
    """
    device = device or get_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    use_domain = domain is not None and cfg.loss.domain_coeff > 0
    use_dann = use_domain and cfg.loss.domain_method == "dann"
    use_coral = use_domain and cfg.loss.domain_method == "coral"
    use_species = species is not None and cfg.loss.species_coeff > 0
    ds = SSLViewDataset(X, cfg, domain=domain, species=species)
    # l'augmentation (numpy/CPU) est le goulot -> on parallélise via les workers.
    # OMP_NUM_THREADS=1 (réglé dans le package) évite la sur-souscription des cœurs.
    nw = cfg.train.num_workers
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=nw, drop_last=True, pin_memory=False,
                    persistent_workers=nw > 0)

    model = VICRegLModel(cfg.model).to(device)
    domain_head = None
    species_prior = None
    params = list(model.parameters())
    n_domains = int(np.max(domain)) + 1 if use_domain else 0
    n_species = int(np.max(species)) + 1 if use_species else 0
    if use_dann:
        domain_head = DomainHead(cfg.model.repr_dim, n_domains, cfg.loss.domain_hidden).to(device)
        params += list(domain_head.parameters())
    if use_species:
        species_prior = SpeciesPrior(n_species, cfg.model.repr_dim).to(device)
        params += list(species_prior.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    total_steps = cfg.train.epochs * max(len(dl), 1)
    history = []
    step = 0
    t0 = time.time()
    domain_tag = (f" | {cfg.loss.domain_method.upper()} n_domains={n_domains} "
                  f"coeff={cfg.loss.domain_coeff}" if use_domain else "")
    species_tag = (f" | SpeciesPrior n_species={n_species} "
                   f"coeff={cfg.loss.species_coeff}" if use_species else "")
    print(f"[pretrain] device={device} | N={len(ds)} | batches/epoch={len(dl)} "
          f"| epochs={cfg.train.epochs} | total_steps={total_steps}{domain_tag}{species_tag}")

    for epoch in range(cfg.train.epochs):
        model.train()
        ep_logs = []
        for batch in dl:
            v1, c1 = batch["v1"].to(device), batch["c1"].to(device)
            v2, c2 = batch["v2"].to(device), batch["c2"].to(device)
            dom = batch["domain"].to(device) if "domain" in batch else None
            sp = batch["species"].to(device) if "species" in batch else None

            lr = _cosine_lr(step, total_steps, cfg.train.lr,
                            cfg.train.warmup_epochs * len(dl), cfg.train.final_lr_frac)
            for pg in opt.param_groups:
                pg["lr"] = lr

            g1, z1, r1 = model(v1)
            g2, z2, r2 = model(v2)
            loss, logs = vicregl_loss(g1, z1, c1, g2, z2, c2, cfg.loss)

            if use_dann:
                lam = dann_lambda(step / max(total_steps - 1, 1), cfg.loss.domain_gamma)
                dloss, dacc = domain_adversarial_loss(domain_head, r1, r2, dom, lam)
                loss = loss + cfg.loss.domain_coeff * dloss
                logs["domain_loss"] = float(dloss.detach())
                logs["domain_acc"] = dacc
                logs["dann_lambda"] = lam
            elif use_coral:
                dloss = 0.5 * (coral_loss(r1, dom) + coral_loss(r2, dom))
                loss = loss + cfg.loss.domain_coeff * dloss
                logs["domain_loss"] = float(dloss.detach())

            if use_species:
                proto = species_prior(sp)
                sloss = 0.5 * (species_prior_loss(r1, proto) + species_prior_loss(r2, proto))
                loss = loss + cfg.loss.species_coeff * sloss
                logs["species_loss"] = float(sloss.detach())

            logs["total"] = float(loss.detach())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            logs["lr"] = lr
            ep_logs.append(logs)
            if step % cfg.train.log_every == 0:
                domain_msg = ""
                if use_dann:
                    domain_msg = (f" | dom_loss {logs['domain_loss']:.3f} "
                                  f"dom_acc {logs['domain_acc']:.2f} lam {logs['dann_lambda']:.2f}")
                elif use_coral:
                    domain_msg = f" | dom_loss {logs['domain_loss']:.3f}"
                species_msg = f" | sp_loss {logs['species_loss']:.3f}" if use_species else ""
                print(f"  ep {epoch:3d} step {step:5d} | loss {logs['total']:.3f} "
                      f"(g {logs['global']:.3f} l {logs['local']:.3f}) "
                      f"inv {logs['inv']:.3f} var {logs['var']:.3f} cov {logs['cov']:.3f} "
                      f"| lr {lr:.2e}{domain_msg}{species_msg}", flush=True)
            step += 1

        mean = {k: float(np.mean([d[k] for d in ep_logs])) for k in ep_logs[0]}
        mean["epoch"] = epoch
        history.append(mean)

        # sauvegarde périodique (insurance contre interruption d'un run long)
        if (save and ckpt_every > 0 and (epoch + 1) % ckpt_every == 0
                and epoch + 1 < cfg.train.epochs):
            p = _save_ckpt(model, cfg, history, run_name)
            print(f"  [ckpt] époque {epoch+1}/{cfg.train.epochs} -> {p}", flush=True)

        # pause de refroidissement (limite la chauffe en sollicitation soutenue)
        if (cfg.train.cooldown_s > 0 and cfg.train.cooldown_every > 0
                and (epoch + 1) % cfg.train.cooldown_every == 0
                and epoch + 1 < cfg.train.epochs):
            time.sleep(cfg.train.cooldown_s)

    dt = time.time() - t0
    print(f"[pretrain] terminé en {dt/60:.1f} min")

    if save:
        p = _save_ckpt(model, cfg, history, run_name)
        print(f"[pretrain] checkpoint -> {p}")
    return model, history


@torch.no_grad()
def extract_features(encoder, X: np.ndarray, device=None, batch_size: int = 512):
    """Représentations gelées (N, repr_dim) à partir de l'encodeur."""
    device = device or get_device()
    encoder = encoder.to(device).eval()
    feats = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(np.asarray(X[i:i + batch_size], dtype=np.float32)).to(device)
        feats.append(encoder.represent(xb).cpu().numpy())
    return np.concatenate(feats)
