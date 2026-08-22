"""Configuration centrale du pipeline MS-VICRegL.

Tous les hyper-paramètres sont dimensionnés pour un Mac M3 Pro 18 Go (backend MPS).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # .../MS_VICRegL
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"
for _d in (DATA, RAW_DIR, PROCESSED, RUNS):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Grille m/z commune (resampling de l'entrée)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridConfig:
    mz_min: float = 2000.0
    mz_max: float = 20000.0
    n_bins: int = 6000          # L : longueur du vecteur d'entrée (3 Da / bin)

    @property
    def edges(self):
        import numpy as np
        return np.linspace(self.mz_min, self.mz_max, self.n_bins + 1)


# --------------------------------------------------------------------------- #
# Augmentations = simulateurs d'artefacts de centre/environnement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AugConfig:
    crop_frac_min: float = 0.6      # fraction de l'axe m/z conservée (crop)
    crop_frac_max: float = 1.0
    warp_amp: float = 3.0           # dérive de calibration, en unités d'index (1 idx = 3 Da)
    warp_ctrl: int = 6              # points de contrôle de la warp (basse fréquence)
    baseline_amp: float = 0.25      # amplitude baseline (relative au max du spectre)
    baseline_ctrl: int = 5
    gain_amp: float = 0.30          # +/- enveloppe de gain multiplicatif
    gain_ctrl: int = 6
    peak_dropout: float = 0.10      # proba d'atténuation par point
    noise_std: float = 0.05         # bruit additif (relatif à l'écart-type du spectre)


# --------------------------------------------------------------------------- #
# Modèle : ResNet-1D + expander global + projector local
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    in_len: int = 6000
    channels: tuple = (64, 128, 256, 512)   # canaux par stage
    blocks: tuple = (2, 2, 2, 2)            # blocs résiduels par stage
    feature_len: int = 47                    # L' : longueur de la carte locale (diviseur de body=188 -> pool MPS rapide)
    repr_dim: int = 512                      # dim de la représentation gelée (= dernier canal)
    expander_dim: int = 2048                 # réduit vs 8192 du papier (RAM 18 Go)
    projector_dim: int = 512                 # projecteur local


# --------------------------------------------------------------------------- #
# Perte VICRegL
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossConfig:
    sim_coeff: float = 25.0     # invariance
    std_coeff: float = 25.0     # variance
    cov_coeff: float = 1.0      # covariance
    alpha: float = 0.75         # poids du critère global vs local
    gamma: int = 20             # nb de meilleures correspondances locales conservées (top-gamma, papier VICRegL)
    # --- invariance de centre explicite (DANN domaine-adversarial, ou CORAL) ---
    domain_method: str = "dann"  # "dann" ou "coral"
    domain_coeff: float = 0.0   # poids du terme de domaine ; 0.0 = désactivé (défaut = comportement inchangé)
    domain_hidden: int = 128    # largeur de la tête de classification de centre (DANN uniquement)
    domain_gamma: float = 10.0  # pente du planning lambda du GRL (DANN uniquement, Ganin & Lempitsky 2015)
    # --- prior par espèce (inspiré DALMA, Garcia-Navarro et al. 2026) ---
    species_coeff: float = 0.0  # poids du terme de prototype par espèce ; 0.0 = désactivé


# --------------------------------------------------------------------------- #
# Entraînement SSL
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup_epochs: int = 10
    final_lr_frac: float = 1e-2     # lr final = lr * frac (cosine)
    num_workers: int = 0       # sur MPS, l'overhead spawn des workers > gain (bench: 0 plus rapide)
    seed: int = 0
    log_every: int = 20             # itérations
    cooldown_s: float = 0.0         # pause (s) pour laisser refroidir la puce
    cooldown_every: int = 0         # ... toutes les N époques (0 = désactivé)


@dataclass(frozen=True)
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def light_config(epochs: int = 80, cooldown_s: float = 8.0,
                 cooldown_every: int = 5) -> "Config":
    """Profil léger & thermiquement doux pour Mac : modèle ~1/8 du calcul du profil
    complet, moins d'époques, pauses de refroidissement régulières."""
    m = replace(CFG.model, channels=(32, 64, 128, 256), blocks=(1, 1, 1, 1),
                repr_dim=256, expander_dim=1024, projector_dim=256)
    t = replace(CFG.train, epochs=epochs, num_workers=0,
                cooldown_s=cooldown_s, cooldown_every=cooldown_every)
    return replace(CFG, model=m, train=t)


def medium_config(epochs: int = 120, cooldown_s: float = 6.0,
                  cooldown_every: int = 5) -> "Config":
    """Profil intermédiaire entre 'light' et 'full' : modèle moyen (~½ du calcul du
    complet), cooldown modéré. Bon compromis qualité / chauffe."""
    m = replace(CFG.model, channels=(48, 96, 192, 384), blocks=(2, 2, 2, 2),
                repr_dim=384, expander_dim=1536, projector_dim=384)
    t = replace(CFG.train, epochs=epochs, num_workers=0,
                cooldown_s=cooldown_s, cooldown_every=cooldown_every)
    return replace(CFG, model=m, train=t)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CFG = Config()
