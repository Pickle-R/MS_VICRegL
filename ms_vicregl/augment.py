"""Augmentations MALDI = simulateurs d'artefacts de centre/environnement.

Chaque vue d'un spectre est construite par :
  1. une transformation de coordonnées (crop + warp non-linéaire de l'axe m/z),
     qui simule la dérive de calibration et fournit la correspondance de position
     nécessaire au critère LOCAL de VICRegL ;
  2. des perturbations d'intensité (baseline, gain, dropout de pics, bruit) qui
     simulent matrice/détecteur/ionisation.

make_views() renvoie deux vues + leurs coordonnées (m/z source) pooled à la
résolution de la carte de features (feature_len), pour l'appariement local.
"""
from __future__ import annotations

import numpy as np

from .config import AugConfig, CFG


def _smooth_curve(L: int, n_ctrl: int, lo: float, hi: float, rng) -> np.ndarray:
    """Courbe lisse basse fréquence par interpolation de n_ctrl points de contrôle."""
    xs = np.linspace(0, L - 1, n_ctrl)
    ys = rng.uniform(lo, hi, n_ctrl)
    return np.interp(np.arange(L), xs, ys)


def _coord_map(L: int, aug: AugConfig, rng) -> np.ndarray:
    """Map index de sortie -> position source (float) dans le spectre de base.

    Combine un crop [s0, s0+w] et une warp lisse (dérive de calibration).
    Les coordonnées sont en unités d'index de la grille (1 idx = 3 Da).
    """
    crop_frac = rng.uniform(aug.crop_frac_min, aug.crop_frac_max)
    w = max(2, int(L * crop_frac))
    s0 = rng.integers(0, L - w + 1)
    base = s0 + (w - 1) * np.arange(L) / (L - 1)                  # crop linéaire
    warp = _smooth_curve(L, aug.warp_ctrl, -aug.warp_amp, aug.warp_amp, rng)
    return np.clip(base + warp, 0, L - 1)


def _one_view(base_x: np.ndarray, aug: AugConfig, rng):
    """Construit une vue augmentée + ses coordonnées source (longueur L)."""
    L = base_x.shape[0]
    coords = _coord_map(L, aug, rng)
    x = np.interp(coords, np.arange(L), base_x)                  # resample (crop+warp)

    # --- perturbations d'intensité (n'affectent pas les coordonnées) ---
    x = x * _smooth_curve(L, aug.gain_ctrl, 1 - aug.gain_amp, 1 + aug.gain_amp, rng)  # gain
    xmax = x.max()
    if xmax > 0:
        x = x + aug.baseline_amp * xmax * _smooth_curve(L, aug.baseline_ctrl, 0.0, 1.0, rng)
    if aug.peak_dropout > 0:                                     # atténuation aléatoire de pics
        keep = (rng.random(L) > aug.peak_dropout).astype(np.float32)
        x = x * keep
    if aug.noise_std > 0:
        x = x + rng.normal(0.0, aug.noise_std * (x.std() + 1e-8), L)
    x = np.clip(x, 0.0, None)
    return x.astype(np.float32), coords.astype(np.float32)


def _pool_coords(coords: np.ndarray, feature_len: int) -> np.ndarray:
    """Pool les coordonnées source à la résolution de la carte de features."""
    L = coords.shape[0]
    if L % feature_len == 0:
        return coords.reshape(feature_len, L // feature_len).mean(axis=1).astype(np.float32)
    idx = np.linspace(0, L, feature_len + 1).astype(int)
    return np.array([coords[idx[i]:idx[i + 1]].mean() for i in range(feature_len)], np.float32)


def make_views(base_x: np.ndarray, feature_len: int | None = None,
               aug: AugConfig | None = None, rng=None):
    """Deux vues augmentées d'un même spectre + coordonnées (pour le matching local).

    Retourne (v1, c1, v2, c2) où v* sont (L,) et c* sont (feature_len,).
    """
    aug = aug or CFG.aug
    feature_len = feature_len or CFG.model.feature_len
    rng = rng or np.random.default_rng()
    v1, c1 = _one_view(base_x, aug, rng)
    v2, c2 = _one_view(base_x, aug, rng)
    return v1, _pool_coords(c1, feature_len), v2, _pool_coords(c2, feature_len)
