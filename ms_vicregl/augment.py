"""Augmentations MALDI = simulateurs d'artefacts de centre/environnement.

Chaque vue d'un spectre est construite par :
  1. un warp non-linéaire lisse de l'axe m/z, seul à avoir une justification
     physique de dérive de calibration (amplitude ~aug.warp_amp idx, cf. réalité
     instrumentale ~1-10 Da) ; il fournit la correspondance de position quasi-
     identité nécessaire au critère LOCAL de VICRegL ;
  2. des perturbations d'intensité (baseline, gain, dropout de pics, bruit,
     masquage d'un segment contigu) qui simulent matrice/détecteur/ionisation
     et fournissent la diversité entre les deux vues nécessaire au critère
     GLOBAL de VICRegL. Le masquage remplace un ancien crop-resize d'échelle
     (RandomResizedCrop façon vision) : celui-ci déplaçait un pic à 10000 Da de
     ~850 Da en médiane (p90 ~3100 Da, cf. simulation), soit 100-1000x l'ordre
     de grandeur d'une vraie dérive de calibration — il n'y a donc aucune valeur
     de crop_frac qui soit à la fois réaliste ET utile comme diversité de vue.
     Le masquage donne de la diversité sans déplacer l'axe m/z.

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

    Warp lisse autour de l'identité (dérive de calibration), sans rescale
    d'échelle. Les coordonnées sont en unités d'index de la grille (1 idx = 3 Da).
    """
    warp = _smooth_curve(L, aug.warp_ctrl, -aug.warp_amp, aug.warp_amp, rng)
    return np.clip(np.arange(L, dtype=np.float64) + warp, 0, L - 1)


def _apply_mask(x: np.ndarray, aug: AugConfig, rng) -> np.ndarray:
    """Met à zéro un segment contigu aléatoire de l'axe m/z (diversité de vue)."""
    L = x.shape[0]
    mask_frac = rng.uniform(aug.mask_frac_min, aug.mask_frac_max)
    m = int(L * mask_frac)
    if m <= 0:
        return x
    start = rng.integers(0, L - m + 1)
    x = x.copy()
    x[start:start + m] = 0.0
    return x


def _one_view(base_x: np.ndarray, aug: AugConfig, rng):
    """Construit une vue augmentée + ses coordonnées source (longueur L)."""
    L = base_x.shape[0]
    coords = _coord_map(L, aug, rng)
    x = np.interp(coords, np.arange(L), base_x)                  # resample (warp)

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
    x = _apply_mask(x, aug, rng)
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
