"""SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping, Ryan et al. 1988) :
soustraction de baseline SANS lissage ni transformée LLS, pour ne retirer que la
dérive lente (matrice/détecteur) sans toucher à la forme des pics -- seule
étape ajoutée au pipeline "raw resample + TIC" (cf. ingest.py), donc opt-in
(snip_iterations=0 par défaut = comportement inchangé).
"""
from __future__ import annotations

import numpy as np


def snip_baseline(y: np.ndarray, iterations: int) -> np.ndarray:
    """Estime la baseline par clipping itératif à fenêtre croissante.

    À l'itération m, chaque point est ramené au min(lui-même, moyenne de ses
    voisins à +/- m) -- les bords sont clampés (pas de wrap/reflect). Aucun
    lissage intermédiaire : c'est la forme la plus simple de l'algorithme.

    y : (..., L), dernière dimension = axe m/z. Retourne la baseline, même forme.
    """
    if iterations <= 0:
        return np.zeros_like(y)
    v = y.astype(np.float64, copy=True)
    L = v.shape[-1]
    idx = np.arange(L)
    for m in range(1, iterations + 1):
        li = np.clip(idx - m, 0, L - 1)
        ri = np.clip(idx + m, 0, L - 1)
        avg = 0.5 * (v[..., li] + v[..., ri])
        v = np.minimum(v, avg)
    return v.astype(y.dtype)


def snip_correct(X: np.ndarray, iterations: int, renorm_tic: bool = True) -> np.ndarray:
    """Soustrait la baseline SNIP puis (par défaut) renormalise TIC.

    Clippe les résidus négatifs à 0 avant renormalisation (la baseline est une
    borne inférieure locale, la soustraction ne peut pas produire de négatif
    aux points de contact mais peut ailleurs à cause de la non-linéarité du min).
    """
    baseline = snip_baseline(X, iterations)
    corrected = np.clip(X - baseline, 0.0, None)
    if renorm_tic:
        s = corrected.sum(axis=-1, keepdims=True)
        corrected = np.divide(corrected, s, out=np.zeros_like(corrected), where=s > 0)
    return corrected.astype(X.dtype)
