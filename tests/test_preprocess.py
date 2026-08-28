"""Tests pour la soustraction de baseline SNIP (ms_vicregl/preprocess.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_vicregl.preprocess import snip_baseline, snip_correct


def test_snip_disabled_is_noop():
    y = np.random.default_rng(0).random(200).astype(np.float32)
    assert np.all(snip_baseline(y, 0) == 0.0)


def test_snip_flat_signal_baseline_equals_signal():
    y = np.full(100, 3.0, dtype=np.float32)
    b = snip_baseline(y, 20)
    assert np.allclose(b, y, atol=1e-6)


def test_snip_recovers_peak_over_ramp():
    L = 400
    x = np.arange(L, dtype=np.float64)
    ramp = 0.01 * x                                    # dérive lente (baseline)
    peak = 5.0 * np.exp(-0.5 * ((x - 200) / 3.0) ** 2)  # pic étroit
    y = (ramp + peak).astype(np.float32)
    baseline = snip_baseline(y, iterations=50)
    corrected = np.clip(y - baseline, 0, None)
    assert baseline[350] < baseline[50] + 10           # baseline reste croissante-ish (pas de pic)
    peak_idx = np.argmax(corrected)
    assert abs(peak_idx - 200) <= 2                    # le pic est préservé et localisé
    assert corrected[0] < 0.5                          # la baseline en dehors du pic est ~aplatie


def test_snip_correct_renormalizes_tic():
    rng = np.random.default_rng(1)
    X = rng.random((5, 300)).astype(np.float32)
    X = X / X.sum(axis=-1, keepdims=True)
    out = snip_correct(X, iterations=10, renorm_tic=True)
    sums = out.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert out.shape == X.shape
    assert (out >= 0).all()


def test_snip_correct_batched_matches_1d():
    rng = np.random.default_rng(2)
    X = rng.random((4, 250)).astype(np.float32)
    batched = snip_correct(X, iterations=15, renorm_tic=False)
    for i in range(X.shape[0]):
        single = snip_correct(X[i], iterations=15, renorm_tic=False)
        assert np.allclose(batched[i], single, atol=1e-4)
