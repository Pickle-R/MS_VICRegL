#!/usr/bin/env python
"""Reconstruit B_Xbin.npy avec un vrai pipeline DSP lourd (sqrt -> wavelet -> SNIP ->
TIC -> binning 3 Da), en réutilisant le pipeline MSClassifPy déjà validé bit-à-bit
contre R (~/Desktop/MSClassifPy).

Pourquoi : le binned_6000 publié par DRIAMS pour le centre B est incomplet — seulement
2386/6416 fichiers `/binned_6000/` dans le tarball source (confirmé : ce n'est pas un
téléchargement corrompu, le trou est dans les données Dryad elles-mêmes). C et D sont
complets à 100%. On reconstruit donc B_Xbin en entier (les 5708 spectres labellisés,
pas seulement les 3322 manquants) pour une méthodologie homogène à l'intérieur du
centre B, plutôt que de mélanger deux pipelines différents sur le même centre.

Note méthodologique : ceci reproduit la MÊME FAMILLE de pipeline que celle décrite par
DALMA (Garcia-Navarro et al. 2026, section 2.2 : stabilisation de variance, lissage,
correction de baseline, calibration, binning) et par MSclassifR — pas nécessairement
une réplique bit-exacte de la recette originale de DRIAMS/Weis et al. pour B/C/D.

Usage:
    python scripts/11_rebuild_binned_B.py
"""
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "Desktop" / "MSClassifPy"))

import numpy as np
import pandas as pd

from ms_vicregl.config import CFG, DATA, PROCESSED
from msclassifr_py.spectrum import MassSpectrum
from msclassifr_py.preprocess import (calibrate_intensity, remove_baseline,
                                      transform_intensity, wav_smoothing)


def _parse_raw_pair(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Spectre brut DRIAMS -> (mass, intensity), SANS binning (contrairement à
    ingest.py::parse_raw, qui histogramme immédiatement)."""
    txt = raw_bytes.decode("utf-8", "ignore")
    lines = [ln for ln in txt.splitlines() if ln and ln[0] not in "#\""]
    if not lines:
        return np.array([]), np.array([])
    arr = np.fromstring(" ".join(lines), sep=" ")
    arr = arr[: (arr.size // 2) * 2].reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def main():
    tar_path = DATA / "DRIAMS_B.tar.gz"
    if not tar_path.exists():
        print(f"Introuvable : {tar_path}")
        return

    meta = pd.read_parquet(PROCESSED / "B_meta.parquet")
    wanted_codes = set(meta["code"])
    print(f"[rebuild B] {len(wanted_codes)} codes labellisés à reconstruire")

    raw_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or "/raw/" not in m.name or not m.name.endswith(".txt"):
                continue
            code = __import__("os").path.basename(m.name)[:-4]
            if code not in wanted_codes:
                continue
            mz, inten = _parse_raw_pair(tar.extractfile(m).read())
            if mz.size >= 2:
                raw_pairs[code] = (mz, inten)
    print(f"[rebuild B] {len(raw_pairs)}/{len(wanted_codes)} spectres raw extraits")

    codes = sorted(raw_pairs)  # même ordre que meta (ingest_tar trie aussi les codes)
    assert codes == sorted(meta["code"]), "ordre des codes désaligné avec B_meta.parquet"

    spectra = [MassSpectrum(raw_pairs[c][0], raw_pairs[c][1]) for c in codes]
    print("[rebuild B] DSP : sqrt -> wavelet -> SNIP -> TIC ...")
    spectra = transform_intensity(spectra, "sqrt")
    spectra = wav_smoothing(spectra, "Wavelet", 4)
    spectra = remove_baseline(spectra, "SNIP", iterations=25)
    spectra = calibrate_intensity(spectra, "TIC")

    edges = CFG.grid.edges
    n_bins = CFG.grid.n_bins
    Xb = np.zeros((len(spectra), n_bins), dtype=np.float32)
    for i, sp in enumerate(spectra):
        hist, _ = np.histogram(sp.mass, bins=edges, weights=sp.intensity)
        Xb[i] = hist.astype(np.float32)

    nz = (Xb != 0).any(axis=1)
    print(f"[rebuild B] couverture après reconstruction : {nz.mean():.4f} "
          f"({(~nz).sum()} lignes encore vides)")
    np.save(PROCESSED / "B_Xbin.npy", Xb)
    print(f"[rebuild B] -> {PROCESSED / 'B_Xbin.npy'} ({Xb.shape})")


if __name__ == "__main__":
    main()
