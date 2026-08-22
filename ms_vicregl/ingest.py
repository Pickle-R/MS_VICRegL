"""Ingestion DRIAMS : lecture en streaming d'un tar.gz -> resample -> TIC -> memmap.

Aucun des milliers de .txt n'est écrit sur disque : on parse chaque spectre raw à la
volée et on ne conserve que le vecteur compact de longueur L. Le binned_6000 est lu
en parallèle (uniquement pour la baseline RF). Un seul passage de décompression.

Sortie dans data/processed/ :
    {center}_X.npy      (N, L) float32  -- raw resamplé + TIC (entrée du CNN)
    {center}_Xbin.npy   (N, L) float32  -- binned_6000 DRIAMS (baseline RF)
    {center}_meta.parquet               -- colonnes: code, species, center
"""
from __future__ import annotations

import os
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, GridConfig


def _basename_code(name: str) -> str:
    return os.path.basename(name)[:-4]  # retire ".txt"


def _floats_fast(s: str) -> np.ndarray:
    """Lecture rapide d'un flux de flottants séparés par des espaces, tolérante aux
    tokens corrompus (fallback ligne-à-ligne si np.fromstring échoue)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.fromstring(s, sep=" ")
        except ValueError:
            pass
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            continue                       # ignore un token non numérique isolé
    return np.asarray(out, dtype=np.float64)


def parse_raw(raw_bytes: bytes, edges: np.ndarray) -> np.ndarray:
    """Spectre brut DRIAMS -> vecteur (L,) : binning par aire (histogram) + TIC.

    Format : 2 lignes '#', 1 ligne d'entête '"mass..." "intensity..."', puis 'm/z intensité'.
    """
    txt = raw_bytes.decode("utf-8", "ignore")
    data = [ln for ln in txt.splitlines() if ln and ln[0] not in "#\""]
    if not data:
        return np.zeros(len(edges) - 1, dtype=np.float32)
    arr = _floats_fast(" ".join(data))
    if arr.size < 2:
        return np.zeros(len(edges) - 1, dtype=np.float32)
    arr = arr[: (arr.size // 2) * 2].reshape(-1, 2)
    mz, inten = arr[:, 0], arr[:, 1]
    hist, _ = np.histogram(mz, bins=edges, weights=inten)   # somme d'intensité / bin (aire)
    s = hist.sum()
    if s > 0:
        hist = hist / s                                     # normalisation TIC (somme = 1)
    return hist.astype(np.float32)


def parse_binned(bin_bytes: bytes, n_bins: int) -> np.ndarray:
    """binned_6000 DRIAMS -> vecteur (n_bins,) : 2e colonne 'binned_intensity'.

    Vectorisé : le corps est 'idx0 val0\\nidx1 val1\\n...' -> on lit le flux à plat
    et on place les valeurs par index (robuste si bins manquants/désordonnés).
    """
    txt = bin_bytes.decode("utf-8", "ignore")
    nl = txt.find("\n")
    body = txt[nl + 1:] if nl >= 0 else txt           # saute l'entête
    arr = _floats_fast(body)
    vals = np.zeros(n_bins, dtype=np.float32)
    if arr.size >= 2:
        arr = arr[: (arr.size // 2) * 2].reshape(-1, 2)
        idx = arr[:, 0].astype(np.int64)
        m = (idx >= 0) & (idx < n_bins)
        vals[idx[m]] = arr[m, 1].astype(np.float32)
    return vals


def ingest_tar(tar_path: Path, center: str, grid: GridConfig | None = None,
               out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Décompresse et ingère un tarball DRIAMS (B, C, ...). Retourne un résumé."""
    from .config import PROCESSED
    grid = grid or CFG.grid
    out_dir = out_dir or PROCESSED
    edges = grid.edges

    raw_d: dict[str, np.ndarray] = {}
    bin_d: dict[str, np.ndarray] = {}
    df_id: pd.DataFrame | None = None
    n_skipped = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:                       # itération SÉQUENTIELLE (1 seule décompression)
            if not m.isfile():
                continue
            n = m.name
            try:
                if "/id/" in n and n.endswith(".csv"):
                    df_id = pd.read_csv(tar.extractfile(m))
                elif "/raw/" in n and n.endswith(".txt"):
                    raw_d[_basename_code(n)] = parse_raw(tar.extractfile(m).read(), edges)
                elif "/binned_6000/" in n and n.endswith(".txt"):
                    bin_d[_basename_code(n)] = parse_binned(tar.extractfile(m).read(), grid.n_bins)
            except Exception as exc:        # un fichier corrompu ne doit pas tout arrêter
                n_skipped += 1
                if verbose and n_skipped <= 5:
                    print(f"  [{center}] SKIP {n}: {exc}", flush=True)
            if verbose and (len(raw_d) % 2000 == 0) and len(raw_d):
                print(f"  [{center}] {len(raw_d)} spectres raw lus...", flush=True)
    if verbose and n_skipped:
        print(f"  [{center}] {n_skipped} fichiers ignorés (illisibles)")

    if df_id is None:
        raise RuntimeError(f"Aucun CSV id trouvé dans {tar_path}")

    code2sp = {str(c): s for c, s in zip(df_id["code"], df_id["species"])}
    codes = [c for c in raw_d if isinstance(code2sp.get(c), str) and code2sp[c].strip()]
    codes.sort()
    if not codes:
        raise RuntimeError(f"Aucun spectre raw labellisé pour {center}")

    L = grid.n_bins
    X = np.stack([raw_d[c] for c in codes]).astype(np.float32)
    Xb = np.stack([bin_d.get(c, np.zeros(L, np.float32)) for c in codes]).astype(np.float32)
    meta = pd.DataFrame({"code": codes,
                         "species": [code2sp[c] for c in codes],
                         "center": center})

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{center}_X.npy", X)
    np.save(out_dir / f"{center}_Xbin.npy", Xb)
    meta.to_parquet(out_dir / f"{center}_meta.parquet")

    summary = {"center": center, "n_spectra": len(codes),
               "n_species": meta["species"].nunique(),
               "X_shape": X.shape, "out_dir": str(out_dir)}
    if verbose:
        print(f"[{center}] OK : {summary['n_spectra']} spectres, "
              f"{summary['n_species']} espèces -> {out_dir}")
    return summary
