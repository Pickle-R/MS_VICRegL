#!/usr/bin/env python
"""Ingestion des tarballs DRIAMS présents dans data/ -> data/processed/.

Usage:
    python scripts/01_ingest.py            # ingère tous les DRIAMS_*.tar.gz trouvés
    python scripts/01_ingest.py B C        # ingère seulement ces centres
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_vicregl.config import DATA
from ms_vicregl.ingest import ingest_tar


def main(argv):
    wanted = [a.upper() for a in argv] or None
    tars = sorted(DATA.glob("DRIAMS_*.tar.gz"))
    if not tars:
        print(f"Aucun DRIAMS_*.tar.gz dans {DATA}")
        return
    for tar in tars:
        center = tar.stem.replace(".tar", "").replace("DRIAMS_", "")  # B, C, ...
        if wanted and center not in wanted:
            continue
        print(f"\n=== Ingestion {tar.name} (centre {center}) ===")
        ingest_tar(tar, center=center)


if __name__ == "__main__":
    main(sys.argv[1:])
