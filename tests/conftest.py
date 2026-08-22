"""Réglages d'environnement chargés par pytest AVANT tout import numpy/torch.

Empêche le segfault OpenMP/BLAS (torch <-> scipy/sklearn) sur macOS.
"""
import os

for k, v in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
             ("VECLIB_MAXIMUM_THREADS", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE")):
    os.environ.setdefault(k, v)
