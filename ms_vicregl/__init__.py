"""MS-VICRegL : identification bactérienne MALDI-TOF robuste aux artefacts.

CNN 1D + auto-supervision VICRegL, pré-traitement minimal (resample + TIC),
robustesse apprise via des augmentations qui simulent les artefacts de centre.
"""
# Évite le segfault OpenMP/BLAS (conflit libomp torch <-> scipy/sklearn sur macOS).
# DOIT être réglé AVANT le premier import de torch/numpy -> fait ici, en tête de package.
import os as _os
for _k, _v in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"),
               ("VECLIB_MAXIMUM_THREADS", "1"), ("KMP_DUPLICATE_LIB_OK", "TRUE")):
    _os.environ.setdefault(_k, _v)

from .config import CFG, Config, get_device

__all__ = ["CFG", "Config", "get_device"]
__version__ = "0.1.0"
