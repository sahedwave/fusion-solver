"""Mesh helpers for structured and unstructured fusion post-processing."""
from __future__ import annotations
import numpy as np

def cell_volumes(mesh, field=None) -> np.ndarray | float:
    if hasattr(mesh, "N_cells"):
        vols = np.asarray(mesh.cell_volume, dtype=np.float64)
        if field is None:
            return vols
        return vols.reshape((vols.size,) + (1,) * (np.asarray(field).ndim - 1))
    return float(mesh.dx * mesh.dy * mesh.dz)

def integrate_spatial(field, mesh) -> float:
    arr = np.asarray(field, dtype=np.float64)
    vols = cell_volumes(mesh, arr)
    return float(np.sum(arr * vols))
