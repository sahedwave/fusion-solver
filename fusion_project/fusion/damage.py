"""
fusion/damage.py — Radiation damage proxy (DPA field)
======================================================

Simplified NRT-model DPA proxy:

    DPA(x,y,z) proportional to  sum_g  sigma_dpa[g] * phi_g(x,y,z)

This is NOT a full NRT (Norgett-Robinson-Torrens) model.
It is a displacement-rate density [displacements/cm^3/s] that serves
as a relative indicator of radiation damage distribution and
first-wall lifetime assessment.

Physical context
----------------
For SS316 in a fusion environment:
    sigma_dpa is largest in the fast group (high-energy PKA production)
    First wall: ~10-20 dpa/year in DEMO-class reactors
    ITER first wall design limit: ~10 dpa over full-power operation

The DPA field should peak near the plasma source (fast neutrons
most concentrated there), consistent with Test 3 and Test 4.

Reference: Norgett, Robinson, Torrens, Nucl. Eng. Des. 33 (1975) 50-54.
"""

from __future__ import annotations
import numpy as np
from fusion.materials import FusionMaterial
from fusion.mesh_utils import integrate_spatial


def compute_dpa(
    phi:      np.ndarray,       # (nx, ny, nz, G)
    material: FusionMaterial,
) -> np.ndarray:
    """
    Volumetric displacement damage rate field (DPA proxy).

        DPA_rate[i,j,k] = sum_g  sigma_dpa[g] * phi[i,j,k,g]
                         [displacements / cm^3 / s]

    Parameters
    ----------
    phi : np.ndarray (nx, ny, nz, G)
        Converged scalar flux  [n/cm^2/s].
    material : FusionMaterial
        Must have sigma_dpa[G] defined.

    Returns
    -------
    DPA_rate : np.ndarray (nx, ny, nz)
        Spatial radiation damage rate  [displacements/cm^3/s].
        Use this field for:
          - Identifying the highest-damage regions (first wall)
          - Relative lifetime comparison between geometry variants
          - Input to component lifetime models

    Physical interpretation
    -----------------------
    DPA_rate is dominated by the fast group (sigma_dpa[0] largest)
    because displacement threshold energies (~40 eV) require energetic
    recoils, which are produced primarily by fast neutrons.
    Thermal neutrons have much lower DPA cross-sections.
    """
    _check_groups(phi, material)
    return np.tensordot(phi, material.sigma_dpa, axes=([-1], [0]))


def compute_group_dpa(
    phi:      np.ndarray,
    material: FusionMaterial,
) -> np.ndarray:
    """
    Per-group DPA rate (NOT group-summed).

        DPA_g[i,j,k] = sigma_dpa[g] * phi[i,j,k,g]

    Returns
    -------
    DPA_g : np.ndarray (nx, ny, nz, G)   [displacements/cm^3/s/group]

    Useful for identifying which energy group dominates damage in
    different spatial regions.
    """
    _check_groups(phi, material)
    return material.sigma_dpa.reshape((1,) * (phi.ndim - 1) + (-1,)) * phi


def integrate_dpa(
    phi:      np.ndarray,
    material: FusionMaterial,
    mesh,
) -> float:
    """
    Volume-integrate the DPA rate over the full domain.

        DPA_total = sum_{i,j,k,g}  sigma_dpa[g] * phi[i,j,k,g] * V_cell

    Returns
    -------
    float : total displacements / s (integrated over domain volume)
    """
    _check_groups(phi, material)
    DPA = compute_dpa(phi, material)
    return integrate_spatial(DPA, mesh)


def peak_dpa(
    phi:      np.ndarray,
    material: FusionMaterial,
) -> tuple[float, tuple[int, int, int]]:
    """
    Find the peak DPA rate and its location.

    Returns
    -------
    (dpa_max, (i_max, j_max, k_max))
        dpa_max  : float — peak DPA rate  [displacements/cm^3/s]
        location : (i,j,k) — grid cell index of peak damage

    Physical expectation:
    For a point source at domain centre, peak DPA occurs at the
    source cell where the fast flux is highest.
    """
    _check_groups(phi, material)
    DPA     = compute_dpa(phi, material)
    idx     = int(np.argmax(DPA))
    loc     = np.unravel_index(idx, DPA.shape)
    return float(DPA[loc]), loc


# ================================================================
# Internal
# ================================================================

def _check_groups(phi: np.ndarray, mat: FusionMaterial) -> None:
    G_phi = phi.shape[-1]
    if G_phi != mat.G:
        raise ValueError(
            f"Group mismatch: phi has G={G_phi}, "
            f"material '{mat.name}' has G={mat.G}."
        )
