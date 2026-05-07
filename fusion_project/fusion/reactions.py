"""
fusion/reactions.py — Neutron reaction rate computation
========================================================

Core equation:

    R(x,y,z) = sum_g  Sigma_{a,g}(x,y,z) * phi_g(x,y,z)

Units:  phi [n/cm^2/s],  sigma_a [cm^-1]  ->  R [reactions/cm^3/s]

This is the fundamental quantity from which TBR, heating, and DPA
are all derived.  Kept as a standalone module so each downstream
calculation can call it independently.
"""

from __future__ import annotations
import numpy as np
from fusion.materials import FusionMaterial


def compute_reaction_rate(
    phi:      np.ndarray,    # (nx, ny, nz, G)
    material: FusionMaterial,
) -> np.ndarray:
    """
    Group-summed absorption reaction rate field.

        R[i,j,k] = sum_g  sigma_a[g] * phi[i,j,k,g]

    Parameters
    ----------
    phi : np.ndarray, shape (nx, ny, nz, G)
        Scalar flux from the converged solver  [n/cm^2/s].
    material : FusionMaterial
        Must have sigma_a[G] defined.

    Returns
    -------
    R : np.ndarray, shape (nx, ny, nz)
        Volumetric absorption reaction rate  [reactions/cm^3/s].

    Mathematical verification
    -------------------------
    Total reaction rate in domain:
        R_total = sum_{i,j,k} R[i,j,k] * dx*dy*dz
    Must scale linearly with flux magnitude (Test 1 in test_phase8.py).
    """
    _check_groups(phi, material)
    return np.einsum('g,ijkg->ijk', material.sigma_a, phi)


def compute_group_reaction_rates(
    phi:      np.ndarray,    # (nx, ny, nz, G)
    material: FusionMaterial,
) -> np.ndarray:
    """
    Per-group reaction rate field (NOT group-summed).

        R_g[i,j,k] = sigma_a[g] * phi[i,j,k,g]

    Returns
    -------
    R_g : np.ndarray, shape (nx, ny, nz, G)   [reactions/cm^3/s/group]
    """
    _check_groups(phi, material)
    return material.sigma_a[np.newaxis, np.newaxis, np.newaxis, :] * phi


def integrate_reaction_rate(
    phi:      np.ndarray,
    material: FusionMaterial,
    mesh,
) -> float:
    """
    Volume-integrate the reaction rate over the full domain.

        R_total = sum_{i,j,k,g}  sigma_a[g] * phi[i,j,k,g] * dx*dy*dz

    Returns
    -------
    float : total reactions / s

    Physical check: in steady state (no fission) this equals the total
    neutron absorption rate, which must balance the source strength for
    vacuum boundary conditions.
    """
    _check_groups(phi, material)
    R   = compute_reaction_rate(phi, material)
    vol = mesh.dx * mesh.dy * mesh.dz
    return float(R.sum()) * vol


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
