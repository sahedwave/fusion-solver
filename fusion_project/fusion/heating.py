"""
fusion/heating.py — Neutron heating / power deposition
=======================================================

Volumetric heat deposition (kerma model):

    Q_heat(x,y,z) = sum_g  E_dep[g] * Sigma_{a,g} * phi_g(x,y,z)

where E_dep[g] is the mean energy deposited per absorption event
in group g (kerma factor, units MeV).

Unit conversion:
    1 MeV = 1.602176634e-13 J
    Q [W/cm^3] = Q [MeV * reactions/cm^3/s] * 1.602176634e-13

Physical context
----------------
First-wall neutron wall loading for ITER:
    ~ 0.5 MW/m^2  (14 MW total / 800 m^2 first-wall area)

Volumetric heating in SS316 first wall:
    ~ 0.1 - 10 W/cm^3 depending on location and reactor design.

The heating map should be highest near the plasma (source) and
decay with distance (Test 3 in test_phase8.py).
"""

from __future__ import annotations
import numpy as np
from fusion.materials import FusionMaterial
from fusion.mesh_utils import integrate_spatial

# MeV -> Joule conversion  (exact, CODATA 2018)
_MEV_TO_J: float = 1.602176634e-13   # J / MeV


def compute_heating(
    phi:      np.ndarray,       # (nx, ny, nz, G)
    material: FusionMaterial,
) -> np.ndarray:
    """
    Volumetric neutron heat deposition field.

        Q_heat[i,j,k] = sum_g  E_dep[g] * sigma_a[g] * phi[i,j,k,g]
                       [MeV / cm^3 / s]

    Parameters
    ----------
    phi : np.ndarray (nx, ny, nz, G)
        Converged scalar flux  [n/cm^2/s].
    material : FusionMaterial
        Must have energy_deposition[G] and sigma_a[G].

    Returns
    -------
    Q_heat : np.ndarray (nx, ny, nz)   [MeV / cm^3 / s]
        Use compute_heating_watts() for SI units.

    Mathematical structure
    ----------------------
    kerma_g[g] = energy_deposition[g] * sigma_a[g]   [MeV/cm]
    Q_heat     = sum_g  kerma_g[g] * phi[...,g]
    """
    _check_groups(phi, material)
    kerma_g = material.energy_deposition * material.sigma_a   # (G,) [MeV/cm]
    return np.tensordot(phi, kerma_g, axes=([-1], [0]))


def compute_heating_watts(
    phi:      np.ndarray,
    material: FusionMaterial,
) -> np.ndarray:
    """
    Volumetric heat deposition in W/cm^3.

        Q_W[i,j,k] = Q_MeV[i,j,k] * 1.602176634e-13  [W/cm^3]

    Returns
    -------
    np.ndarray (nx, ny, nz)   [W / cm^3]
    """
    return compute_heating(phi, material) * _MEV_TO_J


def integrate_power(
    phi:      np.ndarray,
    material: FusionMaterial,
    mesh,
    unit:     str = "MW",
) -> float:
    """
    Total power deposited in the material region.

        P = sum_{i,j,k}  Q_heat[i,j,k] * dx*dy*dz

    Parameters
    ----------
    unit : str
        "W"     — watts
        "MW"    — megawatts (default, appropriate for reactor-scale)
        "MeV_s" — raw MeV/s (no J conversion)

    Returns
    -------
    float : integrated power in requested unit.

    For ITER-scale geometry with 3e17 n/s, expect P ~ 0.1-100 MW.
    """
    _check_groups(phi, material)
    Q           = compute_heating(phi, material)    # [MeV/cm^3/s]
    P_MeV_per_s = integrate_spatial(Q, mesh)         # [MeV/s]

    if unit == "MeV_s":
        return P_MeV_per_s
    P_watts = P_MeV_per_s * _MEV_TO_J              # [W]
    if unit == "W":
        return P_watts
    if unit == "MW":
        return P_watts * 1e-6                       # [MW]
    raise ValueError(f"Unknown unit '{unit}'. Use 'W', 'MW', or 'MeV_s'.")


def peak_heat_flux(
    phi:      np.ndarray,
    material: FusionMaterial,
    mesh,
    face:     str = "xmin",
) -> float:
    """
    Peak surface heat flux on a specified boundary face  [W/cm^2].

    Approximates:  q_surface = Q_vol * cell_thickness

    Parameters
    ----------
    face : str
        "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"

    Returns
    -------
    float : peak surface heat flux  [W/cm^2]

    Physical check (Test 3):
    For a source at domain centre, the face closest to the source
    should exhibit the highest heat flux.
    """
    _check_groups(phi, material)
    Q_w = compute_heating_watts(phi, material)    # (...,) [W/cm^3]
    if hasattr(mesh, "N_cells"):
        return float(Q_w.max())
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    face_slices = {
        "xmin": (Q_w[0,  :, :],  dx),
        "xmax": (Q_w[-1, :, :],  dx),
        "ymin": (Q_w[:,  0, :],  dy),
        "ymax": (Q_w[:, -1, :],  dy),
        "zmin": (Q_w[:, :,  0],  dz),
        "zmax": (Q_w[:, :, -1],  dz),
    }
    if face not in face_slices:
        raise ValueError(
            f"Unknown face '{face}'. "
            "Choose from: xmin, xmax, ymin, ymax, zmin, zmax."
        )

    face_arr, thickness = face_slices[face]
    # Volumetric [W/cm^3] * thickness [cm] -> surface [W/cm^2]
    return float(face_arr.max()) * thickness


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
