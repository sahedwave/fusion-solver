"""
fusion/source.py — D-T 14.1 MeV neutron source model
======================================================

Produces Q_ext[i,j,k,g]  (neutrons / s / cm^3 / group)
that feeds directly into the existing solver's external source array.

Physics
-------
The D-T fusion reaction:
    D + T  ->  4He (3.5 MeV) + n (14.1 MeV)

The 14.1 MeV neutron is placed entirely in group 0 (highest-energy group),
consistent with the G-group downscatter ordering used by sn_core:
    g=0  fast     E > 0.1 MeV   (includes 14.1 MeV D-T peak)
    g=1  epi      1 eV < E < 0.1 MeV
    g=2  thermal  E < 1 eV

Source strength normalisation (mathematical guarantee):
    sum_{i,j,k}  Q_ext[i,j,k,0] * V_cell  ==  strength   [n/s]
    where V_cell = dx * dy * dz  [cm^3]

Verified by source_strength(Q_ext, mesh) == strength.

Geometry options
----------------
"point"      — single-cell delta source at domain centre.
"volumetric" — uniform source cube of side ~ plasma_fraction * L.
"gaussian"   — 3-D isotropic Gaussian source centred on domain,
               σ = gaussian_sigma_cm.  Numerically smooth — preferred
               for fine meshes where a point source causes large
               cell-to-cell flux gradients that can slow GMRES.
               The discrete weights are normalised exactly to strength.

NO solver internals are touched or imported here.
"""

from __future__ import annotations
import numpy as np
from fusion.mesh_utils import integrate_spatial
from sn_core import dt_source_spectrum, make_spectrum_source


def make_dt_source(
    mesh,
    G: int,
    geometry: str = "point",
    strength: float = 1.0,
    plasma_fraction: float = 0.25,
    gaussian_sigma_cm: float | None = None,
    energy_bounds: np.ndarray | None = None,
    source_group_mapping: np.ndarray | None = None,
) -> np.ndarray:
    """Production Dynamic-G source builder requiring explicit metadata mapping."""
    if energy_bounds is not None:
        spectrum = dt_source_spectrum(energy_bounds)
        if spectrum.shape != (G,):
            raise ValueError(f"energy_bounds imply G={spectrum.shape[0]}, but requested G={G}")
        return make_spectrum_source(mesh, spectrum, strength=strength, geometry=geometry, plasma_fraction=plasma_fraction, gaussian_sigma_cm=gaussian_sigma_cm)
    if source_group_mapping is None:
        raise ValueError("make_dt_source requires energy_bounds or source_group_mapping in production mode. Provide explicit metadata-driven mapping.")
    spectrum = np.asarray(source_group_mapping, dtype=np.float64)
    if spectrum.shape != (G,):
        raise ValueError(f"source_group_mapping must have shape {(G,)}, got {spectrum.shape}")
    if np.any(spectrum < 0):
        raise ValueError("source_group_mapping must be nonnegative")
    total = float(np.sum(spectrum))
    if total <= 0.0:
        raise ValueError("source_group_mapping must have positive sum")
    spectrum = spectrum / total
    return make_spectrum_source(mesh, spectrum, strength=strength, geometry=geometry, plasma_fraction=plasma_fraction, gaussian_sigma_cm=gaussian_sigma_cm)


def source_strength(Q_ext: np.ndarray, mesh) -> float:
    """
    Integrate Q_ext over the full volume -> total neutron emission rate.

        S = sum_{i,j,k,g}  Q_ext[i,j,k,g] * dx * dy * dz   [n/s]

    This equals `strength` passed to make_dt_source() to floating-point
    precision.

    Returns
    -------
    float : total neutrons / s
    """
    return integrate_spatial(np.sum(Q_ext, axis=-1), mesh)
