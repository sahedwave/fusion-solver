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
    G:                  int,
    geometry:           str   = "point",
    strength:           float = 1.0,
    plasma_fraction:    float = 0.25,
    gaussian_sigma_cm:  float | None = None,
    energy_bounds:      np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the D-T external source array Q_ext[i,j,k,g].

    Parameters
    ----------
    mesh : Mesh
        Solver mesh object with attributes nx, ny, nz, dx, dy, dz.
    G : int
        Number of energy groups (must match solver material).
    geometry : str
        "point"      — single cell at domain centre.
        "volumetric" — uniform source in a central plasma cube of
                       side ~ plasma_fraction * (nx*dx).
        "gaussian"   — 3-D isotropic Gaussian centred on domain,
                       width = gaussian_sigma_cm (defaults to
                       0.15 * nx * dx if not supplied).
    strength : float
        Total neutron emission rate [n/s].
        Normalisation enforced: sum_{i,j,k} Q_ext[i,j,k,0] * V_cell == strength.
    plasma_fraction : float
        Fraction of domain length used for plasma region (volumetric only).
    gaussian_sigma_cm : float or None
        Standard deviation of the Gaussian kernel [cm] (gaussian only).
        Defaults to 15 % of the domain length in x.

    Returns
    -------
    Q_ext : np.ndarray, shape (nx, ny, nz, G)  [n / cm^3 / s]
        Source in group 0 only (14.1 MeV fast group).
        All other groups are zero.

    Raises
    ------
    ValueError : bad geometry string, G < 1, or strength <= 0.
    """
    if energy_bounds is not None:
        spectrum = dt_source_spectrum(energy_bounds)
        if spectrum.shape != (G,):
            raise ValueError(
                f"energy_bounds imply G={spectrum.shape[0]}, but requested G={G}"
            )
        return make_spectrum_source(
            mesh,
            spectrum,
            strength=strength,
            geometry=geometry,
            plasma_fraction=plasma_fraction,
            gaussian_sigma_cm=gaussian_sigma_cm,
        )

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    vol_cell   = dx * dy * dz           # cm^3

    if G < 1:
        raise ValueError(f"G must be >= 1, got {G}")
    if strength <= 0.0:
        raise ValueError(f"strength must be positive, got {strength}")

    Q_ext = np.zeros((nx, ny, nz, G), dtype=np.float64)

    if geometry == "point":
        ci, cj, ck = nx // 2, ny // 2, nz // 2
        # [n/s] / [cm^3] = [n / cm^3 / s]
        Q_ext[ci, cj, ck, 0] = strength / vol_cell

    elif geometry == "volumetric":
        half = max(1, int(round(nx * plasma_fraction / 2)))
        ci, cj, ck = nx // 2, ny // 2, nz // 2
        i0, i1 = max(0, ci - half), min(nx, ci + half)
        j0, j1 = max(0, cj - half), min(ny, cj + half)
        k0, k1 = max(0, ck - half), min(nz, ck + half)
        n_cells = (i1 - i0) * (j1 - j0) * (k1 - k0)
        if n_cells == 0:
            raise RuntimeError("Volumetric plasma region has zero cells.")
        # Distribute evenly: strength / (n_cells * vol_cell) [n/cm^3/s]
        Q_ext[i0:i1, j0:j1, k0:k1, 0] = strength / (n_cells * vol_cell)

    elif geometry == "gaussian":
        # ── Gaussian kernel ─────────────────────────────────────────
        # Default σ = 15 % of the x-domain length, in physical cm.
        if gaussian_sigma_cm is None:
            gaussian_sigma_cm = 0.15 * nx * dx
        if gaussian_sigma_cm <= 0.0:
            raise ValueError(
                f"gaussian_sigma_cm must be positive, got {gaussian_sigma_cm}"
            )

        # Physical cell-centre coordinates relative to domain centre
        cx_phys = (nx / 2.0) * dx          # centre x [cm]
        cy_phys = (ny / 2.0) * dy
        cz_phys = (nz / 2.0) * dz
        xi = (np.arange(nx) + 0.5) * dx - cx_phys   # (nx,) offsets [cm]
        yj = (np.arange(ny) + 0.5) * dy - cy_phys   # (ny,)
        zk = (np.arange(nz) + 0.5) * dz - cz_phys   # (nz,)

        # Build 3-D weight array: w[i,j,k] = exp(-r^2 / (2 sigma^2))
        r2 = (xi[:, None, None] ** 2 +
              yj[None, :, None] ** 2 +
              zk[None, None, :] ** 2)                 # (nx,ny,nz)
        w = np.exp(-r2 / (2.0 * gaussian_sigma_cm ** 2))

        # Normalise so that sum_{i,j,k} w[i,j,k] * vol_cell == strength
        w_total = w.sum() * vol_cell
        if w_total == 0.0:
            raise RuntimeError("Gaussian kernel is zero everywhere — "
                               "sigma_cm is too small for this mesh.")
        Q_ext[:, :, :, 0] = w * (strength / w_total)

    else:
        raise ValueError(
            f"Unknown geometry '{geometry}'. "
            "Choose 'point', 'volumetric', or 'gaussian'."
        )

    return Q_ext


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
