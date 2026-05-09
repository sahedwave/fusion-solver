"""
fusion/tbr.py — Tritium Breeding Ratio (TBR)
=============================================

Core equation:

    TBR = (1 / S_DT) * integral_V  Sigma_{(n,alpha),Li}(E) * phi(E) dV

Discretised form:

    TBR = [sum_{i,j,k in Li-region} sum_g  sigma_a[g] * phi[i,j,k,g] * V_cell]
          / S_DT

where:
    sigma_a[g]     = macroscopic Li breeding XS in group g  [cm^-1]
                     (encodes both Li-6(n,alpha)T and Li-7(n,n'alpha)T)
    phi[i,j,k,g]   = converged scalar flux  [n/cm^2/s]
    V_cell          = dx * dy * dz  [cm^3]
    S_DT            = total D-T neutron source strength  [n/s]

Physical context
----------------
ITER design target:       TBR >= 1.05
Typical blanket module:   TBR ~ 1.05 - 1.20
Self-sufficiency margin:  TBR > 1.0 required (accounts for losses)

For the 3-group model:
    g=0 fast:    Li-7(n,n'alpha)T threshold reaction  (smaller sigma)
    g=1 epi:     Li-6 resonance region
    g=2 thermal: Li-6(n,alpha)T dominant  (sigma ~ 1/v, large at 0.025 eV)
"""

from __future__ import annotations
import numpy as np
from fusion.materials import FusionMaterial
from fusion.mesh_utils import integrate_spatial


def compute_tbr(
    phi:                 np.ndarray,          # (nx, ny, nz, G)
    li_material:         FusionMaterial,      # must have is_breeder=True
    mesh,
    source_strength_val: float,               # total D-T source [n/s]
    li_region_mask:      np.ndarray | None = None,  # (nx,ny,nz) bool
) -> tuple[float, np.ndarray]:
    """
    Compute the Tritium Breeding Ratio and spatial breeding map.

    Parameters
    ----------
    phi : np.ndarray (nx, ny, nz, G)
        Converged scalar flux from solver  [n/cm^2/s].
    li_material : FusionMaterial
        A lithium-bearing breeder material (is_breeder must be True).
        sigma_a encodes the (n,alpha) + (n,n'alpha) breeding reactions.
    mesh : Mesh
        Solver mesh for cell volume.
    source_strength_val : float
        Total D-T neutron production rate  [n/s].
        Obtained via fusion.source.source_strength(Q_ext, mesh).
    li_region_mask : np.ndarray (nx,ny,nz) bool, optional
        Boolean mask selecting the Li-bearing region.
    legacy_group_semantics : bool, optional
        Compatibility-only legacy mode for historical 3-group Li-6/Li-7
        splitting. When True and G==3, permits the old inferred mapping
        (fast=Li-7, epi/thermal=Li-6). Default False requires explicit
        breeding_channels metadata and never infers group semantics.
        None = integrate over full domain.

    Returns
    -------
    tbr : float
        Scalar TBR (dimensionless).
        TBR > 1.0 means the blanket produces more T than consumed.
    breeding_map : np.ndarray (nx, ny, nz)
        Spatial tritium production rate  [T-atoms/cm^3/s].

    Mathematical derivation
    -----------------------
    T_production[i,j,k] = sum_g  sigma_a[g] * phi[i,j,k,g]
                         [T-atoms / cm^3 / s]

    T_total = sum_{i,j,k in mask}  T_production[i,j,k] * V_cell
            [T-atoms / s]

    TBR = T_total / S_DT   (dimensionless)

    Raises
    ------
    ValueError : if li_material.is_breeder is False or
                 source_strength_val <= 0.
    """
    if not li_material.is_breeder:
        raise ValueError(
            f"Material '{li_material.name}' is not a breeder "
            "(is_breeder=False). Use Li4SiO4() or set is_breeder=True."
        )
    if source_strength_val <= 0.0:
        raise ValueError(
            f"source_strength_val must be positive, got {source_strength_val}."
        )

    _check_groups(phi, li_material)

    # Spatial tritium production rate: sum_g sigma_a[g] * phi[...,g]
    # sigma_a for Li4SiO4 encodes both (n,alpha) and (n,n'alpha) reactions
    breeding_map = np.tensordot(phi, li_material.sigma_a, axes=([-1], [0]))

    # Apply spatial mask (restrict to Li-bearing zone)
    if li_region_mask is not None:
        breeding_map = breeding_map * li_region_mask.astype(float)

    # Volume-integrate -> total T production rate [T-atoms/s]
    T_production = integrate_spatial(breeding_map, mesh)

    # Normalise by D-T source strength -> dimensionless TBR
    tbr = T_production / source_strength_val

    return tbr, breeding_map


def compute_tbr_components(
    phi:                 np.ndarray,          # (nx, ny, nz, G)
    li6_enrichment:      float,
    mesh,
    source_strength_val: float,
    li_region_mask:      np.ndarray | None = None,
    legacy_group_semantics: bool = False,
    strict_dynamic_g: bool = False,
) -> dict:
    """
    Compute TBR split explicitly into Li-6 and Li-7 contributions.

    The spec requires separating the two breeding reactions:
        Li-6(n,alpha)T     — dominant at thermal energies, large σ
        Li-7(n,n'alpha)T   — threshold reaction (E > 2.5 MeV), fast group

    This is achieved by constructing two synthetic materials:
      - li6_only : sigma_a contains only the Li-6 contribution
                   (epi + thermal groups; fast set to 0)
      - li7_only : sigma_a contains only the Li-7 contribution
                   (fast group only; epi + thermal set to 0)

    Parameters
    ----------
    phi : np.ndarray (nx, ny, nz, G)
        Converged scalar flux  [n/cm^2/s].
    li6_enrichment : float
        Atomic fraction of Li-6.  0.076 = natural.
    mesh : Mesh
        Solver mesh for cell volume.
    source_strength_val : float
        Total D-T neutron emission rate  [n/s].
    li_region_mask : np.ndarray (nx,ny,nz) bool, optional
        Boolean mask selecting the Li-bearing region.
    legacy_group_semantics : bool, optional
        Compatibility-only legacy mode for historical 3-group Li-6/Li-7
        splitting. When True and G==3, permits the old inferred mapping
        (fast=Li-7, epi/thermal=Li-6). Default False requires explicit
        breeding_channels metadata and never infers group semantics.

    Returns
    -------
    dict with keys:
        "tbr_total"   : float  — combined TBR (identical to compute_tbr())
        "tbr_li6"     : float  — TBR from Li-6(n,alpha)T only
        "tbr_li7"     : float  — TBR from Li-7(n,n'alpha)T only
        "map_li6"     : np.ndarray (nx,ny,nz) — Li-6 breeding rate [T/cm^3/s]
        "map_li7"     : np.ndarray (nx,ny,nz) — Li-7 breeding rate [T/cm^3/s]
        "li6_fraction": float  — fraction of TBR from Li-6 (0-1)

    Notes
    -----
    Compatibility-only legacy mode (G=3 only, opt-in):
        This fallback is not external-physics validated for arbitrary ``G``.
        Li-6 reactions dominate groups 1 (epi) and 2 (thermal).
        Li-7 reactions dominate group 0 (fast) via threshold reaction.
        The split is encoded via the enrichment scaling in Li4SiO4():
          sigma_a[fast]  scales with Li-7 fraction → assigned to tbr_li7
          sigma_a[epi]   scales with Li-6 fraction → assigned to tbr_li6
          sigma_a[therm] scales with Li-6 fraction → assigned to tbr_li6

    Outside compatibility-only legacy mode, explicit ``breeding_channels``
    metadata must be provided on the breeder material (``li6_breeding`` and
    ``li7_breeding`` vectors of shape ``(G,)``). Non-legacy paths never infer
    group semantics; production claims require explicit channels/metadata
    generated for the solved energy structure.
    """
    from fusion.materials import Li4SiO4
    import copy, dataclasses

    if source_strength_val <= 0.0:
        raise ValueError(
            f"source_strength_val must be positive, got {source_strength_val}."
        )

    G = phi.shape[-1]
    nat_mat = Li4SiO4(G=G, li6_enrichment=li6_enrichment)

    channels = nat_mat.breeding_channels or {}
    if strict_dynamic_g and legacy_group_semantics:
        raise ValueError("strict_dynamic_g=True forbids legacy_group_semantics fallback; provide explicit breeding_channels.")
    if "li6_breeding" in channels or "li7_breeding" in channels:
        if "li6_breeding" not in channels or "li7_breeding" not in channels:
            raise ValueError("breeding_channels must provide both 'li6_breeding' and 'li7_breeding'")
        sigma_a_li6 = np.asarray(channels["li6_breeding"], dtype=np.float64)
        sigma_a_li7 = np.asarray(channels["li7_breeding"], dtype=np.float64)
        if sigma_a_li6.shape != (G,) or sigma_a_li7.shape != (G,):
            raise ValueError(f"breeding channel vectors must have shape {(G,)}")
    elif G == 3 and legacy_group_semantics:
        # Compatibility-only legacy mode for historical 3-group behavior.
        # Li-6 component: keep epi (g=1) and thermal (g=2), zero fast (g=0)
        sigma_a_li6 = nat_mat.sigma_a.copy()
        sigma_a_li6[0] = 0.0

        # Li-7 component: keep fast (g=0) only
        sigma_a_li7 = np.zeros(G)
        sigma_a_li7[0] = nat_mat.sigma_a[0]
    else:
        raise ValueError(
            "compute_tbr_components requires explicit breeding_channels metadata "
            "('li6_breeding' and 'li7_breeding') for production claims and all "
            "non-legacy paths. Set legacy_group_semantics=True only for "
            "compatibility-only legacy mode (G==3), which is not external-physics "
            "validated for arbitrary G."
        )

    # Build synthetic single-reaction materials by replacing sigma_a
    # dataclasses.replace keeps all other fields identical
    mat_li6 = dataclasses.replace(
        nat_mat,
        name    = f"Li4SiO4 Li-6-only ({li6_enrichment*100:.1f}%)",
        sigma_a = sigma_a_li6,
    )
    mat_li7 = dataclasses.replace(
        nat_mat,
        name    = f"Li4SiO4 Li-7-only ({li6_enrichment*100:.1f}%)",
        sigma_a = sigma_a_li7,
    )

    tbr_total, map_total = compute_tbr(phi, nat_mat, mesh,
                                       source_strength_val, li_region_mask)
    tbr_li6,   map_li6  = compute_tbr(phi, mat_li6,  mesh,
                                       source_strength_val, li_region_mask)
    tbr_li7,   map_li7  = compute_tbr(phi, mat_li7,  mesh,
                                       source_strength_val, li_region_mask)

    li6_frac = tbr_li6 / tbr_total if tbr_total > 0 else 0.0

    return {
        "tbr_total":    tbr_total,
        "tbr_li6":      tbr_li6,
        "tbr_li7":      tbr_li7,
        "map_li6":      map_li6,
        "map_li7":      map_li7,
        "li6_fraction": li6_frac,
    }


def tbr_sensitivity_enrichment(
    phi:                 np.ndarray,
    mesh,
    source_strength_val: float,
    enrichments:         list | None = None,
) -> dict:
    """
    Compute TBR as a function of Li-6 enrichment fraction.

    Verifies the monotonicity criterion:
        higher Li-6 fraction  ->  higher TBR  (Test 2 in test_phase8.py)

    Parameters
    ----------
    enrichments : list of float
        Li-6 atomic fractions to sweep.
        Default: [0.076, 0.25, 0.50, 0.90]
        (natural -> 90% enriched, as in ITER/DEMO design studies)

    Returns
    -------
    dict mapping enrichment -> TBR scalar
    """
    from fusion.materials import Li4SiO4

    if enrichments is None:
        enrichments = [0.076, 0.25, 0.50, 0.90]

    G       = phi.shape[-1]
    results = {}
    for enr in sorted(enrichments):
        mat      = Li4SiO4(G=G, li6_enrichment=enr)
        tbr, _   = compute_tbr(phi, mat, mesh, source_strength_val)
        results[enr] = tbr

    return results


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
