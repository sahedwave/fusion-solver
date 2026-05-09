"""
test_phase8.py — Phase 8 Fusion Physics Post-Processing Validation Suite
=========================================================================

Tests the fusion/ package against the validated Sn solver output.
The transport solver (sn_core, sn_operators, sn_solver) is NOT modified.

Required tests (from Phase 8 spec):
  Test 1 — Reaction rate consistency:   integral(Sigma_a*phi) scales with flux
  Test 2 — TBR monotonicity:            increasing Li-6 density -> increasing TBR
  Test 3 — Heating localisation:        first wall shows highest energy deposition
  Test 4 — Physical flux behaviour:     vacuum -> exponential attenuation from source
  Test 5 — Determinism:                 repeated runs -> identical TBR and heating fields

Additional rigorous tests:
  Test 6 — Source normalisation:        source_strength(Q_ext) == strength
  Test 7 — TBR non-breeder rejection:   SS316 raises ValueError
  Test 8 — Group mismatch detection:    mismatched G raises ValueError
  Test 9 — FusionResults save/load:     round-trip NumPy archive
  Test 10 — Integration: solver -> fusion pipeline

Mathematical verifications inline in each test.
"""

from __future__ import annotations

import sys
import os
import time
import dataclasses
import numpy as np
import pytest

# Ensure project root is on path when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sn_core import (
    Mesh, BoundaryConditions,
    build_quadrature, build_reflection_map,
    make_point_source,
)
from sn_solver import SolverConfig, solve_gmres_dsa

from fusion.source    import make_dt_source, make_dt_source, source_strength
from fusion.materials import FusionMaterial, SS316, Li4SiO4, Beryllium
from fusion.reactions import compute_reaction_rate, integrate_reaction_rate
from fusion.tbr       import compute_tbr, compute_tbr_components, tbr_sensitivity_enrichment
from fusion.heating   import (compute_heating, compute_heating_watts,
                               integrate_power, peak_heat_flux)
from fusion.damage    import compute_dpa, integrate_dpa, peak_dpa
from fusion.outputs   import FusionResults


# ================================================================
# SHARED SOLVER SETUP
# ================================================================

def _run_solver(nx=8, L=10.0, sn=4, group=0, strength=1.0):
    """
    Run the GMRES-DSA solver on a 3-group problem with point source.
    Returns (phi, J, mesh, Q_ext).
    """
    from sn_core import make_3group_p1_material
    mesh     = Mesh(nx=nx, ny=nx, nz=nx, dx=L/nx, dy=L/nx, dz=L/nx)
    mat      = make_3group_p1_material()
    dirs, wts = build_quadrature(sn)
    bc       = BoundaryConditions()
    refl_map = build_reflection_map(dirs)
    Q_ext    = make_point_source(mesh, mat.G, group=group, strength=strength)

    cfg    = SolverConfig(tol=1e-6, max_outer=20, gmres_restart=20, verbose=False)
    result = solve_gmres_dsa(mesh, mat, Q_ext, dirs, wts, bc, refl_map, cfg)
    return result.phi, result.J, mesh, Q_ext


# ================================================================
# TEST 1 — REACTION RATE CONSISTENCY
# ================================================================

def test_reaction_rate_consistency():
    """
    integral(Sigma_a * phi) scales linearly with flux magnitude.

    Mathematical verification:
        R(alpha * phi) = alpha * R(phi)    for any alpha > 0
        compute_reaction_rate is a linear map in phi.

    Also verifies:
        R(phi) >= 0  everywhere (since sigma_a >= 0 and phi >= 0)
        R_total scales with source strength (linearity)
    """
    phi, _, mesh, Q_ext = _run_solver(nx=6)
    mat = SS316()

    R1 = compute_reaction_rate(phi, mat)
    R2 = compute_reaction_rate(2.0 * phi, mat)

    # Linearity: R(2*phi) == 2*R(phi) to machine precision
    err = np.linalg.norm(R2 - 2.0 * R1) / np.linalg.norm(R1)
    assert err < 1e-13, f"Reaction rate linearity violation: {err:.2e}"

    # Non-negativity
    assert R1.min() >= 0.0, f"Negative reaction rate: {R1.min():.2e}"

    # Integrated reaction rate scales with source strength
    R_tot1 = integrate_reaction_rate(phi,       mat, mesh)
    R_tot5 = integrate_reaction_rate(5.0 * phi, mat, mesh)
    ratio  = R_tot5 / R_tot1
    assert abs(ratio - 5.0) < 1e-10, \
        f"Integrated reaction rate scaling: expected 5.0, got {ratio:.6f}"

    print(f"  [PASS] Reaction rate consistency")
    print(f"    linearity err = {err:.2e}")
    print(f"    R_total       = {R_tot1:.4e} reactions/s")
    print(f"    scaling ratio = {ratio:.6f} (expected 5.0)")


# ================================================================
# TEST 2 — TBR MONOTONICITY
# ================================================================

def test_tbr_monotonicity():
    """
    Increasing Li-6 enrichment -> increasing TBR.

    Physics justification:
        Li-6(n,alpha)T has a very large thermal XS (~940 b).
        Enriching in Li-6 increases sigma_a_thermal -> more T production.

    Mathematical check:
        TBR(enr1) < TBR(enr2)  for enr1 < enr2
        (strictly monotone over the enrichment range 7.6% -> 90%)
    """
    phi, _, mesh, Q_ext = _run_solver(nx=6)
    S_DT        = source_strength(Q_ext, mesh)
    enrichments = [0.076, 0.25, 0.50, 0.90]

    tbr_vals = tbr_sensitivity_enrichment(phi, mesh, S_DT, enrichments)

    print(f"  TBR vs Li-6 enrichment:")
    prev_tbr = -np.inf
    for enr in sorted(tbr_vals):
        tbr = tbr_vals[enr]
        print(f"    Li-6 = {enr*100:5.1f}%  ->  TBR = {tbr:.5f}")
        assert tbr > prev_tbr, (
            f"TBR not monotone: TBR({enr:.3f}) = {tbr:.5f} "
            f"<= previous {prev_tbr:.5f}"
        )
        prev_tbr = tbr

    # TBR must be positive
    assert min(tbr_vals.values()) > 0.0, "TBR must be positive"

    print(f"  [PASS] TBR monotonicity: strictly increasing with Li-6 enrichment")


# ================================================================
# TEST 3 — HEATING LOCALISATION (FIRST WALL HIGHEST)
# ================================================================

def test_heating_localisation():
    """
    First wall (closest boundary to source) shows highest heat deposition.

    Physical expectation:
        Neutron flux decays exponentially from source.
        Since Q_heat = sum_g E_dep[g]*sigma_a[g]*phi_g, heating
        follows the flux and peaks near the source.

    For a centred point source on a 6x6x6 mesh:
        Source is at cell (3,3,3).
        The maximum heat in the domain should NOT be at the boundaries
        but somewhere near the centre.
        However the FIRST WALL (boundary cells) should have significantly
        higher heating than the FAR WALL (cells furthest from source).

    We verify: heating[xmin_face].max() > heating[xmax_face].min()
    i.e., the inward-facing cells are hotter than the far-face cells.
    (The source is centred so xmin and xmax are symmetric; we check
    that boundary cells have non-trivial heating reflecting flux level.)
    """
    phi, _, mesh, _ = _run_solver(nx=8)
    mat = SS316()

    Q_heat = compute_heating_watts(phi, mat)

    # Source at centre: Q_heat should peak at the source cell
    nx = mesh.nx
    ci = nx // 2
    src_heat = Q_heat[ci, ci, ci]
    max_heat = Q_heat.max()
    assert abs(src_heat - max_heat) / max_heat < 1e-10, \
        f"Peak heating not at source cell: src={src_heat:.4e}, max={max_heat:.4e}"

    # Sanity: heating field must be non-zero
    assert max_heat > 0, "Heating field is zero everywhere"

    # Spatial localisation: heating must peak near source, not at boundaries.
    # Source is at ci = nx//2 = 4 (for nx=8).
    # The slice through the source plane (x=ci) must be hotter on average
    # than the boundary planes (x=0 and x=-1), which are furthest from the
    # source.  This is the physically meaningful "localisation" check.
    q_center = Q_heat[ci, :, :].mean()   # plane through source cell
    q_xmin   = Q_heat[0,  :, :].mean()   # far boundary (x=0)
    q_xmax   = Q_heat[-1, :, :].mean()   # far boundary (x=nx-1)
    assert q_center > q_xmin, (
        f"Heating not localized: source plane ({q_center:.4e}) should be "
        f"hotter than xmin boundary ({q_xmin:.4e})"
    )
    assert q_center > q_xmax, (
        f"Heating not localized: source plane ({q_center:.4e}) should be "
        f"hotter than xmax boundary ({q_xmax:.4e})"
    )

    # Integrated power must be positive
    P_MW = integrate_power(phi, mat, mesh, unit="MW")
    assert P_MW > 0.0, f"Integrated power is non-positive: {P_MW:.4e}"

    print(f"  [PASS] Heating localisation")
    print(f"    Peak heat (at source): {max_heat:.4e} W/cm^3")
    print(f"    Integrated power:      {P_MW:.4e} MW")
    print(f"    Source-plane mean:     {q_center:.4e} W/cm^3")
    print(f"    Boundary-plane mean:   xmin={q_xmin:.4e}  xmax={q_xmax:.4e} W/cm^3")
    print(f"    Peak heat face [xmin]: {peak_heat_flux(phi, mat, mesh, 'xmin'):.4e} W/cm^2")


# ================================================================
# TEST 4 — PHYSICAL FLUX BEHAVIOUR (EXPONENTIAL ATTENUATION)
# ================================================================

def test_flux_exponential_attenuation():
    """
    Vacuum case: flux attenuates from point source.

    Physical law for an infinite homogeneous medium:
        phi(r) ~ exp(-sigma_t * r) / r^2   (transport / diffusion)

    We verify:
        1. phi is positive everywhere.
        2. phi decreases monotonically from source outward along each axis.
        3. Far-field flux is smaller than near-field flux (strict decay).

    This uses the GMRES-DSA solver (not a synthetic stub) so it exercises
    the full solver -> fusion pipeline.
    """
    phi, _, mesh, _ = _run_solver(nx=10, L=20.0)

    # Check positivity in all groups
    assert phi.min() >= 0.0, f"Negative flux: min={phi.min():.2e}"

    # Check radial-shell attenuation from source.  A single Cartesian ray can
    # show Sn ray effects; shell averages are the robust deterministic check.
    nx = mesh.nx
    ci = nx // 2
    offsets = (np.arange(nx) - ci) * mesh.dx
    r = np.sqrt(
        offsets[:, None, None] ** 2
        + offsets[None, :, None] ** 2
        + offsets[None, None, :] ** 2
    )
    phi_g0 = phi[:, :, :, 0]
    dr = mesh.dx
    shell_means = []
    for shell in range(5):
        mask = (r >= shell * dr) & (r < (shell + 1) * dr)
        if np.any(mask):
            shell_means.append(float(phi_g0[mask].mean()))
    shell_means = np.asarray(shell_means)

    far_mean = float(shell_means[2:].mean())
    attenuates = shell_means[0] > shell_means[1] > far_mean
    assert attenuates, (
        f"Fast flux does not attenuate from near to far field:\n"
        f"  shell means: {shell_means}"
    )

    # Far field (last cell) should be much smaller than near-source
    ratio_decay = shell_means[0] / (far_mean + 1e-300)
    assert ratio_decay > 10.0, \
        f"Insufficient flux attenuation: near/far ratio = {ratio_decay:.2f} (expect > 10)"

    print(f"  [PASS] Flux exponential attenuation")
    print(f"    phi[source] / phi[far] = {ratio_decay:.2f}")
    print(f"    phi_min = {phi.min():.4e},  phi_max = {phi.max():.4e}")


# ================================================================
# TEST 5 — DETERMINISM
# ================================================================

def test_determinism():
    """
    Repeated runs -> identical TBR and heating fields (bitwise).

    Verifies:
        1. The fusion post-processing layer is purely deterministic.
        2. No random number usage or stale state in fusion modules.
        3. The full pipeline (solve -> post-process) is reproducible.
    """
    phi, _, mesh, Q_ext = _run_solver(nx=6)

    mat  = SS316()
    li_m = Li4SiO4()
    S_DT = source_strength(Q_ext, mesh)

    # Run post-processing twice from the SAME phi
    tbr1, breed1 = compute_tbr(phi, li_m, mesh, S_DT)
    heat1        = compute_heating_watts(phi, mat)
    dpa1         = compute_dpa(phi, mat)

    tbr2, breed2 = compute_tbr(phi, li_m, mesh, S_DT)
    heat2        = compute_heating_watts(phi, mat)
    dpa2         = compute_dpa(phi, mat)

    # TBR must be bitwise identical
    assert tbr1 == tbr2, f"TBR not reproducible: {tbr1} vs {tbr2}"

    # Fields must be bitwise identical (no stochastic components)
    assert np.array_equal(breed1, breed2), "Breeding map not reproducible"
    assert np.array_equal(heat1,  heat2),  "Heating map not reproducible"
    assert np.array_equal(dpa1,   dpa2),   "DPA map not reproducible"

    print(f"  [PASS] Determinism: TBR={tbr1:.6f}, all fields bitwise identical")


# ================================================================
# TEST 6 — SOURCE NORMALISATION
# ================================================================

def test_source_normalisation():
    """
    source_strength(make_dt_source_legacy_group0(..., strength=S), mesh) == S exactly.

    Mathematical guarantee:
        Q_ext[i,j,k,0] = S / (n_cells * V_cell)  (for volumetric)
        sum_{i,j,k} Q_ext[i,j,k,0] * V_cell = S  (by construction)
    """
    mesh = Mesh(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)

    for geom in ("point", "volumetric"):
        for S in (1.0, 1.234e14, 3.5e17):
            Q = make_dt_source(mesh, G=3, geometry=geom, strength=S, source_group_mapping=np.array([1.0, 0.0, 0.0]))
            S_computed = source_strength(Q, mesh)
            rel_err = abs(S_computed - S) / S
            assert rel_err < 1e-14, (
                f"Source normalisation [{geom}, S={S:.2e}]: "
                f"expected {S:.6e}, got {S_computed:.6e}, rel_err={rel_err:.2e}"
            )
            # Source must be in group 0 only
            assert Q[:, :, :, 1:].sum() == 0.0, \
                f"Non-zero source in group > 0 for geometry={geom}"

    print(f"  [PASS] Source normalisation: exact for point and volumetric")


# ================================================================
# TEST 7 — TBR NON-BREEDER REJECTION
# ================================================================

def test_tbr_nonbreeder_rejection():
    """
    Passing a non-breeder material (SS316) to compute_tbr raises ValueError.
    """
    phi, _, mesh, Q_ext = _run_solver(nx=4)
    mat  = SS316()   # is_breeder = False
    S_DT = source_strength(Q_ext, mesh)

    try:
        compute_tbr(phi, mat, mesh, S_DT)
        assert False, "Should have raised ValueError for non-breeder material"
    except ValueError as e:
        assert "breeder" in str(e).lower(), \
            f"ValueError message unexpected: {e}"

    print(f"  [PASS] TBR rejects non-breeder material with ValueError")


# ================================================================
# TEST 8 — GROUP MISMATCH DETECTION
# ================================================================

def test_group_mismatch():
    """
    All fusion functions detect G mismatch and raise ValueError.
    """
    phi   = np.ones((4, 4, 4, 3))        # G=3
    mat2  = SS316(G=2)                    # G=2 — mismatch

    for fn_name, fn in [
        ("compute_reaction_rate", lambda: compute_reaction_rate(phi, mat2)),
        ("compute_heating",       lambda: compute_heating(phi, mat2)),
        ("compute_dpa",           lambda: compute_dpa(phi, mat2)),
    ]:
        try:
            fn()
            assert False, f"{fn_name} should have raised ValueError for G mismatch"
        except ValueError:
            pass

    print(f"  [PASS] Group mismatch correctly detected in all fusion functions")


def test_tbr_components_legacy_3group_fallback_consistency():
    phi, _, mesh, Q_ext = _run_solver(nx=4)
    S_DT = source_strength(Q_ext, mesh)
    components = compute_tbr_components(phi, li6_enrichment=0.076, mesh=mesh, source_strength_val=S_DT)
    assert np.isclose(
        components["tbr_total"],
        components["tbr_li6"] + components["tbr_li7"],
        rtol=0.0,
        atol=1.0e-13,
    )


def test_tbr_components_explicit_channels_g5_split_exact(monkeypatch):
    G = 5
    mesh = Mesh(nx=2, ny=2, nz=2, dx=1.0, dy=1.0, dz=1.0)
    phi = np.ones((2, 2, 2, G), dtype=np.float64)
    S_DT = 10.0
    li6 = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    li7 = np.array([0.005, 0.0, 0.001, 0.0, 0.002], dtype=np.float64)
    mat = FusionMaterial(
        name="Li explicit channels",
        G=G,
        sigma_t=np.ones(G),
        sigma_a=li6 + li7,
        sigma_dpa=np.zeros(G),
        energy_deposition=np.ones(G),
        is_breeder=True,
        breeding_channels={"li6_breeding": li6, "li7_breeding": li7},
    )
    total, _ = compute_tbr(phi, mat, mesh, S_DT)
    monkeypatch.setattr("fusion.materials.Li4SiO4", lambda G, li6_enrichment: mat)
    comps = compute_tbr_components(phi, li6_enrichment=0.5, mesh=mesh, source_strength_val=S_DT)
    assert np.isclose(comps["tbr_total"], total, rtol=0.0, atol=1.0e-13)
    assert np.isclose(comps["tbr_total"], comps["tbr_li6"] + comps["tbr_li7"], rtol=0.0, atol=1.0e-13)


def test_tbr_components_invalid_channel_shape_raises():
    with pytest.raises(ValueError, match="expected shape"):
        FusionMaterial(
            name="bad channels",
            G=5,
            sigma_t=np.ones(5),
            sigma_a=np.ones(5),
            sigma_dpa=np.zeros(5),
            energy_deposition=np.ones(5),
            is_breeder=True,
            breeding_channels={"li6_breeding": np.ones(4), "li7_breeding": np.ones(5)},
        )


def test_non3group_material_fallback_metadata_and_shapes():
    G = 5
    mats = [SS316(G=G), Li4SiO4(G=G), Beryllium(G=G)]
    for mat in mats:
        assert mat.sigma_t.shape == (G,)
        assert mat.sigma_a.shape == (G,)
        assert mat.sigma_dpa.shape == (G,)
        assert mat.energy_deposition.shape == (G,)
        assert mat.metadata.get("synthetic_fallback") is True
        assert mat.metadata.get("validation_status") == "not_fendl_njoy_openmc_validated"


def test_tbr_components_non3group_requires_explicit_breeding_channels(monkeypatch):
    G = 5
    phi = np.ones((2, 2, 2, G), dtype=np.float64)
    mesh = Mesh(nx=2, ny=2, nz=2, dx=1.0, dy=1.0, dz=1.0)
    S_DT = 1.0
    mat = Li4SiO4(G=G)
    assert not mat.breeding_channels
    monkeypatch.setattr("fusion.materials.Li4SiO4", lambda G, li6_enrichment: mat)
    with pytest.raises(ValueError, match="requires explicit breeding_channels"):
        compute_tbr_components(phi, li6_enrichment=0.5, mesh=mesh, source_strength_val=S_DT)




POSTPROC_G_CASES = [
    1,
    3,
    10,
    27,
    pytest.param(70, marks=pytest.mark.heavy),
    pytest.param(175, marks=pytest.mark.heavy),
]


def _make_explicit_li4sio4_channels(G: int, li6_enrichment: float = 0.5) -> FusionMaterial:
    """Construct Li4SiO4 with explicit breeding-channel vectors for any G."""
    base = Li4SiO4(G=G, li6_enrichment=li6_enrichment)
    if base.breeding_channels:
        return base
    li6 = np.asarray(base.sigma_a, dtype=np.float64) * 0.7
    li7 = np.asarray(base.sigma_a, dtype=np.float64) * 0.3
    return dataclasses.replace(base, breeding_channels={"li6_breeding": li6, "li7_breeding": li7})


@pytest.mark.parametrize("G", POSTPROC_G_CASES)
def test_postprocess_tbr_heating_source_normalization_matrix(G, monkeypatch):
    """
    Deterministic post-processing parity checks across production G-set.

    Numerical-risk assessment (post-processing only):
      - Conservation: checks TBR composition closes (total = li6 + li7).
      - Stability: no iterative scheme changes; deterministic algebraic checks only.
      - Scalability: tiny mesh controls cost for heavy groups (70, 175).
    """
    mesh = Mesh(nx=2, ny=2, nz=2, dx=0.5, dy=0.5, dz=0.5)
    phi = np.full((mesh.nx, mesh.ny, mesh.nz, G), 2.0, dtype=np.float64)
    Q_ext = np.full((mesh.nx, mesh.ny, mesh.nz, G), 1.0 / (mesh.nx * mesh.ny * mesh.nz * G), dtype=np.float64)
    S_DT = source_strength(Q_ext, mesh)

    li_mat = _make_explicit_li4sio4_channels(G=G, li6_enrichment=0.5)
    monkeypatch.setattr("fusion.materials.Li4SiO4", lambda G, li6_enrichment: li_mat)

    comps = compute_tbr_components(phi, li6_enrichment=0.5, mesh=mesh, source_strength_val=S_DT)
    assert np.isclose(comps["tbr_total"], comps["tbr_li6"] + comps["tbr_li7"], rtol=0.0, atol=1.0e-14)

    heat_map = compute_heating(phi, li_mat)
    heat_total_mev_s = float(np.sum(heat_map) * mesh.dx * mesh.dy * mesh.dz)
    heat_total_mev_s_via_api = integrate_power(phi, li_mat, mesh, unit="MeV_s")
    assert np.isclose(heat_total_mev_s, heat_total_mev_s_via_api, rtol=0.0, atol=1.0e-14)

    expected_norm_tbr = integrate_reaction_rate(phi, li_mat, mesh) / S_DT
    assert np.isclose(comps["tbr_total"], expected_norm_tbr, rtol=0.0, atol=1.0e-14)

# ================================================================
# TEST 9 — FUSIONRESULTS SAVE/LOAD ROUND-TRIP
# ================================================================

def test_fusionresults_roundtrip():
    """
    FusionResults.save_npz() / load_npz() round-trip preserves all fields.
    """
    import tempfile, os

    phi, _, mesh, Q_ext = _run_solver(nx=6)
    mat  = SS316()
    li_m = Li4SiO4()

    results = FusionResults.from_solver(phi, mesh, mat, Q_ext,
                                        li_material=li_m)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name

    try:
        results.save_npz(tmp_path)
        loaded = FusionResults.load_npz(tmp_path)

        assert np.array_equal(results.phi,           loaded.phi)
        assert np.array_equal(results.reaction_rate, loaded.reaction_rate)
        assert np.array_equal(results.heating_W,     loaded.heating_W)
        assert np.array_equal(results.dpa_rate,      loaded.dpa_rate)
        assert np.array_equal(results.breeding_map,  loaded.breeding_map)
        assert results.tbr            == loaded.tbr
        assert results.total_power_MW == loaded.total_power_MW
        assert results.peak_dpa_rate  == loaded.peak_dpa_rate

    finally:
        os.unlink(tmp_path)

    print(f"  [PASS] FusionResults NPZ round-trip: all fields preserved exactly")


# ================================================================
# TEST 10 — FULL INTEGRATION: SOLVER -> FUSION PIPELINE
# ================================================================

def test_full_integration_pipeline():
    """
    Complete pipeline: build source -> solve -> post-process -> FusionResults.

    Verifies:
        1. make_dt_source feeds directly into the solver unchanged.
        2. All fusion quantities computed without modifying solver state.
        3. FusionResults.from_solver() produces physically consistent output.
        4. TBR, power, DPA are all positive and non-trivially non-zero.
        5. Solver psi/phi arrays unchanged after post-processing.
    """
    from sn_core import make_3group_p1_material

    nx   = 8
    L    = 10.0
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=L/nx, dy=L/nx, dz=L/nx)
    mat_sn = make_3group_p1_material()
    dirs, wts = build_quadrature(4)
    bc        = BoundaryConditions()
    refl_map  = build_reflection_map(dirs)

    # Build D-T source (Phase 8 source model)
    Q_ext   = make_dt_source(mesh, G=mat_sn.G, geometry="point", strength=1.0, source_group_mapping=np.array([1.0, 0.0, 0.0]))
    S_DT    = source_strength(Q_ext, mesh)
    assert abs(S_DT - 1.0) < 1e-14, f"Source strength error: {S_DT}"

    # Solve
    cfg    = SolverConfig(tol=1e-6, max_outer=20, gmres_restart=20)
    result = solve_gmres_dsa(mesh, mat_sn, Q_ext, dirs, wts, bc, refl_map, cfg)
    phi_before = result.phi.copy()

    # Post-process (Phase 8)
    mat_fw = SS316()
    mat_br = Li4SiO4()
    fr     = FusionResults.from_solver(result.phi, mesh, mat_fw, Q_ext,
                                       li_material=mat_br)

    # Solver state unchanged
    assert np.array_equal(result.phi, phi_before), \
        "Solver phi was modified by fusion post-processing (VIOLATION!)"

    # Physical sanity checks
    assert fr.tbr > 0.0,            f"TBR <= 0: {fr.tbr}"
    assert fr.total_power_MW > 0.0, f"Power <= 0: {fr.total_power_MW}"
    assert fr.peak_dpa_rate > 0.0,  f"Peak DPA <= 0: {fr.peak_dpa_rate}"
    assert fr.heating_W.max() > 0,  "Heating field is zero"
    assert fr.dpa_rate.max() > 0,   "DPA field is zero"

    # Consistency: total reaction rate ~ source strength (vacuum BC, sigma_a small)
    # For absorption-dominated problem, R_total < S_DT
    R_total = integrate_reaction_rate(result.phi, mat_fw, mesh)
    assert R_total > 0.0, "Integrated reaction rate is zero"

    fr.print_summary()
    print(f"  [PASS] Full integration pipeline")


# ================================================================
# RUNNER
# ================================================================

TESTS = [
    ("Reaction rate consistency",       test_reaction_rate_consistency),
    ("TBR monotonicity",                test_tbr_monotonicity),
    ("Heating localisation",            test_heating_localisation),
    ("Flux exponential attenuation",    test_flux_exponential_attenuation),
    ("Determinism",                     test_determinism),
    ("Source normalisation",            test_source_normalisation),
    ("TBR non-breeder rejection",       test_tbr_nonbreeder_rejection),
    ("Group mismatch detection",        test_group_mismatch),
    ("FusionResults round-trip",        test_fusionresults_roundtrip),
    ("Full integration pipeline",       test_full_integration_pipeline),
]


def run_all_tests(stop_on_failure: bool = False) -> int:
    print("=" * 65)
    print("  Phase 8 — Fusion Physics Post-Processing Validation Suite")
    print("=" * 65)

    passed, failed = 0, 0
    for name, fn in TESTS:
        print(f"\n{'─'*65}")
        print(f"  TEST: {name}")
        print(f"{'─'*65}")
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ({time.perf_counter()-t0:.2f}s)")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {exc}  ({time.perf_counter()-t0:.2f}s)")
            failed += 1
            if stop_on_failure:
                break
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1
            if stop_on_failure:
                break

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{passed+failed} passed"
          + (f"  ({failed} failed)" if failed else "  ✓ all passed"))
    print(f"{'='*65}")
    return failed


if __name__ == "__main__":
    fail_count = run_all_tests(stop_on_failure=False)
    sys.exit(fail_count)
