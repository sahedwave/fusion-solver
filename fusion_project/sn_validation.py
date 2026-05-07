"""
sn_validation.py — Validation Suite (PARTISN-Style Architecture)
=================================================================

Same 10 physical tests; updated for the fully refactored operator classes
where SystemOperator.apply(phi) takes ONE argument (no external J).
Physics results are identical to the previous architecture.

Tests
-----
 1. Operator linearity      A(αφ₁ + βφ₂) = αA(φ₁) + βA(φ₂)
 2. Operator reproducibility  same input → identical output
 3. DSA positive definiteness
 4. Vacuum BC regression    GMRES-DSA ≈ SI
 5. Fully reflective → uniform flux
 6. Reflective x-faces → x-symmetry
 7. Angular flux symmetry at reflective surface
 8. Scattering ratio sweep  iteration count vs c ∈ {0.5, 0.8, 0.9, 0.95, 0.99}
 9. Mesh refinement independence
10. 3-group P1 physics sanity
"""

from __future__ import annotations

import time
import numpy as np

from sn_core import (
    Mesh, P1Material, BoundaryConditions,
    build_quadrature, build_reflection_map,
    make_single_group_material, make_3group_p1_material,
    make_point_source, make_uniform_source,
)
from sn_operators import (
    TransportOperator, ScatteringOperator,
    SystemOperator, DSAPreconditioner,
)
from sn_solver import (
    SolverConfig, SolverResult,
    build_operators, consistency_sweep,
    solve_gmres_dsa, solve_source_iteration,
)


# ================================================================
# HELPERS
# ================================================================

def _default_setup(nx=8, L=4.0, sn=4, G=1, c=0.5):
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=L/nx, dy=L/nx, dz=L/nx)
    mat  = (make_single_group_material(1.0, c) if G == 1
            else make_3group_p1_material())
    directions, weights = build_quadrature(sn)
    bc       = BoundaryConditions()
    refl_map = build_reflection_map(directions)
    return mesh, mat, directions, weights, bc, refl_map


def _run_gmres(mesh, mat, Q_ext, directions, weights, bc, refl_map,
               tol=1e-8, verbose=False) -> SolverResult:
    cfg = SolverConfig(tol=tol, max_outer=30, gmres_restart=30,
                       inner_tol=1e-10, verbose=verbose)
    return solve_gmres_dsa(mesh=mesh, mat=mat, Q_ext=Q_ext,
                           directions=directions, weights=weights,
                           bc=bc, refl_map=refl_map, cfg=cfg)


def _run_si(mesh, mat, Q_ext, directions, weights, bc, refl_map,
            tol=1e-8, verbose=False) -> SolverResult:
    return solve_source_iteration(mesh=mesh, mat=mat, Q_ext=Q_ext,
                                  directions=directions, weights=weights,
                                  bc=bc, refl_map=refl_map,
                                  tol=tol, max_iter=2000, verbose=verbose)


# ================================================================
# TEST 1: OPERATOR LINEARITY
# ================================================================

def test_operator_linearity():
    """A(αφ₁ + βφ₂) = αA(φ₁) + βA(φ₂) to machine precision.

    No manual T.reset_psi() calls — operator must be pure without them.
    """
    mesh, mat, dirs, wts, bc, rmap = _default_setup(nx=6, c=0.8)
    Q_ext = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))

    T = TransportOperator(mesh, mat, dirs, wts, bc, rmap)
    S = ScatteringOperator(mat, dirs)
    A = SystemOperator(T, S, Q_ext)

    rng  = np.random.default_rng(42)
    shp  = (mesh.nx, mesh.ny, mesh.nz, mat.G)
    phi1 = rng.standard_normal(shp)
    phi2 = rng.standard_normal(shp)
    a, b = 3.7, -1.4

    # No manual reset — apply() must manage its own state
    Ap1 = A.apply(phi1)
    Ap2 = A.apply(phi2)
    Acb = A.apply(a * phi1 + b * phi2)
    lc  = a * Ap1 + b * Ap2

    err = np.linalg.norm(Acb - lc) / np.linalg.norm(lc)
    assert err < 1e-12, f"Linearity violation: {err:.2e}"
    print(f"  [PASS] Operator linearity: relative error = {err:.2e}")


# ================================================================
# TEST 2: OPERATOR REPRODUCIBILITY
# ================================================================

def test_operator_reproducibility():
    """Same input → identical output with NO manual reset between calls.

    This is the definitive operator-purity test.  If A.apply() is truly
    pure, consecutive calls on the same φ must agree bitwise (err == 0.0)
    without any external state management.
    """
    mesh, mat, dirs, wts, bc, rmap = _default_setup(nx=6, c=0.8)
    Q_ext = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))

    T = TransportOperator(mesh, mat, dirs, wts, bc, rmap)
    S = ScatteringOperator(mat, dirs)
    A = SystemOperator(T, S, Q_ext)

    phi = np.random.default_rng(99).standard_normal(
        (mesh.nx, mesh.ny, mesh.nz, mat.G))

    # No T.reset_psi() — operator must self-manage
    out1 = A.apply(phi)
    out2 = A.apply(phi)   # psi_ang from call 1 must not leak into call 2

    err = np.linalg.norm(out1 - out2)
    assert err == 0.0, f"Reproducibility failure: {err}"
    print(f"  [PASS] Operator reproducibility: ‖Aφ₁ − Aφ₂‖ = {err:.2e}")


# ================================================================
# TEST 3: DSA POSITIVE DEFINITENESS
# ================================================================

def test_dsa_positive_definite():
    """P_DSA must be symmetric positive definite."""
    mesh = Mesh(nx=4, ny=4, nz=4, dx=1.0, dy=1.0, dz=1.0)
    mat  = make_single_group_material(1.0, 0.9)
    bc   = BoundaryConditions()

    P       = DSAPreconditioner(mesh, mat, bc, verbose=False)
    A_dense = P._assemble(mesh, mat, bc).toarray()

    sym_err = np.linalg.norm(A_dense - A_dense.T)
    assert sym_err < 1e-12, f"Not symmetric: {sym_err:.2e}"

    eigvals = np.linalg.eigvalsh(A_dense)
    lam_min = eigvals.min()
    assert lam_min > 0, f"Not positive definite: λ_min = {lam_min:.4e}"
    print(f"  [PASS] DSA SPD: sym_err={sym_err:.2e}, "
          f"λ_min={lam_min:.4e}, λ_max={eigvals.max():.4e}")


# ================================================================
# TEST 4: VACUUM BC REGRESSION
# ================================================================

def test_vacuum_regression():
    """GMRES-DSA and SI must agree on vacuum-BC problems."""
    mesh, mat, dirs, wts, bc, rmap = _default_setup(nx=8, c=0.5)
    Q_ext = make_point_source(mesh, mat.G)
    tol   = 1e-8

    r_gm = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap, tol)
    r_si = _run_si   (mesh, mat, Q_ext, dirs, wts, bc, rmap, tol)

    rel  = (np.linalg.norm(r_gm.phi - r_si.phi) /
            np.linalg.norm(r_si.phi))
    assert rel < 1e-5, f"Vacuum regression diff = {rel:.2e}"
    print(f"  [PASS] Vacuum regression: ‖φ_GMRES − φ_SI‖/‖φ_SI‖ = {rel:.2e}")


# ================================================================
# TEST 5: FULLY REFLECTIVE → UNIFORM FLUX
# ================================================================

def test_fully_reflective_uniform():
    """Uniform source in fully reflective box → spatially uniform φ."""
    nx   = 8
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=1.0, dy=1.0, dz=1.0)
    mat  = make_single_group_material(1.0, 0.8)
    bc   = BoundaryConditions(xmin=True, xmax=True, ymin=True,
                               ymax=True, zmin=True, zmax=True)
    dirs, wts = build_quadrature(4)
    rmap      = build_reflection_map(dirs)
    Q_ext     = make_uniform_source(mesh, mat.G)

    result    = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap)
    phi_g     = result.phi[:, :, :, 0]
    dev       = (phi_g.max() - phi_g.min()) / phi_g.mean()
    assert dev < 0.01, f"φ not uniform, deviation = {dev:.4e}"
    print(f"  [PASS] Fully reflective uniform: deviation = {dev:.4e}")


# ================================================================
# TEST 6: REFLECTIVE X-FACES → X-SYMMETRY
# ================================================================

def test_reflective_x_symmetry():
    """Symmetric source + reflective x-faces → φ[i] = φ[nx-1-i]."""
    nx   = 10
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=0.5, dy=0.5, dz=0.5)
    mat  = make_single_group_material(1.0, 0.8)
    bc   = BoundaryConditions(xmin=True, xmax=True)
    dirs, wts = build_quadrature(4)
    rmap      = build_reflection_map(dirs)

    Q_ext = np.zeros((nx, nx, nx, 1))
    Q_ext[nx//2-1 : nx//2+1, :, :, 0] = 1.0

    result = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap)
    phi_g  = result.phi[:, :, :, 0]
    asym   = np.abs(phi_g - phi_g[::-1, :, :]).max() / phi_g.max()
    assert asym < 1e-6, f"x-symmetry broken: {asym:.2e}"
    print(f"  [PASS] Reflective x-symmetry: max asymmetry = {asym:.2e}")


# ================================================================
# TEST 7: ANGULAR FLUX SYMMETRY AT REFLECTIVE SURFACE
# ================================================================

def test_angular_symmetry_at_surface():
    """At reflective face: ψ(Ω_m) = ψ(Ω_reflected) at boundary cells."""
    nx   = 6
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=1.0, dy=1.0, dz=1.0)
    mat  = make_single_group_material(1.0, 0.7)
    bc   = BoundaryConditions(xmin=True, xmax=True, ymin=True,
                               ymax=True, zmin=True, zmax=True)
    dirs, wts = build_quadrature(4)
    rmap      = build_reflection_map(dirs)
    Q_ext     = make_uniform_source(mesh, mat.G)

    result  = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap)
    psi     = result.psi
    mapping = rmap['xmin']
    max_err = 0.0

    for m in range(len(wts)):
        if dirs[m, 0] > 0:
            mr      = mapping[m]
            psi_in  = psi[0, :, :, m,  0]
            psi_ref = psi[0, :, :, mr, 0]
            ref_val = np.abs(psi_ref).max()
            if ref_val > 1e-14:
                err = np.abs(psi_in - psi_ref).max() / ref_val
                max_err = max(max_err, err)

    assert max_err < 0.05, f"Angular symmetry err = {max_err:.4e}"
    print(f"  [PASS] Angular flux symmetry: max relative err = {max_err:.4e}")


# ================================================================
# TEST 8: SCATTERING RATIO SWEEP
# ================================================================

def test_scattering_ratio_sweep():
    """GMRES-DSA must converge faster than SI for c ≥ 0.8."""
    c_values = [0.5, 0.8, 0.9, 0.95, 0.99]
    nx, tol  = 8, 1e-6
    print(f"\n  {'c':>6}  {'SI iters':>10}  {'GMRES outer':>12}  "
          f"{'GMRES total':>12}  {'Speedup':>9}")
    print(f"  {'-'*55}")

    for c in c_values:
        mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=0.5, dy=0.5, dz=0.5)
        mat  = make_single_group_material(1.0, c)
        bc   = BoundaryConditions()
        dirs, wts = build_quadrature(4)
        rmap      = build_reflection_map(dirs)
        Q_ext     = make_point_source(mesh, mat.G)

        r_si = _run_si   (mesh, mat, Q_ext, dirs, wts, bc, rmap, tol)
        r_gm = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap, tol)

        si_n  = r_si.n_outer
        gm_n  = r_gm.n_gmres_total
        gm_o  = r_gm.n_outer
        spdup = si_n / max(gm_n, 1)
        print(f"  {c:6.3f}  {si_n:>10}  {gm_o:>12}  {gm_n:>12}  {spdup:>8.1f}×")

        if c >= 0.8:
            assert gm_n < si_n, f"GMRES not faster than SI at c={c}"

    print(f"  [PASS] GMRES-DSA faster than SI for all c ≥ 0.8")


# ================================================================
# TEST 9: MESH REFINEMENT INDEPENDENCE
# ================================================================

def test_mesh_independence():
    """DSA preconditioning → GMRES iteration count mesh-independent."""
    nx_vals = [4, 8, 16]
    c, tol  = 0.9, 1e-6
    print(f"\n  {'nx':>6}  {'GMRES total its':>16}")
    print(f"  {'-'*26}")

    counts = []
    for nx in nx_vals:
        L    = float(nx) * 0.5
        mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=L/nx, dy=L/nx, dz=L/nx)
        mat  = make_single_group_material(1.0, c)
        bc   = BoundaryConditions()
        dirs, wts = build_quadrature(4)
        rmap      = build_reflection_map(dirs)
        Q_ext     = make_point_source(mesh, mat.G)

        r = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap, tol)
        counts.append(r.n_gmres_total)
        print(f"  {nx:6}  {r.n_gmres_total:>16}")

    ratio = max(counts) / max(min(counts), 1)
    assert ratio < 3.0, f"Mesh-dependence ratio = {ratio:.2f}"
    print(f"  [PASS] Mesh independence: max/min ratio = {ratio:.2f}")


# ================================================================
# TEST 10: 3-GROUP P1 PHYSICS
# ================================================================

def test_3group_physics():
    """3-group downscatter: peak at source cell, J≈0 at centre."""
    nx   = 12
    mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=10./nx, dy=10./nx, dz=10./nx)
    mat  = make_3group_p1_material()
    bc   = BoundaryConditions()
    dirs, wts = build_quadrature(4)
    rmap      = build_reflection_map(dirs)
    Q_ext     = make_point_source(mesh, mat.G, group=0)

    result = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap)
    phi    = result.phi
    J      = result.J
    src    = (nx//2, nx//2, nx//2)

    for g in range(mat.G):
        assert phi[:, :, :, g].max() == phi[src[0], src[1], src[2], g], \
            f"Group {g}: peak not at source cell"
        J_c = J[src[0], src[1], src[2], g, :]
        assert np.all(np.abs(J_c) < 1e-5), \
            f"Group {g}: J at centre not near zero: {J_c}"

    print(f"  [PASS] 3-group physics: peak at source, J≈0 at centre")
    for g, nm in enumerate(['fast', 'epi ', 'thrm']):
        phi_g = phi[:, :, :, g]
        print(f"    φ_{nm} max={phi_g.max():.4e}  mean={phi_g.mean():.4e}")


# ================================================================
# TEST 11: OPERATOR DETERMINISM — NO-RESET CONSECUTIVE CALLS
# ================================================================

def test_operator_determinism_no_reset():
    """
    Phase 6 mandatory test: two consecutive A.apply(phi) calls with
    NO manual reset in between must agree to ‖err‖ < 1e-12.

    Verifies the save/zero/restore contract in _pure_sweep().
    If this fails, psi_ang is still leaking across calls.
    """
    mesh, mat, dirs, wts, bc, rmap = _default_setup(nx=6, c=0.95)
    Q_ext = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))

    T = TransportOperator(mesh, mat, dirs, wts, bc, rmap)
    S = ScatteringOperator(mat, dirs)
    A = SystemOperator(T, S, Q_ext)

    rng = np.random.default_rng(2024)
    phi = rng.standard_normal((mesh.nx, mesh.ny, mesh.nz, mat.G))

    # Deliberately pollute psi_ang with a prior call on a DIFFERENT input
    phi_other = rng.standard_normal((mesh.nx, mesh.ny, mesh.nz, mat.G))
    _ = A.apply(phi_other)          # leaves psi_ang in an arbitrary state

    # Now call twice on the target phi — no reset permitted
    r1 = A.apply(phi)
    r2 = A.apply(phi)

    err = np.linalg.norm(r1 - r2)
    assert err < 1e-12, (
        f"Determinism failure (c=0.95, no reset): ‖r1-r2‖ = {err:.4e}\n"
        f"  psi_ang is still leaking across A.apply() calls."
    )
    print(f"  [PASS] Determinism (no reset, c=0.95): ‖r1-r2‖ = {err:.2e}")


# ================================================================
# TEST 12: CONSERVATION SANITY AFTER FIX
# ================================================================

def test_conservation_sanity():
    """
    Global balance: ∫ Q_ext dV  ≈  ∫ Σ_a φ dV + leakage.

    The save/zero/restore fix must not introduce artificial gain or loss.
    Check both vacuum and reflective configurations.
    """
    tol_phys = 5e-2

    for label, bc_args in [
        ("vacuum",     {}),
        ("reflective", dict(xmin=True, xmax=True)),
    ]:
        nx   = 8
        mesh = Mesh(nx=nx, ny=nx, nz=nx, dx=1.0, dy=1.0, dz=1.0)
        mat  = make_single_group_material(1.0, 0.5)
        bc   = BoundaryConditions(**bc_args)
        dirs, wts = build_quadrature(4)
        rmap      = build_reflection_map(dirs)
        Q_ext     = make_uniform_source(mesh, mat.G)   # Q = 1 everywhere

        result = _run_gmres(mesh, mat, Q_ext, dirs, wts, bc, rmap, tol=1e-8)

        # Production rate  = ∫ Q_ext dV = nx³ × 1.0
        vol_cell  = mesh.dx * mesh.dy * mesh.dz
        prod_rate = float(Q_ext.sum()) * vol_cell

        # Absorption rate = ∫ Σ_a · φ dV
        sigma_a   = float(mat.sigma_a[0])
        abs_rate  = sigma_a * float(result.phi[:, :, :, 0].sum()) * vol_cell
        psi = result.psi[:, :, :, :, 0]
        leakage = 0.0
        mu, eta, xi = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        if not bc.xmin:
            idx = np.nonzero(mu < 0)[0]
            leakage += float(np.sum(np.take(psi[0, :, :, :], idx, axis=2) * (wts[idx] * -mu[idx]))) * mesh.dy * mesh.dz
        if not bc.xmax:
            idx = np.nonzero(mu > 0)[0]
            leakage += float(np.sum(np.take(psi[-1, :, :, :], idx, axis=2) * (wts[idx] * mu[idx]))) * mesh.dy * mesh.dz
        if not bc.ymin:
            idx = np.nonzero(eta < 0)[0]
            leakage += float(np.sum(np.take(psi[:, 0, :, :], idx, axis=2) * (wts[idx] * -eta[idx]))) * mesh.dx * mesh.dz
        if not bc.ymax:
            idx = np.nonzero(eta > 0)[0]
            leakage += float(np.sum(np.take(psi[:, -1, :, :], idx, axis=2) * (wts[idx] * eta[idx]))) * mesh.dx * mesh.dz
        if not bc.zmin:
            idx = np.nonzero(xi < 0)[0]
            leakage += float(np.sum(np.take(psi[:, :, 0, :], idx, axis=2) * (wts[idx] * -xi[idx]))) * mesh.dx * mesh.dy
        if not bc.zmax:
            idx = np.nonzero(xi > 0)[0]
            leakage += float(np.sum(np.take(psi[:, :, -1, :], idx, axis=2) * (wts[idx] * xi[idx]))) * mesh.dx * mesh.dy

        rel_err = abs(abs_rate + leakage - prod_rate) / prod_rate
        assert rel_err < tol_phys, (
            f"Conservation failure ({label}): "
            f"prod={prod_rate:.4e}  abs={abs_rate:.4e}  "
            f"leak={leakage:.4e}  rel={rel_err:.2e}"
        )
        print(
            f"  [PASS] Conservation ({label:>12s}): "
            f"prod={prod_rate:.4e}  abs={abs_rate:.4e}  "
            f"leak={leakage:.4e}  rel_err={rel_err:.2e}"
        )


# ================================================================
# TEST 13: GMRES OPERATOR CONSISTENCY (matvec sees fixed A)
# ================================================================

def test_gmres_operator_consistency():
    """
    Verify that the operator seen by GMRES is fixed across all matvec calls.

    Strategy: evaluate A at three random Krylov-like vectors v1, v2, v3
    both (a) directly via A.apply() and (b) via a fresh operator built
    identically.  The results must agree to < 1e-12.

    This catches any residual state that survive the save/restore if the
    implementation is wrong.
    """
    mesh, mat, dirs, wts, bc, rmap = _default_setup(nx=6, c=0.95)
    Q_ext = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))
    shp   = (mesh.nx, mesh.ny, mesh.nz, mat.G)

    # Build the operator under test
    T = TransportOperator(mesh, mat, dirs, wts, bc, rmap)
    S = ScatteringOperator(mat, dirs)
    A = SystemOperator(T, S, Q_ext)

    # Build a reference operator (fresh instance, will always start clean)
    T_ref = TransportOperator(mesh, mat, dirs, wts, bc, rmap)
    S_ref = ScatteringOperator(mat, dirs)
    A_ref = SystemOperator(T_ref, S_ref, Q_ext)

    rng = np.random.default_rng(777)
    max_err = 0.0

    for trial in range(5):
        v = rng.standard_normal(shp)
        # Pollute A's psi_ang with a different vector first
        _ = A.apply(rng.standard_normal(shp))
        Av      = A.apply(v)
        Av_ref  = A_ref.apply(v)
        err     = np.linalg.norm(Av - Av_ref)
        max_err = max(max_err, err)

    assert max_err < 1e-12, (
        f"GMRES operator inconsistency: max ‖A(v) - A_ref(v)‖ = {max_err:.4e}"
    )
    print(
        f"  [PASS] GMRES operator consistency (5 trials, c=0.95): "
        f"max err = {max_err:.2e}"
    )


# ================================================================
# RUNNER
# ================================================================

TESTS = [
    ("Operator linearity",                    test_operator_linearity),
    ("Operator reproducibility",              test_operator_reproducibility),
    ("DSA positive definiteness",             test_dsa_positive_definite),
    ("Vacuum BC regression",                  test_vacuum_regression),
    ("Fully reflective → uniform flux",       test_fully_reflective_uniform),
    ("Reflective x-faces → x-symmetry",      test_reflective_x_symmetry),
    ("Angular flux symmetry at surface",      test_angular_symmetry_at_surface),
    ("Scattering ratio sweep",                test_scattering_ratio_sweep),
    ("Mesh refinement independence",          test_mesh_independence),
    ("3-group P1 physics",                    test_3group_physics),
    # ── Phase 6 purity tests ──────────────────────────────────────
    ("Operator determinism (no reset)",       test_operator_determinism_no_reset),
    ("Conservation sanity after fix",         test_conservation_sanity),
    ("GMRES operator consistency",            test_gmres_operator_consistency),
]


def run_all_tests(stop_on_failure: bool = False) -> None:
    print("=" * 65)
    print("  Sn GMRES-DSA Validation Suite  [PARTISN-style architecture]")
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


if __name__ == "__main__":
    run_all_tests()
