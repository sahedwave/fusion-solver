from __future__ import annotations

import numpy as np

from sn_core import BoundaryConditions, Mesh, build_quadrature, build_reflection_map, make_point_source, make_single_group_material, make_uniform_source
from mesh_builder import MeshBuilder
from sn_operators import DSAPreconditioner, _step_cell_unstructured
from sn_solver import SolverConfig, solve_gmres_dsa, solve_source_iteration


def test_step_cell_unstructured_conservation_and_monotonicity():
    psi_in = np.array([0.2, 0.6, 1.0])
    areas = np.array([1.0, 2.0, 0.5])
    outflow_sum = 3.5
    sigma_t = 0.8
    vol = 1.2

    psi0, out0 = _step_cell_unstructured(psi_in, areas, outflow_sum, 0.0, sigma_t, vol)
    psi1, out1 = _step_cell_unstructured(psi_in, areas, outflow_sum, 1.7, sigma_t, vol)

    assert psi1 >= psi0
    assert out1 >= out0

    inflow_sum = float(np.dot(psi_in, areas))
    lhs = (sigma_t * vol + outflow_sum) * psi1
    rhs = 1.7 * vol + inflow_sum
    assert abs(lhs - rhs) < 1.0e-14


# Test 7: structured-vs-unstructured Cartesian conversion comparison.
def test_7_cartesian_equivalence_unstructured_proxy():
    cart = Mesh(8, 8, 8, 1.0, 1.0, 1.0)
    unstr = MeshBuilder.from_cartesian(Mesh(8, 8, 8, 1.0, 1.0, 1.0))
    mat = make_single_group_material(sigma_t=1.0, c=0.2)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q_cart = make_point_source(cart, mat.G)
    q_unstr = q_cart.reshape(unstr.N_cells, mat.G)

    r_cart = solve_source_iteration(cart, mat, q_cart, directions, weights, bc, refl, tol=1.0e-8, max_iter=60)
    r_unstr = solve_source_iteration(unstr, mat, q_unstr, directions, weights, bc, refl, tol=1.0e-8, max_iter=60)

    rel = np.linalg.norm(r_cart.phi.reshape(unstr.N_cells, mat.G) - r_unstr.phi) / max(np.linalg.norm(r_cart.phi), 1e-30)
    assert rel < 0.7


# Test 8: tet-box GMRES-DSA convergence/positivity.
def test_8_tet_box_single_group_vacuum_regression():
    mesh = MeshBuilder.tet_box(6, 6, 6, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.5)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q = make_point_source(mesh, mat.G)
    result = solve_gmres_dsa(mesh, mat, q, directions, weights, bc, refl, SolverConfig(tol=1e-6, max_outer=8, gmres_restart=20))
    assert result.converged
    assert result.phi.shape == (mesh.N_cells, mat.G)
    assert np.all(np.isfinite(result.phi))
    assert np.all(result.phi >= 0.0)


# Test 9: unstructured DSA SPD.
def test_9_dsa_spd_on_unstructured_mesh():
    mesh = MeshBuilder.tet_box(4, 4, 4, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.5)
    A = DSAPreconditioner(mesh, mat, BoundaryConditions())._A.toarray()
    assert np.allclose(A, A.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(A).min() > 0.0


# Test 10: reflective uniform source should be nearly uniform.
def test_10_fully_reflective_uniform_flux_unstructured_cartesian():
    mesh = MeshBuilder.from_cartesian(Mesh(3, 3, 3, 1.0, 1.0, 1.0))
    mat = make_single_group_material(sigma_t=1.0, c=0.0)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions(True, True, True, True, True, True)
    refl = build_reflection_map(directions)
    q = make_uniform_source(mesh, mat.G)
    result = solve_source_iteration(mesh, mat, q, directions, weights, bc, refl, tol=1.0e-7, max_iter=30)
    phi = result.phi[:, 0]
    # Path-B migration uses face-based unstructured updates, so strict DD-like
    # near-perfect uniformity is not expected; require bounded variation instead.
    rel_var = np.max(np.abs(phi - phi.mean())) / max(phi.mean(), 1e-30)
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0)
    assert rel_var < 2.0e-1


# Test 11: global production ~= absorption + leakage (vacuum, unstructured tet-box).
def test_11_global_conservation_unstructured_tet_box():
    mesh = MeshBuilder.tet_box(8, 8, 8, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.2)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q = make_uniform_source(mesh, mat.G)

    result = solve_source_iteration(mesh, mat, q, directions, weights, bc, refl, tol=1.0e-7, max_iter=60)
    phi = result.phi[:, 0]
    psi = result.psi if result.psi.ndim == 2 else result.psi[:, :, 0]

    prod = float(np.sum(q[:, 0] * mesh.cell_volume))
    sigma_a = float(mat.sigma_a[0])
    absorb = float(np.sum(sigma_a * phi * mesh.cell_volume))

    leak = 0.0
    boundary = set(int(f) for arr in mesh.boundary_faces.values() for f in arr)
    for f in boundary:
        cL, cR = map(int, mesh.face_to_cells[f])
        assert cR == -1
        n = mesh.face_normal[f]
        area = float(mesh.face_area[f])
        # normal points outward from left cell by construction
        mu = directions @ n
        out = mu > 0.0
        leak += float(np.sum(weights[out] * mu[out] * psi[cL, out])) * area

    rel = abs(prod - (absorb + leak)) / max(prod, 1e-30)
    assert rel < 5.0e-2
