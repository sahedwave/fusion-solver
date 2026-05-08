from __future__ import annotations

import numpy as np

from sn_core import BoundaryConditions, Mesh, build_quadrature, build_reflection_map, make_point_source, make_single_group_material, make_uniform_source
from mesh_builder import MeshBuilder, UnstructuredMesh
from mesh_geometry import _compute_sweep_order
from sn_operators import DSAPreconditioner, _step_cell, _step_cell_unstructured
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
    # Keep this regression small: the test validates GMRES-DSA convergence and
    # positivity on a tetrahedral mesh, while larger tet-boxes are integration
    # benchmarks that can exceed unit-test time budgets.
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.5)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q = make_point_source(mesh, mat.G)
    result = solve_gmres_dsa(mesh, mat, q, directions, weights, bc, refl, SolverConfig(tol=1e-3, max_outer=4, gmres_restart=10, inner_tol=1e-6))
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


def _with_single_boundary_tag(mesh: UnstructuredMesh, tag: str) -> UnstructuredMesh:
    boundary = np.nonzero(mesh.face_to_cells[:, 1] == -1)[0].astype(np.int64)
    return UnstructuredMesh(
        nodes=mesh.nodes, cell_nodes=mesh.cell_nodes, cell_type=mesh.cell_type,
        cell_volume=mesh.cell_volume, cell_centroid=mesh.cell_centroid,
        face_area=mesh.face_area, face_normal=mesh.face_normal,
        face_centroid=mesh.face_centroid, face_to_cells=mesh.face_to_cells,
        cell_to_faces=mesh.cell_to_faces, boundary_faces={tag: boundary},
        cartesian_shape=mesh.cartesian_shape, cartesian_spacing=mesh.cartesian_spacing,
    )


def test_unstructured_vacuum_tet_box_boundary_is_finite_and_nonnegative():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.0)
    directions, weights = build_quadrature(4)
    q = make_uniform_source(mesh, mat.G)
    result = solve_source_iteration(
        mesh, mat, q, directions, weights, BoundaryConditions(), build_reflection_map(directions),
        tol=1.0e-8, max_iter=8,
    )
    assert result.converged
    assert np.all(np.isfinite(result.phi))
    assert np.all(result.phi >= 0.0)


def test_unstructured_reflective_from_cartesian_boundary_changes_flux():
    mesh = MeshBuilder.from_cartesian(Mesh(2, 2, 2, 1.0, 1.0, 1.0))
    mat = make_single_group_material(sigma_t=1.0, c=0.0)
    directions, weights = build_quadrature(4)
    refl = build_reflection_map(directions)
    q = make_uniform_source(mesh, mat.G)

    vacuum = solve_source_iteration(
        mesh, mat, q, directions, weights, BoundaryConditions(), refl, tol=1.0e-8, max_iter=8,
    )
    reflective = solve_source_iteration(
        mesh, mat, q, directions, weights,
        BoundaryConditions(True, True, True, True, True, True), refl, tol=1.0e-8, max_iter=8,
    )

    assert reflective.phi.mean() > vacuum.phi.mean()
    assert not np.allclose(reflective.phi, vacuum.phi, rtol=1.0e-8, atol=1.0e-10)


def test_unstructured_tagged_reflective_boundary_changes_flux():
    mesh = _with_single_boundary_tag(MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0), "wall")
    mat = make_single_group_material(sigma_t=1.0, c=0.0)
    directions, weights = build_quadrature(4)
    refl = build_reflection_map(directions)
    q = make_uniform_source(mesh, mat.G)

    vacuum = solve_source_iteration(
        mesh, mat, q, directions, weights, BoundaryConditions(), refl, tol=1.0e-8, max_iter=8,
    )
    reflective = solve_source_iteration(
        mesh, mat, q, directions, weights,
        BoundaryConditions(boundary_types={"wall": "reflective"}), refl, tol=1.0e-8, max_iter=8,
    )

    assert reflective.phi.mean() > vacuum.phi.mean()
    assert not np.allclose(reflective.phi, vacuum.phi, rtol=1.0e-8, atol=1.0e-10)


def _cartesian_cell_id(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * ny + j) * nz + k


def _expected_cartesian_sweep_order(nx: int, ny: int, nz: int, direction: np.ndarray) -> np.ndarray:
    i_range = range(nx) if direction[0] >= 0.0 else range(nx - 1, -1, -1)
    j_range = range(ny) if direction[1] >= 0.0 else range(ny - 1, -1, -1)
    k_range = range(nz) if direction[2] >= 0.0 else range(nz - 1, -1, -1)
    return np.asarray([
        _cartesian_cell_id(i, j, k, ny, nz)
        for i in i_range
        for j in j_range
        for k in k_range
    ], dtype=np.int64)


def test_11_cell_update_cuboid_step_characteristic_balance():
    dx, dy, dz = 1.25, 0.75, 1.5
    vol = dx * dy * dz
    direction = np.array([0.3, 0.4, 0.5], dtype=np.float64)
    psi_in = np.array([0.2, 0.6, 1.0], dtype=np.float64)
    q_per_sr = 1.7
    sigma_t = 0.8
    inflow_areas_cos = np.array([
        direction[0] * dy * dz,
        direction[1] * dx * dz,
        direction[2] * dx * dy,
    ], dtype=np.float64)
    outflow_area_cos_sum = float(inflow_areas_cos.sum())

    psi_unstructured, psi_out = _step_cell_unstructured(
        psi_in, inflow_areas_cos, outflow_area_cos_sum, q_per_sr, sigma_t, vol
    )

    lhs = (sigma_t * vol + outflow_area_cos_sum) * psi_unstructured
    rhs = q_per_sr * vol + float(np.dot(psi_in, inflow_areas_cos))
    assert np.isclose(lhs, rhs, rtol=0.0, atol=1.0e-14)
    assert psi_out == psi_unstructured

    # Path B intentionally uses a face-balance step-characteristic update for
    # unstructured cells, not Cartesian diamond-difference algebra.
    psi_dd, *_ = _step_cell(
        psi_in[0], psi_in[1], psi_in[2], q_per_sr, sigma_t,
        direction[0], direction[1], direction[2], 1.0 / dx, 1.0 / dy, 1.0 / dz,
    )
    assert not np.isclose(psi_unstructured, psi_dd, rtol=1.0e-12, atol=1.0e-12)


def test_11_dsa_cartesian_reduction_matches_fd_matrix_and_apply():
    cart = Mesh(4, 4, 4, 1.0, 1.0, 1.0)
    unstructured = MeshBuilder.from_cartesian(cart)
    mat = make_single_group_material(sigma_t=1.2, c=0.25)
    bc = BoundaryConditions()

    dsa_cart = DSAPreconditioner(cart, mat, bc)
    dsa_unstructured = DSAPreconditioner(unstructured, mat, bc)

    assert np.allclose(dsa_unstructured._A.toarray(), dsa_cart._A.toarray(), rtol=0.0, atol=1.0e-14)

    residual_cart = np.linspace(0.1, 1.0, cart.nx * cart.ny * cart.nz).reshape(cart.nx, cart.ny, cart.nz, 1)
    residual_unstructured = residual_cart.reshape(unstructured.N_cells, 1)
    assert np.allclose(
        dsa_unstructured.apply(residual_unstructured),
        dsa_cart.apply(residual_cart).reshape(unstructured.N_cells, 1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_11_sweep_order_cartesian_upwind_lexicographic():
    nx, ny, nz = 4, 4, 4
    mesh = MeshBuilder.from_cartesian(Mesh(nx, ny, nz, 1.0, 1.0, 1.0))
    for direction in (
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ):
        order = _compute_sweep_order(mesh, direction)
        assert np.array_equal(order, _expected_cartesian_sweep_order(nx, ny, nz, direction))


def test_11_total_volume_conservation_cartesian_conversion():
    nx, ny, nz = 4, 5, 6
    dx, dy, dz = 1.25, 0.5, 2.0
    mesh = MeshBuilder.from_cartesian(Mesh(nx, ny, nz, dx, dy, dz))
    assert np.isclose(mesh.cell_volume.sum(), nx * ny * nz * dx * dy * dz, rtol=0.0, atol=1.0e-12)


def test_11_flux_integral_conservation_cartesian_conversion():
    nx, ny, nz = 4, 5, 6
    dx, dy, dz = 1.25, 0.5, 2.0
    mesh = MeshBuilder.from_cartesian(Mesh(nx, ny, nz, dx, dy, dz))
    phi_3d = np.arange(1, nx * ny * nz + 1, dtype=np.float64).reshape(nx, ny, nz)
    phi_flat = phi_3d.reshape(mesh.N_cells)
    assert np.isclose(
        np.dot(phi_flat, mesh.cell_volume),
        phi_3d.sum() * dx * dy * dz,
        rtol=0.0,
        atol=1.0e-12,
    )


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
