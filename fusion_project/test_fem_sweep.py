from __future__ import annotations

import numpy as np

from sn_core import BoundaryConditions, Mesh, build_quadrature, build_reflection_map, make_point_source, make_single_group_material, make_uniform_source
from mesh_builder import MeshBuilder
from sn_operators import DSAPreconditioner
from sn_solver import SolverConfig, solve_gmres_dsa, solve_source_iteration


def test_cartesian_equivalence_unstructured_proxy():
    cart = Mesh(4, 4, 4, 1.0, 1.0, 1.0)
    unstr = MeshBuilder.from_cartesian(cart)
    mat = make_single_group_material(sigma_t=1.0, c=0.2)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q_cart = make_point_source(cart, mat.G)
    q_unstr = q_cart.reshape(unstr.N_cells, mat.G)
    r_cart = solve_source_iteration(cart, mat, q_cart, directions, weights, bc, refl, tol=1.0e-8, max_iter=50)
    r_unstr = solve_source_iteration(unstr, mat, q_unstr, directions, weights, bc, refl, tol=1.0e-8, max_iter=50)
    diff = np.linalg.norm(r_cart.phi.reshape(unstr.N_cells, mat.G) - r_unstr.phi)
    assert diff / np.linalg.norm(r_cart.phi) < 1.0e-10


def test_tet_box_single_group_vacuum_regression():
    mesh = MeshBuilder.tet_box(3, 3, 3, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.5)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions()
    refl = build_reflection_map(directions)
    q = make_point_source(mesh, mat.G)
    result = solve_gmres_dsa(mesh, mat, q, directions, weights, bc, refl, SolverConfig(max_outer=3, gmres_restart=20))
    assert result.phi.shape == (mesh.N_cells, mat.G)
    assert np.all(result.phi >= 0.0)


def test_dsa_spd_on_unstructured_mesh():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    mat = make_single_group_material(sigma_t=1.0, c=0.5)
    A = DSAPreconditioner(mesh, mat, BoundaryConditions())._A.toarray()
    assert np.allclose(A, A.T, atol=1.0e-12)
    assert np.linalg.eigvalsh(A).min() > 0.0


def test_fully_reflective_uniform_flux_unstructured_cartesian():
    mesh = MeshBuilder.from_cartesian(Mesh(3, 3, 3, 1.0, 1.0, 1.0))
    mat = make_single_group_material(sigma_t=1.0, c=0.0)
    directions, weights = build_quadrature(4)
    bc = BoundaryConditions(True, True, True, True, True, True)
    refl = build_reflection_map(directions)
    q = make_uniform_source(mesh, mat.G)
    result = solve_source_iteration(mesh, mat, q, directions, weights, bc, refl, tol=1.0e-7, max_iter=20)
    phi = result.phi[:, 0]
    assert np.max(np.abs(phi - phi.mean())) / phi.mean() < 1.0e-2
