from __future__ import annotations

import tracemalloc
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import sn_operators
from sn_core import BoundaryConditions, Mesh, P1Material, build_quadrature, build_reflection_map, make_spectrum_source
from sn_multigroup import (
    MaterialXS,
    estimate_memory_bytes,
    load_multigroup_library,
    make_sparse_synthetic_library,
    save_multigroup_library,
)
from sn_operators import _scattering_source, _scattering_source_direction_group
from sn_solver import SolverConfig, solve_gmres_dsa


def _sparse_test_matrices(G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_t = np.linspace(0.9, 1.25, G)
    sigma_s0 = np.zeros((G, G), dtype=np.float64)
    sigma_s1 = np.zeros((G, G), dtype=np.float64)
    for src in range(G):
        for out, frac in ((src, 0.22), ((src + 1) % G, 0.08), ((src - 2) % G, 0.015)):
            sigma_s0[src, out] += frac * sigma_t[src]
            sigma_s1[src, out] += (0.04 if src == out else 0.01) * sigma_t[src]
    return sigma_t, sigma_s0, sigma_s1


def test_sparse_material_exposes_dense_and_sparse_roundtrip() -> None:
    G = 9
    sigma_t, sigma_s0, sigma_s1 = _sparse_test_matrices(G)
    mat = MaterialXS(
        "sparse",
        sigma_t,
        sparse.coo_matrix(sigma_s0),
        sparse.csr_matrix(sigma_s1),
        reactions={"absorption": sigma_t - sigma_s0.sum(axis=1)},
    )
    assert np.allclose(mat.sigma_s0, sigma_s0)
    assert np.allclose(mat.sigma_s1, sigma_s1)
    assert sparse.isspmatrix_csc(mat.sigma_s0_sparse)
    assert sparse.isspmatrix_csc(mat.sigma_s1_sparse)

    lib = make_sparse_synthetic_library(12)
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in (".json", ".npz", ".h5"):
            path = Path(tmp) / f"lib{suffix}"
            save_multigroup_library(lib, path)
            got = load_multigroup_library(path)
            ref_mat = next(iter(lib.materials.values()))
            got_mat = next(iter(got.materials.values()))
            assert np.allclose(got_mat.sigma_s0, ref_mat.sigma_s0)
            assert np.allclose(got_mat.sigma_s1, ref_mat.sigma_s1)
            assert got_mat.sigma_s0_sparse.nnz == ref_mat.sigma_s0_sparse.nnz
            assert got_mat.sigma_s1_sparse.nnz == ref_mat.sigma_s1_sparse.nnz


def test_dense_vs_sparse_scattering_source_equivalence() -> None:
    rng = np.random.default_rng(1234)
    G = 11
    sigma_t, sigma_s0, sigma_s1 = _sparse_test_matrices(G)
    dense_mat = P1Material(sigma_t, sigma_s0, sigma_s1)
    sparse_mat = P1Material(sigma_t, sparse.csc_matrix(sigma_s0), sparse.csc_matrix(sigma_s1))
    directions, _ = build_quadrature(4)
    phi = rng.random((2, 2, 3, G))
    J = rng.random((2, 2, 3, G, 3))

    dense = _scattering_source(phi, J, dense_mat, directions)
    sparse_all = _scattering_source(phi, J, sparse_mat, directions)
    assert np.allclose(sparse_all, dense, rtol=0.0, atol=1.0e-13)

    for m, direction in enumerate(directions):
        for g in range(G):
            blocked = _scattering_source_direction_group(phi, J, sparse_mat, direction, g)
            assert np.allclose(blocked, dense[..., m, g], rtol=0.0, atol=1.0e-13)


def test_solver_equivalence_dense_vs_sparse_small_multigroup() -> None:
    G = 4
    sigma_t, sigma_s0, sigma_s1 = _sparse_test_matrices(G)
    dense_mat = P1Material(sigma_t, sigma_s0, sigma_s1)
    sparse_mat = P1Material(sigma_t, sparse.csr_matrix(sigma_s0), sparse.csr_matrix(sigma_s1))
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    directions, weights = build_quadrature(4)
    spectrum = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float64)
    Q = make_spectrum_source(mesh, spectrum, strength=0.8, geometry="volumetric")
    config = SolverConfig(tol=2.0e-7, max_outer=5, gmres_restart=8, inner_tol=1.0e-9)

    dense_result = solve_gmres_dsa(
        mesh, dense_mat, Q, directions, weights, BoundaryConditions(), build_reflection_map(directions), config
    )
    sparse_result = solve_gmres_dsa(
        mesh, sparse_mat, Q, directions, weights, BoundaryConditions(), build_reflection_map(directions), config
    )
    assert np.allclose(sparse_result.phi, dense_result.phi, rtol=2.0e-10, atol=2.0e-11)
    assert np.allclose(sparse_result.J, dense_result.J, rtol=2.0e-10, atol=2.0e-11)


def test_sweep_path_does_not_allocate_full_dense_scattering_source(monkeypatch: pytest.MonkeyPatch) -> None:
    G = 8
    lib = make_sparse_synthetic_library(G)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(2, 2, 1, 1.0, 1.0, 1.0)
    directions, weights = build_quadrature(4)
    transport = sn_operators.TransportOperator(mesh, mat, directions, weights, BoundaryConditions(), build_reflection_map(directions))
    phi = np.ones((mesh.nx, mesh.ny, mesh.nz, G), dtype=np.float64)
    J = np.zeros(phi.shape + (3,), dtype=np.float64)
    Q = np.zeros_like(phi)

    def fail_full_scatter(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("sweep constructed full dense Q_scatter")

    monkeypatch.setattr(sn_operators, "_scattering_source", fail_full_scatter)
    tracemalloc.start()
    try:
        _, phi_new, _ = transport.sweep(Q, phi, J)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    dense_q_bytes = mesh.nx * mesh.ny * mesh.nz * len(weights) * G * 8
    assert np.all(np.isfinite(phi_new))
    assert peak < 20 * dense_q_bytes


def test_solver_path_avoids_legacy_dense_scattering_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guardrail: production solve path must not call legacy dense _scattering_source."""
    G = 12
    lib = make_sparse_synthetic_library(G)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    directions, weights = build_quadrature(4)
    spectrum = np.linspace(1.0, 2.0, G, dtype=np.float64)
    Q = make_spectrum_source(mesh, spectrum, strength=1.0, geometry="point")
    config = SolverConfig(tol=1.0e-6, max_outer=4, gmres_restart=8, inner_tol=1.0e-8)

    def fail_full_scatter(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("solver path used legacy dense _scattering_source")

    monkeypatch.setattr(sn_operators, "_scattering_source", fail_full_scatter)
    tracemalloc.start()
    try:
        result = solve_gmres_dsa(
            mesh, mat, Q, directions, weights, BoundaryConditions(), build_reflection_map(directions), config
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    dense_q_bytes = mesh.nx * mesh.ny * mesh.nz * len(weights) * G * 8
    assert np.all(np.isfinite(result.phi))
    assert peak < 24 * dense_q_bytes


@pytest.mark.benchmark
@pytest.mark.parametrize("G", [70, 175])
def test_sparse_memory_estimate_report_for_large_group_counts(G: int) -> None:
    nnz = 4 * G
    est = estimate_memory_bytes(20, 20, 20, 80, G, scattering_nnz=nnz)
    assert est["scattering_matrix_sparse_pair"] < est["scattering_matrix_dense_pair"]
    assert est["scattering_source_blocked"] < est["scattering_source_dense"]
