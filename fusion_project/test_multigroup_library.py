from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from sn_core import (
    BoundaryConditions,
    Mesh,
    build_quadrature,
    build_reflection_map,
    dt_source_spectrum,
    make_spectrum_source,
)
from mesh_builder import MeshBuilder
from sn_multigroup import (
    MaterialXS,
    MultigroupLibrary,
    estimate_memory_bytes,
    load_multigroup_library,
    make_synthetic_library,
    save_multigroup_library,
)
from sn_operators import _scattering_source, _scattering_source_direction_group, _step_cell, _step_cell_python
from sn_solver import SolverConfig, solve_gmres_dsa


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))


def test_schema_validation() -> None:
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values()))
    _check("library G", lib.G == 10)
    _check("material conversion", mat.to_p1_material().G == 10)
    try:
        MaterialXS("bad", np.ones(3), np.ones((3, 2)), np.zeros((3, 3)))
    except ValueError:
        _check("shape mismatch rejected", True)
    else:
        raise AssertionError("shape mismatch was not rejected")


def test_json_npz_roundtrip() -> None:
    lib = make_synthetic_library(10)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for suffix in ("json", "npz"):
            path = tmp_path / f"library.{suffix}"
            save_multigroup_library(lib, path)
            loaded = load_multigroup_library(path)
            key = next(iter(lib.materials))
            _check(f"{suffix} roundtrip G", loaded.G == lib.G)
            _check(
                f"{suffix} roundtrip sigma_t",
                bool(np.allclose(loaded.materials[key].sigma_t, lib.materials[key].sigma_t)),
            )


def test_real_schema_example() -> None:
    lib = load_multigroup_library("data/multigroup/example_real_schema.json")
    _check("real schema example G", lib.G == 2)
    _check("real schema material present", "mock_steel" in lib.materials)
    _check("real schema converts", lib.materials["mock_steel"].to_p1_material().G == 2)


def test_sources() -> None:
    mesh = Mesh(4, 4, 4, 1.0, 1.0, 1.0)
    spectrum = np.arange(1, 11, dtype=np.float64)
    Q = make_spectrum_source(mesh, spectrum, strength=7.5, geometry="gaussian")
    _check("spectrum source shape", Q.shape == (4, 4, 4, 10), str(Q.shape))
    _check("spectrum source normalization", abs(float(Q.sum()) - 7.5) < 1.0e-12)

    bounds_desc = np.geomspace(2.0e7, 1.0e-5, 11)
    dt = dt_source_spectrum(bounds_desc)
    _check("D-T spectrum shape", dt.shape == (10,))
    _check("D-T spectrum normalized", float(dt.sum()) == 1.0)
    idx = int(np.argmax(dt))
    lo = min(bounds_desc[idx], bounds_desc[idx + 1])
    hi = max(bounds_desc[idx], bounds_desc[idx + 1])
    _check("D-T group contains 14.1 MeV", lo <= 14.1e6 <= hi)


def test_spectrum_source_conservation_structured_unstructured() -> None:
    spectrum = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    strength = 4.2

    # Structured reference.
    mesh_s = Mesh(4, 3, 2, 0.5, 0.75, 1.25)
    Qs = make_spectrum_source(mesh_s, spectrum, strength=strength, geometry="gaussian")
    vol_s = mesh_s.dx * mesh_s.dy * mesh_s.dz
    _check("structured spectrum shape", Qs.shape == (mesh_s.nx, mesh_s.ny, mesh_s.nz, spectrum.size))
    _check("structured source conservation", abs(float(np.sum(Qs) * vol_s) - strength) < 1.0e-12)

    # Cartesian-converted unstructured.
    mesh_c = MeshBuilder.from_cartesian(mesh_s)
    Qc = make_spectrum_source(mesh_c, spectrum, strength=strength, geometry="volumetric", plasma_fraction=0.35)
    _check("from_cartesian spectrum shape", Qc.shape == (mesh_c.N_cells, spectrum.size))
    _check("from_cartesian source conservation", abs(float(np.sum(Qc * mesh_c.cell_volume[:, None])) - strength) < 1.0e-12)

    # General tet mesh.
    mesh_t = MeshBuilder.tet_box(4, 3, 2, 1.0, 1.0, 1.0)
    Qt = make_spectrum_source(mesh_t, spectrum, strength=strength, geometry="point")
    _check("tet_box spectrum shape", Qt.shape == (mesh_t.N_cells, spectrum.size))
    _check("tet_box source conservation", abs(float(np.sum(Qt * mesh_t.cell_volume[:, None])) - strength) < 1.0e-12)


def _solver_smoke(G: int) -> None:
    lib = make_synthetic_library(G)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(3, 3, 3, 1.0, 1.0, 1.0)
    dirs, wts = build_quadrature(4)
    spectrum = dt_source_spectrum(lib.energy_bounds)
    Q = make_spectrum_source(mesh, spectrum, strength=1.0, geometry="point")
    result = solve_gmres_dsa(
        mesh,
        mat,
        Q,
        dirs,
        wts,
        BoundaryConditions(),
        build_reflection_map(dirs),
        SolverConfig(tol=1.0e-6, max_outer=8, gmres_restart=20, inner_tol=1.0e-8),
    )
    _check(f"{G}-group smoke finite", bool(np.all(np.isfinite(result.phi))))
    _check(f"{G}-group smoke nonnegative", float(result.phi.min()) >= 0.0)
    _check(f"{G}-group smoke nonzero", float(result.phi.sum()) > 0.0)


def test_solver_smoke() -> None:
    _solver_smoke(10)
    _solver_smoke(27)


def test_positivity_diagnostics() -> None:
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(3, 3, 3, 1.0, 1.0, 1.0)
    dirs, wts = build_quadrature(4)
    Q = make_spectrum_source(mesh, dt_source_spectrum(lib.energy_bounds), strength=1.0)
    result = solve_gmres_dsa(
        mesh,
        mat,
        Q,
        dirs,
        wts,
        BoundaryConditions(),
        build_reflection_map(dirs),
        SolverConfig(tol=1.0e-6, max_outer=8, gmres_restart=20, inner_tol=1.0e-8),
    )
    diag = result.positivity_diagnostics
    _check("positivity diagnostics keys", {"negative_flux_before_floor", "negative_cell_count", "clipped_flux_integral", "relative_balance_change", "rebalance_applied"} <= set(diag))
    _check("positivity diagnostics nonnegative clip", diag["clipped_flux_integral"] >= 0.0)


def test_memory_estimate() -> None:
    est = estimate_memory_bytes(10, 10, 10, 80, 175)
    _check("memory estimate keys", {"angular_flux", "scattering_source_dense", "scattering_source_blocked", "scalar_flux", "current"} <= set(est))
    _check("memory estimate positive", all(value > 0 for value in est.values()))
    _check("blocked scattering memory smaller", est["scattering_source_blocked"] < est["scattering_source_dense"])


def test_blocked_scattering_equivalence() -> None:
    rng = np.random.default_rng(42)
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values())).to_p1_material()
    directions, _ = build_quadrature(4)
    phi = rng.random((2, 3, 2, mat.G))
    J = rng.random((2, 3, 2, mat.G, 3))
    dense = _scattering_source(phi, J, mat, directions)
    max_err = 0.0
    for m, direction in enumerate(directions):
        for g in range(mat.G):
            blocked = _scattering_source_direction_group(phi, J, mat, direction, g)
            max_err = max(max_err, float(np.max(np.abs(blocked - dense[:, :, :, m, g]))))
    _check("blocked scattering equals dense", max_err < 1.0e-12, f"max_err={max_err:.2e}")


def test_step_cell_acceleration_equivalence() -> None:
    args = (0.2, 0.1, 0.05, 0.7, 1.1, 0.3, 0.4, 0.5, 2.0, 2.5, 3.0)
    ref = _step_cell_python(*args)
    got = _step_cell(*args)
    _check("step cell accelerated equivalence", bool(np.allclose(got, ref, rtol=0.0, atol=1.0e-14)))


def main() -> None:
    test_schema_validation()
    test_json_npz_roundtrip()
    test_real_schema_example()
    test_sources()
    test_solver_smoke()
    test_positivity_diagnostics()
    test_memory_estimate()
    test_blocked_scattering_equivalence()
    test_step_cell_acceleration_equivalence()
    print("Multigroup library validation complete.")


if __name__ == "__main__":
    main()
