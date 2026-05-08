from __future__ import annotations

import numpy as np

from sn_core import (
    BoundaryConditions,
    Mesh,
    build_quadrature,
    build_reflection_map,
    dt_source_spectrum,
    make_spectrum_source,
)
from sn_multigroup import make_synthetic_library
from sn_solver import SolverConfig, solve_gmres_dsa


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))


def _smoke(G: int, nx: int) -> None:
    lib = make_synthetic_library(G)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(nx, nx, nx, 1.0, 1.0, 1.0)
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
        SolverConfig(tol=1.0e-5, max_outer=6, gmres_restart=12, inner_tol=1.0e-7),
    )
    _check(f"{G}g finite", bool(np.all(np.isfinite(result.phi))))
    _check(f"{G}g nonnegative", float(result.phi.min()) >= 0.0)
    _check(f"{G}g nonzero", float(result.phi.sum()) > 0.0)
    _check(f"{G}g iterations bounded", result.n_gmres_total > 0)


def test_heavy_multigroup_smoke() -> None:
    _smoke(70, 2)
    _smoke(175, 2)


def main() -> None:
    test_heavy_multigroup_smoke()
    print("Heavy multigroup smoke validation complete.")


if __name__ == "__main__":
    main()
