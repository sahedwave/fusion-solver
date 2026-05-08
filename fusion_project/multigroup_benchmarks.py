"""Deterministic multigroup benchmark/regression harness.

The routines in this module intentionally live outside the solver and operator
layers.  They measure existing multigroup solve behavior without modifying
transport numerics, hidden source normalization, or sweep allocation strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import resource
from typing import Any, Iterable

import numpy as np

from sn_core import (
    BoundaryConditions,
    Mesh,
    build_quadrature,
    build_reflection_map,
    dt_source_spectrum,
    make_spectrum_source,
)
from sn_multigroup import make_sparse_synthetic_library
from sn_solver import SolverConfig, solve_gmres_dsa

BENCHMARK_DATA_DIR = Path(__file__).resolve().parent / "data" / "benchmarks"
DEFAULT_REPORT_PATH = BENCHMARK_DATA_DIR / "latest_multigroup_benchmark_report.json"


@dataclass(frozen=True)
class MultigroupBenchmarkCase:
    """Input definition for a deterministic synthetic multigroup benchmark."""

    name: str
    tier: str
    groups: int
    mesh_n: int
    quadrature_order: int
    solver_tol: float = 1.0e-5
    max_outer: int = 6
    gmres_restart: int = 12
    inner_tol: float = 1.0e-7
    source_strength: float = 1.0


FAST_CASES: tuple[MultigroupBenchmarkCase, ...] = (
    MultigroupBenchmarkCase("fast_10g_n2_s4", "fast", 10, 2, 4, solver_tol=1.0e-6, max_outer=8, gmres_restart=12),
    MultigroupBenchmarkCase("fast_27g_n2_s4", "fast", 27, 2, 4, solver_tol=1.0e-6, max_outer=8, gmres_restart=12),
)

HEAVY_CASES: tuple[MultigroupBenchmarkCase, ...] = (
    MultigroupBenchmarkCase("heavy_70g_n3_s4", "heavy", 70, 3, 4, solver_tol=1.0e-5, max_outer=6, gmres_restart=12),
    MultigroupBenchmarkCase("heavy_175g_n2_s4", "heavy", 175, 2, 4, solver_tol=1.0e-5, max_outer=6, gmres_restart=12),
)

ALL_CASES: tuple[MultigroupBenchmarkCase, ...] = FAST_CASES + HEAVY_CASES


def _mesh_volume(mesh: Mesh) -> float:
    return float(mesh.nx * mesh.ny * mesh.nz * mesh.dx * mesh.dy * mesh.dz)


def _integral(values: np.ndarray, mesh: Mesh) -> float:
    return float(np.sum(values) * mesh.dx * mesh.dy * mesh.dz)


def _source_conservation(Q: np.ndarray, mesh: Mesh, expected_strength: float) -> dict[str, float]:
    integrated = _integral(Q, mesh)
    return {
        "external_source_integral": integrated,
        "expected_source_strength": float(expected_strength),
        "source_strength_error": integrated - float(expected_strength),
        "relative_source_strength_error": abs(integrated - float(expected_strength)) / max(abs(float(expected_strength)), 1.0e-300),
    }


def run_multigroup_benchmark(case: MultigroupBenchmarkCase, *, collect_peak_memory: bool = True) -> dict[str, Any]:
    """Run one deterministic synthetic multigroup benchmark and return JSON metrics."""

    lib = make_sparse_synthetic_library(case.groups, name=f"benchmark_{case.groups}g")
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(case.mesh_n, case.mesh_n, case.mesh_n, 1.0, 1.0, 1.0)
    directions, weights = build_quadrature(case.quadrature_order)
    spectrum = dt_source_spectrum(lib.energy_bounds)
    Q = make_spectrum_source(mesh, spectrum, strength=case.source_strength)
    cfg = SolverConfig(
        tol=case.solver_tol,
        max_outer=case.max_outer,
        gmres_restart=case.gmres_restart,
        inner_tol=case.inner_tol,
        verbose=False,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if collect_peak_memory else None
    start = time.perf_counter()
    result = solve_gmres_dsa(
        mesh,
        mat,
        Q,
        directions,
        weights,
        BoundaryConditions(),
        build_reflection_map(directions),
        cfg,
    )
    wall_time = time.perf_counter() - start
    peak_bytes = None
    if collect_peak_memory:
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KiB; macOS reports bytes.  The test
        # environment is Linux, but keep the conversion conservative for CLI use.
        peak_bytes = int(rss_after * 1024 if rss_after < 10**10 else rss_after)
        if rss_before is not None and rss_after < rss_before:
            peak_bytes = int(rss_before * 1024 if rss_before < 10**10 else rss_before)

    phi = result.phi
    residual_final = float(result.residuals[-1]) if result.residuals else 0.0
    source = _source_conservation(Q, mesh, case.source_strength)
    scalar_flux_integral = _integral(phi, mesh)
    metrics: dict[str, Any] = {
        "schema": "fusion_multigroup_benchmark_v1",
        "case": asdict(case),
        "group_count": case.groups,
        "mesh": {
            "nx": case.mesh_n,
            "ny": case.mesh_n,
            "nz": case.mesh_n,
            "cells": int(case.mesh_n ** 3),
            "volume": _mesh_volume(mesh),
        },
        "quadrature": {
            "order": case.quadrature_order,
            "directions": int(len(directions)),
            "weight_sum": float(np.sum(weights)),
        },
        "wall_time_seconds": wall_time,
        "peak_memory_bytes": peak_bytes,
        "converged": bool(result.converged),
        "outer_iterations": int(result.n_outer),
        "gmres_iterations_total": int(result.n_gmres_total),
        "residual_final": residual_final,
        "residual_history": [float(v) for v in result.residuals],
        "scalar_flux": {
            "l1": float(np.linalg.norm(phi.ravel(), ord=1)),
            "l2": float(np.linalg.norm(phi.ravel(), ord=2)),
            "linf": float(np.linalg.norm(phi.ravel(), ord=np.inf)),
            "sum": float(np.sum(phi)),
            "integral": scalar_flux_integral,
            "min": float(np.min(phi)),
            "max": float(np.max(phi)),
            "mean": float(np.mean(phi)),
        },
        "source_conservation": source,
        "positivity": result.positivity_diagnostics,
        "library": {
            "synthetic": True,
            "physics_validation": False,
            "description": "Synthetic software benchmark; not an experimental or benchmark-physics validation case.",
            "sigma_s0_nnz": int(mat.sigma_s0_sparse.nnz),
            "sigma_s1_nnz": int(mat.sigma_s1_sparse.nnz),
        },
    }
    return metrics


def run_cases(cases: Iterable[MultigroupBenchmarkCase], *, collect_peak_memory: bool = True) -> dict[str, Any]:
    results = [run_multigroup_benchmark(case, collect_peak_memory=collect_peak_memory) for case in cases]
    return {
        "schema": "fusion_multigroup_benchmark_report_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }


def select_cases(tier: str) -> tuple[MultigroupBenchmarkCase, ...]:
    if tier == "fast":
        return FAST_CASES
    if tier == "heavy":
        return HEAVY_CASES
    if tier == "all":
        return ALL_CASES
    raise ValueError(f"unknown benchmark tier {tier!r}")


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic multigroup benchmark tiers.")
    parser.add_argument("--tier", choices=("fast", "heavy", "all"), default="fast", help="benchmark tier to run")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH, help="JSON report output path")
    parser.add_argument("--no-memory", action="store_true", help="disable peak RSS memory collection")
    args = parser.parse_args(argv)

    report = run_cases(select_cases(args.tier), collect_peak_memory=not args.no_memory)
    write_report(report, args.output)
    print(f"Wrote {len(report['results'])} benchmark result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
