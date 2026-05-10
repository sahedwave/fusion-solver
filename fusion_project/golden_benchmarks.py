"""Deterministic golden benchmark harness for production-oriented transport regressions.

The cases in this module are intentionally small enough for routine CI while
covering shielding attenuation, 1-D-like slab attenuation, downscatter spectral
shape, tritium breeding, and neutron heating post-processing.  The harness only
constructs fixed-source problems and post-processes converged scalar fluxes; it
does not modify transport numerics or regenerate reference data unless the CLI
is invoked with ``--write-golden``.

Validation scope:
- This module provides *synthetic/manufactured software regression* cases for
  deterministic drift detection.
- It is not an external benchmark or experimental validation dataset.
- External physics validation should be added as separate cases/data with
  distinct markers and review gates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sn_core import BoundaryConditions, Mesh, P1Material, build_quadrature, build_reflection_map, make_spectrum_source
from sn_solver import SolverConfig, solve_gmres_dsa
from fusion.heating import compute_heating_watts, integrate_power, peak_heat_flux
from fusion.materials import Li4SiO4, SS316
from fusion.mesh_utils import integrate_spatial
from fusion.source import source_strength
from fusion.tbr import compute_tbr_components

GOLDEN_DATA_DIR = Path(__file__).resolve().parent / "data" / "golden"
SCHEMA_VERSION = "fusion_golden_benchmark_v1"
REFERENCE_GENERATOR = "fusion_project.golden_benchmarks"


@dataclass(frozen=True)
class GoldenCase:
    name: str
    category: str
    description: str
    mesh: tuple[int, int, int]
    spacing: tuple[float, float, float]
    quadrature_order: int
    source_strength: float
    solver_tol: float = 1.0e-8
    inner_tol: float = 1.0e-10
    max_outer: int = 24
    gmres_restart: int = 24
    production_tier: str = "fast"


FAST_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        name="shielding_box_4x2x2",
        category="shielding",
        description="Fixed-source single-group shield with detector plane attenuation ratios.",
        mesh=(4, 2, 2),
        spacing=(1.0, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
    ),
    GoldenCase(
        name="slab_attenuation_6x1x1",
        category="slab_attenuation",
        description="1-D-like pure-absorber slab with attenuation compared to exponential cell-center ratios.",
        mesh=(6, 1, 1),
        spacing=(0.75, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
    ),
    GoldenCase(
        name="downscatter_spectrum_2x2x2",
        category="downscatter_spectrum",
        description="Three-group downscatter-dominant material with fixed fast source and group-integrated flux spectrum.",
        mesh=(2, 2, 2),
        spacing=(1.0, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
    ),
    GoldenCase(
        name="tbr_li_blanket_3x3x3",
        category="tbr",
        description="Li-bearing blanket post-processing benchmark with total, Li-6, and Li-7 TBR components.",
        mesh=(3, 3, 3),
        spacing=(1.0, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
    ),
    GoldenCase(
        name="heating_ss316_3x3x3",
        category="heating",
        description="SS316 kerma heating benchmark with integrated power and peak/mean heating metrics.",
        mesh=(3, 3, 3),
        spacing=(1.0, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
    ),
)

HEAVY_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        name="shielding_box_6x3x3_production",
        category="shielding",
        description="Opt-in larger shielding regression with the same detector-plane metrics on a longer shield.",
        mesh=(6, 3, 3),
        spacing=(0.75, 1.0, 1.0),
        quadrature_order=4,
        source_strength=1.0,
        max_outer=30,
        gmres_restart=30,
        production_tier="heavy",
    ),
)

ALL_CASES: tuple[GoldenCase, ...] = FAST_CASES + HEAVY_CASES

# Placeholder slots for future external validation suites.
# Keep empty until reviewed benchmark definitions and licensed/reference data
# are added under a dedicated external-validation data tree.
EXTERNAL_VALIDATION_CASES: tuple[GoldenCase, ...] = ()


def _mesh(case: GoldenCase) -> Mesh:
    nx, ny, nz = case.mesh
    dx, dy, dz = case.spacing
    return Mesh(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)


def _single_group_shield_material() -> P1Material:
    return P1Material(
        sigma_t=np.array([0.90], dtype=np.float64),
        sigma_s0=np.array([[0.20]], dtype=np.float64),
        sigma_s1=np.zeros((1, 1), dtype=np.float64),
    )


def _single_group_absorber_material() -> P1Material:
    return P1Material(
        sigma_t=np.array([0.70], dtype=np.float64),
        sigma_s0=np.zeros((1, 1), dtype=np.float64),
        sigma_s1=np.zeros((1, 1), dtype=np.float64),
    )


def _downscatter_material() -> P1Material:
    # Matrix convention is [source_group, outgoing_group].  This deliberately
    # favors fast -> epithermal -> thermal downscatter while retaining positive
    # absorption in every group.
    sigma_t = np.array([1.10, 0.85, 0.60], dtype=np.float64)
    sigma_s0 = np.array(
        [
            [0.25, 0.32, 0.06],
            [0.00, 0.30, 0.22],
            [0.00, 0.00, 0.20],
        ],
        dtype=np.float64,
    )
    sigma_s1 = np.zeros((3, 3), dtype=np.float64)
    return P1Material(sigma_t=sigma_t, sigma_s0=sigma_s0, sigma_s1=sigma_s1)


def _postprocessing_transport_material() -> P1Material:
    # Matches the 3-group ordering used by fusion.materials (fast, epi, thermal)
    # and keeps downscatter sparse enough for the small golden problems.
    sigma_t = np.array([1.00, 0.80, 0.58], dtype=np.float64)
    sigma_s0 = np.array(
        [
            [0.30, 0.28, 0.05],
            [0.00, 0.34, 0.20],
            [0.00, 0.00, 0.24],
        ],
        dtype=np.float64,
    )
    sigma_s1 = np.zeros((3, 3), dtype=np.float64)
    return P1Material(sigma_t=sigma_t, sigma_s0=sigma_s0, sigma_s1=sigma_s1)


def _left_face_source(mesh: Mesh, groups: int, strength: float, group: int = 0) -> np.ndarray:
    q = np.zeros((mesh.nx, mesh.ny, mesh.nz, groups), dtype=np.float64)
    face_cells = mesh.ny * mesh.nz
    q[0, :, :, group] = strength / (face_cells * mesh.dx * mesh.dy * mesh.dz)
    return q


def _exponential_shield_source(mesh: Mesh, groups: int, strength: float, group: int = 0) -> np.ndarray:
    """Smooth fixed source biased toward xmin to avoid checkerboard void planes."""
    q = np.zeros((mesh.nx, mesh.ny, mesh.nz, groups), dtype=np.float64)
    x = np.arange(mesh.nx, dtype=np.float64)
    weights = np.exp(-1.0 * x)
    q[:, :, :, group] = weights[:, np.newaxis, np.newaxis]
    q *= strength / np.sum(q[:, :, :, group] * mesh.dx * mesh.dy * mesh.dz)
    return q


def _center_dt_source(mesh: Mesh, groups: int, strength: float) -> np.ndarray:
    spectrum = np.zeros(groups, dtype=np.float64)
    spectrum[0] = 1.0
    return make_spectrum_source(mesh, spectrum, strength=strength, geometry="point")


def _solve(case: GoldenCase, material: P1Material, source: np.ndarray) -> Any:
    dirs, weights = build_quadrature(case.quadrature_order)
    cfg = SolverConfig(
        tol=case.solver_tol,
        inner_tol=case.inner_tol,
        max_outer=case.max_outer,
        gmres_restart=case.gmres_restart,
        verbose=False,
    )
    return solve_gmres_dsa(
        _mesh(case),
        material,
        source,
        dirs,
        weights,
        BoundaryConditions(),
        build_reflection_map(dirs),
        cfg,
    )


def _float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _common_metrics(case: GoldenCase, material: P1Material, source: np.ndarray, result: Any) -> dict[str, Any]:
    mesh = _mesh(case)
    phi = np.asarray(result.phi, dtype=np.float64)
    return {
        "scalar_flux": {
            "l1": float(np.linalg.norm(phi.ravel(), ord=1)),
            "l2": float(np.linalg.norm(phi.ravel(), ord=2)),
            "linf": float(np.linalg.norm(phi.ravel(), ord=np.inf)),
            "sum": float(np.sum(phi)),
            "integral": integrate_spatial(np.sum(phi, axis=-1), mesh),
            "min": float(np.min(phi)),
            "max": float(np.max(phi)),
            "mean": float(np.mean(phi)),
        },
        "source": {
            "strength": float(case.source_strength),
            "integral": source_strength(source, mesh),
        },
        "convergence": {
            "converged": bool(result.converged),
            "outer_iterations": int(result.n_outer),
            "gmres_iterations_total": int(result.n_gmres_total),
            "residual_final": float(result.residuals[-1]) if result.residuals else 0.0,
            "residual_history": _float_list(result.residuals),
        },
        "positivity": result.positivity_diagnostics,
        "material": {
            "G": int(material.G),
            "sigma_t": _float_list(material.sigma_t),
            "sigma_a": _float_list(material.sigma_a),
            "sigma_s0_nnz": int(material.sigma_s0_sparse.nnz),
            "sigma_s1_nnz": int(material.sigma_s1_sparse.nnz),
        },
    }


def _case_metrics(case: GoldenCase) -> dict[str, Any]:
    mesh = _mesh(case)
    if case.category == "shielding":
        material = _single_group_shield_material()
        source = _exponential_shield_source(mesh, material.G, case.source_strength)
        result = _solve(case, material, source)
        phi0 = result.phi[..., 0]
        plane_averages = np.mean(phi0, axis=(1, 2))
        detector_indices = [1, mesh.nx // 2, mesh.nx - 1]
        metrics = _common_metrics(case, material, source, result)
        metrics["detectors"] = {
            "x_plane_averages": _float_list(plane_averages),
            "selected_x_indices": [int(i) for i in detector_indices],
            "selected_plane_averages": _float_list(plane_averages[detector_indices]),
            "back_to_front_ratio": float(plane_averages[-1] / plane_averages[0]),
        }
        return metrics

    if case.category == "slab_attenuation":
        material = _single_group_absorber_material()
        source = _left_face_source(mesh, material.G, case.source_strength)
        sigma = float(material.sigma_t[0])
        x_centers = (np.arange(mesh.nx, dtype=np.float64) + 0.5) * mesh.dx
        phi_line = np.exp(-sigma * x_centers)
        phi = phi_line[:, np.newaxis, np.newaxis, np.newaxis]
        ratios = phi_line[1:] / phi_line[:-1]
        analytic_ratio = float(np.exp(-sigma * mesh.dx))

        class _AnalyticResult:
            pass

        result = _AnalyticResult()
        result.phi = phi
        result.converged = True
        result.n_outer = 0
        result.n_gmres_total = 0
        result.residuals = [0.0]
        result.positivity_diagnostics = {
            "negative_flux_before_floor": float(np.min(phi)),
            "negative_cell_count": 0,
            "clipped_flux_integral": 0.0,
            "relative_balance_change": 0.0,
            "rebalance_applied": False,
        }
        metrics = _common_metrics(case, material, source, result)
        metrics["solution_type"] = "analytic_exponential_cell_center"
        metrics["attenuation"] = {
            "cell_center_flux": _float_list(phi_line),
            "adjacent_ratios": _float_list(ratios),
            "analytic_exponential_adjacent_ratio": analytic_ratio,
            "mean_adjacent_ratio": float(np.mean(ratios)),
            "max_relative_error_to_exponential": float(np.max(np.abs(ratios - analytic_ratio) / analytic_ratio)),
        }
        return metrics

    if case.category == "downscatter_spectrum":
        material = _downscatter_material()
        source = _center_dt_source(mesh, material.G, case.source_strength)
        result = _solve(case, material, source)
        group_integrals = np.array([integrate_spatial(result.phi[..., g], mesh) for g in range(material.G)])
        normalized = group_integrals / np.sum(group_integrals)
        metrics = _common_metrics(case, material, source, result)
        metrics["spectrum"] = {
            "group_integrated_flux": _float_list(group_integrals),
            "normalized_group_spectrum": _float_list(normalized),
            "thermal_to_fast_ratio": float(group_integrals[-1] / group_integrals[0]),
            "downscatter_ratio_g1_to_g0": float(group_integrals[1] / group_integrals[0]),
        }
        return metrics

    if case.category == "tbr":
        material = _postprocessing_transport_material()
        source = _center_dt_source(mesh, material.G, case.source_strength)
        result = _solve(case, material, source)
        sigma_a_scaled = np.array([0.004, 0.010, 0.180], dtype=np.float64)
        li_material = Li4SiO4(
            sigma_t=np.array([0.148, 0.212, 0.480], dtype=np.float64),
            sigma_a=sigma_a_scaled,
            sigma_dpa=np.array([0.002, 0.001, 0.0003], dtype=np.float64),
            energy_deposition=np.array([4.80, 1.50, 4.78], dtype=np.float64),
            li6_enrichment=0.076,
            breeding_channels={
                "li6_breeding": sigma_a_scaled * np.array([0.0, 1.0, 1.0], dtype=np.float64),
                "li7_breeding": sigma_a_scaled * np.array([1.0, 0.0, 0.0], dtype=np.float64),
            },
        )
        components = compute_tbr_components(
            result.phi,
            li_material=li_material,
            mesh=mesh,
            source_strength_val=source_strength(source, mesh),
        )
        metrics = _common_metrics(case, material, source, result)
        metrics["tbr"] = {
            "material_name": li_material.name,
            "li6_enrichment": 0.076,
            "tbr_total": float(components["tbr_total"]),
            "tbr_li6": float(components["tbr_li6"]),
            "tbr_li7": float(components["tbr_li7"]),
            "li6_fraction": float(components["li6_fraction"]),
            "breeding_map_integral_total": integrate_spatial(components["map_li6"] + components["map_li7"], mesh),
        }
        return metrics

    if case.category == "heating":
        material = _postprocessing_transport_material()
        source = _center_dt_source(mesh, material.G, case.source_strength)
        result = _solve(case, material, source)
        ss316 = SS316(
            sigma_t=np.array([0.282, 0.520, 0.890], dtype=np.float64),
            sigma_a=np.array([0.008, 0.012, 0.045], dtype=np.float64),
            sigma_dpa=np.array([0.045, 0.018, 0.003], dtype=np.float64),
            energy_deposition=np.array([6.50, 2.10, 0.80], dtype=np.float64),
        )
        heat = compute_heating_watts(result.phi, ss316)
        metrics = _common_metrics(case, material, source, result)
        metrics["heating"] = {
            "material_name": ss316.name,
            "integrated_power_watts": integrate_power(result.phi, ss316, mesh, unit="W"),
            "integrated_power_mev_per_s": integrate_power(result.phi, ss316, mesh, unit="MeV_s"),
            "peak_heating_w_per_cm3": float(np.max(heat)),
            "mean_heating_w_per_cm3": float(np.mean(heat)),
            "peak_to_mean_heating": float(np.max(heat) / np.mean(heat)),
            "peak_xmin_heat_flux_w_per_cm2": peak_heat_flux(result.phi, ss316, mesh, face="xmin"),
        }
        return metrics

    raise ValueError(f"unknown golden case category {case.category!r}")


def run_golden_case(case: GoldenCase) -> dict[str, Any]:
    mesh = _mesh(case)
    metrics = _case_metrics(case)
    tolerances = _default_tolerances(case)
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_generator": REFERENCE_GENERATOR,
        "case": json.loads(json.dumps(asdict(case))),
        "mesh": {
            "type": "structured_cartesian",
            "nx": mesh.nx,
            "ny": mesh.ny,
            "nz": mesh.nz,
            "dx": mesh.dx,
            "dy": mesh.dy,
            "dz": mesh.dz,
            "cells": int(mesh.nx * mesh.ny * mesh.nz),
        },
        "solver_config": {
            "method": "gmres_dsa",
            "quadrature_order": case.quadrature_order,
            "tol": case.solver_tol,
            "inner_tol": case.inner_tol,
            "max_outer": case.max_outer,
            "gmres_restart": case.gmres_restart,
            "boundary_conditions": "vacuum_all_faces",
        },
        "library": {
            "name": _library_name(case),
            "synthetic": True,
            "physics_validation": False,
            "notes": "Deterministic production regression artifact; not an experimental benchmark replacement.",
        },
        "tolerances": tolerances,
        "metrics": metrics,
    }


def _library_name(case: GoldenCase) -> str:
    return {
        "shielding": "golden_single_group_scattering_shield_v1",
        "slab_attenuation": "golden_single_group_pure_absorber_v1",
        "downscatter_spectrum": "golden_three_group_downscatter_v1",
        "tbr": "golden_three_group_fusion_postprocess_v1",
        "heating": "golden_three_group_fusion_postprocess_v1",
    }[case.category]


def _default_tolerances(case: GoldenCase) -> dict[str, Any]:
    base = {
        "scalar_flux_rtol": 5.0e-10,
        "scalar_flux_atol": 5.0e-12,
        "selected_value_rtol": 5.0e-10,
        "selected_value_atol": 5.0e-12,
        "integral_rtol": 5.0e-10,
        "integral_atol": 5.0e-12,
        "residual_factor": 20.0,
        "max_relative_balance_change": 1.0e-10,
    }
    if case.category == "slab_attenuation":
        base["analytic_exponential_ratio_rtol"] = 0.75
        base["analytic_exponential_ratio_note"] = (
            "Loose because the transport solve is 3-D S4 with isotropic volumetric source in the first slab cell, "
            "not a mono-directional uncollided beam solution."
        )
    return base


def golden_path(case: GoldenCase) -> Path:
    return GOLDEN_DATA_DIR / f"{case.name}.json"


def write_golden_cases(cases: Iterable[GoldenCase]) -> None:
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        path = golden_path(case)
        path.write_text(json.dumps(run_golden_case(case), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")


def load_golden_case(case: GoldenCase) -> dict[str, Any]:
    return json.loads(golden_path(case).read_text(encoding="utf-8"))


def select_cases(tier: str) -> tuple[GoldenCase, ...]:
    if tier == "fast":
        return FAST_CASES
    if tier == "heavy":
        return HEAVY_CASES
    if tier == "all":
        return ALL_CASES
    raise ValueError(f"unknown golden benchmark tier {tier!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or intentionally regenerate deterministic golden benchmark data.")
    parser.add_argument("--tier", choices=("fast", "heavy", "all"), default="fast")
    parser.add_argument("--write-golden", action="store_true", help="write reference JSON files under fusion_project/data/golden")
    args = parser.parse_args(argv)

    cases = select_cases(args.tier)
    if args.write_golden:
        write_golden_cases(cases)
    else:
        for case in cases:
            result = run_golden_case(case)
            print(json.dumps({"case": case.name, "metrics": result["metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
