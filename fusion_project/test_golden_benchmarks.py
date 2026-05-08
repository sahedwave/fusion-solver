from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_benchmarks import FAST_CASES, HEAVY_CASES, GoldenCase, load_golden_case, run_golden_case


def _assert_close(name: str, got: float, expected: float, tolerances: dict[str, Any], key: str = "selected_value") -> None:
    rtol = float(tolerances[f"{key}_rtol"])
    atol = float(tolerances[f"{key}_atol"])
    assert np.isclose(got, expected, rtol=rtol, atol=atol), f"{name}: got {got:.17g}, expected {expected:.17g}"


def _assert_sequence_close(name: str, got: list[float], expected: list[float], tolerances: dict[str, Any], key: str = "selected_value") -> None:
    assert len(got) == len(expected), f"{name}: length drift got {len(got)}, expected {len(expected)}"
    for idx, (g, e) in enumerate(zip(got, expected)):
        _assert_close(f"{name}[{idx}]", float(g), float(e), tolerances, key=key)


def _assert_common_metrics(actual: dict[str, Any], golden: dict[str, Any]) -> None:
    assert actual["schema_version"] == golden["schema_version"]
    assert actual["case"] == golden["case"]
    assert actual["mesh"] == golden["mesh"]
    assert actual["solver_config"] == golden["solver_config"]
    assert actual["library"] == golden["library"]

    tol = golden["tolerances"]
    metrics = actual["metrics"]
    reference = golden["metrics"]

    assert metrics["convergence"]["converged"] is True
    assert metrics["convergence"]["outer_iterations"] <= max(reference["convergence"]["outer_iterations"] + 1, reference["convergence"]["outer_iterations"])
    assert metrics["convergence"]["gmres_iterations_total"] <= max(
        reference["convergence"]["gmres_iterations_total"] * 2,
        reference["convergence"]["gmres_iterations_total"] + 4,
    )
    assert metrics["convergence"]["residual_final"] <= max(
        1.0e-12,
        float(tol["residual_factor"]) * reference["convergence"]["residual_final"] + 1.0e-14,
    )

    assert metrics["positivity"]["relative_balance_change"] <= tol["max_relative_balance_change"]
    assert metrics["positivity"]["rebalance_applied"] is False
    assert metrics["scalar_flux"]["min"] >= -tol["scalar_flux_atol"]
    assert metrics["source"]["integral"] == pytest.approx(reference["source"]["integral"], rel=1.0e-13, abs=1.0e-13)

    for key in ("l1", "l2", "linf", "sum", "integral", "max", "mean"):
        _assert_close(f"scalar_flux.{key}", metrics["scalar_flux"][key], reference["scalar_flux"][key], tol, key="scalar_flux")


def _assert_category_metrics(actual: dict[str, Any], golden: dict[str, Any]) -> None:
    category = golden["case"]["category"]
    tol = golden["tolerances"]
    metrics = actual["metrics"]
    reference = golden["metrics"]

    if category == "shielding":
        _assert_sequence_close("detectors.x_plane_averages", metrics["detectors"]["x_plane_averages"], reference["detectors"]["x_plane_averages"], tol)
        _assert_sequence_close("detectors.selected_plane_averages", metrics["detectors"]["selected_plane_averages"], reference["detectors"]["selected_plane_averages"], tol)
        _assert_close("detectors.back_to_front_ratio", metrics["detectors"]["back_to_front_ratio"], reference["detectors"]["back_to_front_ratio"], tol)
        assert metrics["detectors"]["x_plane_averages"][-1] < metrics["detectors"]["x_plane_averages"][0]
        return

    if category == "slab_attenuation":
        _assert_sequence_close("attenuation.cell_center_flux", metrics["attenuation"]["cell_center_flux"], reference["attenuation"]["cell_center_flux"], tol)
        _assert_sequence_close("attenuation.adjacent_ratios", metrics["attenuation"]["adjacent_ratios"], reference["attenuation"]["adjacent_ratios"], tol)
        _assert_close("attenuation.mean_adjacent_ratio", metrics["attenuation"]["mean_adjacent_ratio"], reference["attenuation"]["mean_adjacent_ratio"], tol)
        assert metrics["attenuation"]["max_relative_error_to_exponential"] <= tol["analytic_exponential_ratio_rtol"]
        return

    if category == "downscatter_spectrum":
        _assert_sequence_close("spectrum.group_integrated_flux", metrics["spectrum"]["group_integrated_flux"], reference["spectrum"]["group_integrated_flux"], tol, key="integral")
        _assert_sequence_close("spectrum.normalized_group_spectrum", metrics["spectrum"]["normalized_group_spectrum"], reference["spectrum"]["normalized_group_spectrum"], tol)
        _assert_close("spectrum.thermal_to_fast_ratio", metrics["spectrum"]["thermal_to_fast_ratio"], reference["spectrum"]["thermal_to_fast_ratio"], tol)
        assert metrics["spectrum"]["normalized_group_spectrum"][0] > metrics["spectrum"]["normalized_group_spectrum"][-1]
        assert metrics["spectrum"]["thermal_to_fast_ratio"] > 0.0
        return

    if category == "tbr":
        for key in ("tbr_total", "tbr_li6", "tbr_li7", "li6_fraction", "breeding_map_integral_total"):
            _assert_close(f"tbr.{key}", metrics["tbr"][key], reference["tbr"][key], tol, key="integral")
        assert metrics["tbr"]["tbr_total"] == pytest.approx(metrics["tbr"]["tbr_li6"] + metrics["tbr"]["tbr_li7"], rel=1.0e-13, abs=1.0e-13)
        return

    if category == "heating":
        for key in ("integrated_power_watts", "integrated_power_mev_per_s", "peak_heating_w_per_cm3", "mean_heating_w_per_cm3", "peak_to_mean_heating", "peak_xmin_heat_flux_w_per_cm2"):
            _assert_close(f"heating.{key}", metrics["heating"][key], reference["heating"][key], tol, key="integral")
        assert metrics["heating"]["peak_heating_w_per_cm3"] >= metrics["heating"]["mean_heating_w_per_cm3"]
        return

    raise AssertionError(f"unhandled golden benchmark category {category!r}")


@pytest.mark.ci_drift
@pytest.mark.parametrize("case", FAST_CASES, ids=[case.name for case in FAST_CASES])
def test_fast_golden_ci_drift_benchmark(case: GoldenCase) -> None:
    golden = load_golden_case(case)
    actual = run_golden_case(case)
    _assert_common_metrics(actual, golden)
    _assert_category_metrics(actual, golden)


@pytest.mark.heavy
@pytest.mark.parametrize("case", HEAVY_CASES, ids=[case.name for case in HEAVY_CASES])
def test_heavy_golden_opt_in_benchmark(case: GoldenCase) -> None:
    golden = load_golden_case(case)
    actual = run_golden_case(case)
    _assert_common_metrics(actual, golden)
    _assert_category_metrics(actual, golden)


@pytest.mark.external_validation
def test_external_physics_benchmarks_placeholder() -> None:
    pytest.skip(
        "External/experimental benchmark validation cases are not yet bundled. "
        "When added, keep them separate from golden CI drift checks."
    )
