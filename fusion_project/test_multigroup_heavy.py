from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from multigroup_benchmarks import (
    BENCHMARK_DATA_DIR,
    FAST_CASES,
    HEAVY_CASES,
    MultigroupBenchmarkCase,
    run_multigroup_benchmark,
    run_cases,
    write_report,
)

BASELINE_PATH = BENCHMARK_DATA_DIR / "synthetic_multigroup_benchmark_baseline.json"
NUMERIC_RTOL = 2.0e-10
NUMERIC_ATOL = 2.0e-12
STRICT_PERF_ENV = "FUSION_BENCHMARK_STRICT_PERF"
GENERATE_REPORT_ENV = "FUSION_BENCHMARK_REPORT"


def _load_baselines() -> dict[str, dict]:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {entry["case"]["name"]: entry for entry in data["results"]}


def _assert_close(name: str, got: float, expected: float, *, rtol: float = NUMERIC_RTOL, atol: float = NUMERIC_ATOL) -> None:
    assert np.isclose(got, expected, rtol=rtol, atol=atol), f"{name}: got {got:.17g}, expected {expected:.17g}"


def _assert_common_regression_metrics(metrics: dict, baseline: dict) -> None:
    case = metrics["case"]
    expected_case = baseline["case"]
    assert case == expected_case
    assert metrics["schema"] == "fusion_multigroup_benchmark_v1"
    assert metrics["group_count"] == expected_case["groups"]
    assert metrics["mesh"] == baseline["mesh"]
    assert metrics["quadrature"] == baseline["quadrature"]
    assert metrics["library"]["synthetic"] is True
    assert metrics["library"]["physics_validation"] is False
    assert metrics["library"]["sigma_s0_nnz"] == baseline["library"]["sigma_s0_nnz"]
    assert metrics["library"]["sigma_s1_nnz"] == baseline["library"]["sigma_s1_nnz"]

    assert metrics["converged"] is True
    assert metrics["outer_iterations"] <= max(baseline["outer_iterations"] + 1, baseline["outer_iterations"])
    assert 0 < metrics["gmres_iterations_total"] <= max(2 * baseline["gmres_iterations_total"], baseline["gmres_iterations_total"] + 4)
    assert metrics["residual_final"] <= max(1.0e-5, 20.0 * baseline["residual_final"] + 1.0e-14)

    source = metrics["source_conservation"]
    _assert_close("external source integral", source["external_source_integral"], baseline["source_conservation"]["external_source_integral"])
    assert source["relative_source_strength_error"] <= 1.0e-13

    flux = metrics["scalar_flux"]
    expected_flux = baseline["scalar_flux"]
    for key in ("l1", "l2", "linf", "sum", "integral", "max", "mean"):
        _assert_close(f"scalar_flux.{key}", flux[key], expected_flux[key], rtol=5.0e-10, atol=5.0e-12)
    assert flux["min"] >= -1.0e-13
    assert flux["max"] > 0.0
    assert flux["l1"] > 0.0

    positivity = metrics["positivity"]
    assert positivity["relative_balance_change"] <= 1.0e-10
    assert positivity["rebalance_applied"] is False


def _assert_optional_performance(metrics: dict, baseline: dict) -> None:
    if os.environ.get(STRICT_PERF_ENV) != "1":
        pytest.skip(f"set {STRICT_PERF_ENV}=1 to enforce environment-sensitive runtime/memory thresholds")
    assert metrics["wall_time_seconds"] <= max(2.5 * baseline["wall_time_seconds"], baseline["wall_time_seconds"] + 5.0)
    if metrics["peak_memory_bytes"] is not None and baseline["peak_memory_bytes"] is not None:
        assert metrics["peak_memory_bytes"] <= max(
            int(3.0 * baseline["peak_memory_bytes"]),
            baseline["peak_memory_bytes"] + 20_000_000,
        )


@pytest.mark.parametrize("case", FAST_CASES, ids=[case.name for case in FAST_CASES])
def test_fast_multigroup_benchmark_regression(case: MultigroupBenchmarkCase) -> None:
    baselines = _load_baselines()
    metrics = run_multigroup_benchmark(case, collect_peak_memory=False)
    _assert_common_regression_metrics(metrics, baselines[case.name])


@pytest.mark.heavy
@pytest.mark.parametrize("case", HEAVY_CASES, ids=[case.name for case in HEAVY_CASES])
def test_heavy_multigroup_benchmark_regression(case: MultigroupBenchmarkCase) -> None:
    baselines = _load_baselines()
    metrics = run_multigroup_benchmark(case, collect_peak_memory=True)
    _assert_common_regression_metrics(metrics, baselines[case.name])
    if os.environ.get(STRICT_PERF_ENV) == "1":
        _assert_optional_performance(metrics, baselines[case.name])


@pytest.mark.benchmark
def test_multigroup_benchmark_report_generation(tmp_path: Path) -> None:
    output = Path(os.environ.get(GENERATE_REPORT_ENV, tmp_path / "multigroup_benchmark_report.json"))
    report = run_cases(FAST_CASES, collect_peak_memory=True)
    write_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema"] == "fusion_multigroup_benchmark_report_v1"
    assert [entry["case"]["name"] for entry in loaded["results"]] == [case.name for case in FAST_CASES]
    for entry in loaded["results"]:
        assert entry["wall_time_seconds"] > 0.0
        assert entry["peak_memory_bytes"] is not None
        assert entry["source_conservation"]["relative_source_strength_error"] <= 1.0e-13


def main() -> None:
    baselines = _load_baselines()
    for case in FAST_CASES + HEAVY_CASES:
        metrics = run_multigroup_benchmark(case, collect_peak_memory=True)
        _assert_common_regression_metrics(metrics, baselines[case.name])
    print("Multigroup benchmark regression complete.")


if __name__ == "__main__":
    main()
