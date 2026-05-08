from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="run opt-in heavy multigroup regression tests",
    )
    parser.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="run opt-in benchmark report-generation tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_heavy = pytest.mark.skip(reason="need --run-heavy to run heavy benchmark/regression tests")
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmark to run benchmark report-generation tests")
    run_heavy = bool(config.getoption("--run-heavy"))
    run_benchmark = bool(config.getoption("--run-benchmark"))
    for item in items:
        keywords = item.keywords
        if "heavy" in keywords and not run_heavy:
            item.add_marker(skip_heavy)
        if "benchmark" in keywords and not run_benchmark:
            item.add_marker(skip_benchmark)
