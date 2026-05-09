from __future__ import annotations

"""Fast governance checks for Dynamic-G CI evidence taxonomy.

Policy scope: test taxonomy only. No solver numerics are modified.
"""

from multigroup_benchmarks import FAST_CASES, HEAVY_CASES
from test_multigroup_library import G_COVERAGE_CONTRACT, REQUIRED_G_COVERAGE, REQUIRED_PATH_KEYS


def test_required_g_matrix_contract_shape() -> None:
    """Ensure CI-auditable G/path contract remains complete."""
    assert set(REQUIRED_G_COVERAGE) == {1, 3, 10, 27, 70, 175}
    for g in REQUIRED_G_COVERAGE:
        mapping = G_COVERAGE_CONTRACT[g]
        assert set(mapping) == set(REQUIRED_PATH_KEYS)
        for path in REQUIRED_PATH_KEYS:
            assert mapping[path].strip()


def test_deterministic_regression_cases_include_required_groups() -> None:
    """Ensure benchmark regression taxonomy retains required deterministic G cases."""
    fast_groups = {case.groups for case in FAST_CASES}
    heavy_groups = {case.groups for case in HEAVY_CASES}

    # Fast deterministic CI lane covers medium groups.
    assert {10, 27}.issubset(fast_groups)
    # Heavy deterministic lane covers large-group regressions.
    assert {70, 175}.issubset(heavy_groups)


def test_g_coverage_contract_references_expected_test_ids() -> None:
    """Ensure contract references are explicit, not prose-only placeholders."""
    for g in REQUIRED_G_COVERAGE:
        path_map = G_COVERAGE_CONTRACT[g]
        # Core evidence must name concrete tests.
        assert "test_" in path_map["solve"]
        assert "test_source_and_io_matrix_parity" in path_map["source"]
        assert "test_source_and_io_matrix_parity" in path_map["io_roundtrip"]
        assert "test_postprocess_tbr_heating_source_normalization_matrix" in path_map["postprocess"]
