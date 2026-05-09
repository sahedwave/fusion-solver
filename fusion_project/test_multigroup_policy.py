from __future__ import annotations

from pathlib import Path
import re


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parent / rel).read_text(encoding="utf-8")


def test_dynamic_g_status_promotion_policy_contract() -> None:
    """Governance gate: Dynamic G cannot be promoted by prose-only edits."""
    multigroup_md = _read("MULTIGROUP.md")
    tbr_py = _read("fusion/tbr.py")
    materials_py = _read("fusion/materials.py")
    library_test = _read("test_multigroup_library.py")
    phase8_test = _read("test_phase8.py")

    row = re.search(r"\|\s*Dynamic G\s*\|\s*\*\*(partial|complete)\*\*\s*\|", multigroup_md)
    assert row is not None, "Dynamic G status row missing in MULTIGROUP.md"
    status = row.group(1)

    # Required evidence anchors that must remain present for any status.
    assert "compatibility-only" in multigroup_md
    assert "not external-physics validated for arbitrary `G`" in multigroup_md
    assert "require explicit `breeding_channels` metadata" in multigroup_md

    assert "Non-legacy paths never infer" in tbr_py
    assert "not external-physics validated for arbitrary" in tbr_py
    assert "compatibility-only" in materials_py
    assert "Production claims require explicit reaction-channel metadata" in materials_py

    # Required matrix-coverage anchors across solver/source/io/postprocess paths.
    for g in (1, 3, 10, 27, 70, 175):
        assert re.search(rf"\b{g}\b", library_test), f"missing matrix/io/source test anchor for G={g}"
        assert re.search(rf"\b{g}\b", phase8_test), f"missing postprocess test anchor for G={g}"

    # Promotion gate: complete is allowed only when evidence hooks remain explicit.
    if status == "complete":
        assert "test_source_and_io_matrix_parity" in library_test
        assert "test_postprocess_tbr_heating_source_normalization_matrix" in phase8_test
        assert "test_g_coverage_contract_complete" in library_test
        assert "@pytest.mark.heavy" in library_test and "@pytest.mark.heavy" in phase8_test
