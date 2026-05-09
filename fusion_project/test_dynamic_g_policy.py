from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTIGROUP_DOC = ROOT / "fusion_project" / "MULTIGROUP.md"

_DYNAMIC_G_COMPLETE_RE = re.compile(r"\|\s*Dynamic G\s*\|\s*\*\*complete\*\*", re.IGNORECASE)
_FIXED_GROUP_RE = re.compile(r"\b(group\s*0|G\s*=\s*3|3-group)\b", re.IGNORECASE)
_ALLOWED_FIXED_GROUP_TAGS = ("legacy", "fallback", "fixed-group-legacy")


def _dynamic_g_is_complete(doc: str) -> bool:
    return bool(_DYNAMIC_G_COMPLETE_RE.search(doc))


def test_dynamic_g_promotion_policy_guardrails() -> None:
    """Block Dynamic G promotion unless evidence and caveats are still explicit."""
    doc_text = MULTIGROUP_DOC.read_text(encoding="utf-8")
    if not _dynamic_g_is_complete(doc_text):
        # Policy only activates when status is promoted to complete.
        return

    required_caveats = (
        "software-compatible fallbacks",
        "_uniform_fill",
        "make_dt_source(..., energy_bounds=None)",
        "compute_tbr_components()",
        "only partially dynamic",
    )
    missing = [c for c in required_caveats if c not in doc_text]
    assert not missing, (
        "Dynamic G marked complete, but fallback-only caveats were removed: "
        + ", ".join(missing)
    )

    # Ensure CI taxonomy has an explicit marker for matrix-wide G evidence.
    pytest_ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "g_matrix:" in pytest_ini, (
        "Dynamic G completion requires explicit pytest marker taxonomy for "
        "group-matrix evidence (missing 'g_matrix' marker in pytest.ini)."
    )

    # Ensure at least one non-heavy matrix test is tagged for routine CI.
    lib_tests = (ROOT / "fusion_project" / "test_multigroup_library.py").read_text(encoding="utf-8")
    assert "@pytest.mark.g_matrix" in lib_tests, (
        "Dynamic G completion requires at least one @pytest.mark.g_matrix "
        "test in fusion_project/test_multigroup_library.py."
    )

    # Fixed-group assumptions are allowed only when explicitly tagged as legacy/fallback.
    offenders: list[str] = []
    for path in (ROOT / "fusion_project").glob("test_*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _FIXED_GROUP_RE.search(line) and not any(tag in line.lower() for tag in _ALLOWED_FIXED_GROUP_TAGS):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")

    assert not offenders, (
        "Dynamic G completion blocked: untagged fixed-group assumptions remain. "
        "Tag intentional legacy assumptions with one of "
        f"{_ALLOWED_FIXED_GROUP_TAGS}. Offenders: {offenders[:10]}"
    )
