from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTIGROUP_DOC = ROOT / "fusion_project" / "MULTIGROUP.md"
PYTEST_INI = ROOT / "pytest.ini"
TEST_LIB = ROOT / "fusion_project" / "test_multigroup_library.py"

_DYNAMIC_G_ROW_RE = re.compile(r"\|\s*Dynamic G\s*\|\s*\*\*(partial|complete)\*\*\s*\|", re.IGNORECASE)
_FIXED_GROUP_RE = re.compile(r"\b(group\s*0|G\s*=\s*3|3-group)\b", re.IGNORECASE)
_ALLOWED_FIXED_GROUP_TAGS = ("legacy", "fallback", "fixed-group-legacy")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _dynamic_g_status(doc_text: str) -> str:
    m = _DYNAMIC_G_ROW_RE.search(doc_text)
    if not m:
        raise SystemExit("Dynamic G status row missing in fusion_project/MULTIGROUP.md")
    return m.group(1).lower()


def _doc_changed_partial_to_complete() -> bool:
    current = _read(MULTIGROUP_DOC)
    current_status = _dynamic_g_status(current)
    if current_status != "complete":
        return False

    base = subprocess.run(
        ["git", "show", "HEAD~1:fusion_project/MULTIGROUP.md"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if base.returncode != 0:
        # No comparable base (e.g. first commit in branch): enforce evidence if status is complete.
        return True

    prev_status = _dynamic_g_status(base.stdout)
    return prev_status == "partial"


def _require_fallback_caveats(doc_text: str) -> None:
    required = (
        "software-compatible fallbacks",
        "_uniform_fill",
        "make_dt_source(..., energy_bounds=None)",
        "compute_tbr_components()",
        "only partially dynamic",
    )
    missing = [item for item in required if item not in doc_text]
    if missing:
        raise SystemExit(
            "Dynamic G promotion blocked: fallback-only caveats missing from MULTIGROUP.md: "
            + ", ".join(missing)
        )


def _require_marker_taxonomy() -> None:
    if "g_matrix:" not in _read(PYTEST_INI):
        raise SystemExit("Dynamic G promotion blocked: pytest.ini missing g_matrix marker taxonomy")
    if "@pytest.mark.g_matrix" not in _read(TEST_LIB):
        raise SystemExit("Dynamic G promotion blocked: no @pytest.mark.g_matrix tests in test_multigroup_library.py")


def _require_no_untagged_fixed_group_assumptions() -> None:
    offenders: list[str] = []
    for path in (ROOT / "fusion_project").glob("test_*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _FIXED_GROUP_RE.search(line) and not any(tag in line.lower() for tag in _ALLOWED_FIXED_GROUP_TAGS):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    if offenders:
        raise SystemExit(
            "Dynamic G promotion blocked: untagged fixed-group assumptions remain. "
            f"First offenders: {offenders[:10]}"
        )


def _run_g_matrix_tests() -> None:
    cmd = [sys.executable, "-m", "pytest", "-m", "g_matrix", "-q"]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit("Dynamic G promotion blocked: not all g_matrix tests passed")


def main() -> int:
    doc_text = _read(MULTIGROUP_DOC)
    if not _doc_changed_partial_to_complete():
        print("Dynamic G policy check: no partial→complete promotion detected; skipping enforcement gate.")
        return 0

    _require_fallback_caveats(doc_text)
    _require_marker_taxonomy()
    _require_no_untagged_fixed_group_assumptions()
    _run_g_matrix_tests()
    print("Dynamic G policy check: promotion evidence satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
