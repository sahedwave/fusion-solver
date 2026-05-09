from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTIGROUP_DOC = ROOT / "fusion_project" / "MULTIGROUP.md"
PYTEST_INI = ROOT / "pytest.ini"
TEST_LIB = ROOT / "fusion_project" / "test_multigroup_library.py"
CHECKLIST_PATH = ROOT / "fusion_project" / "dynamic_g_promotion_checklist.yaml"
HEAVY_ARTIFACT_META = ROOT / "fusion_project" / "data" / "benchmarks" / "heavy_ci_status.json"
LEGACY_BASELINE_PATH = ROOT / "fusion_project" / "legacy_surface_baseline.json"

_DYNAMIC_G_ROW_RE = re.compile(r"\|\s*Dynamic G\s*\|\s*\*\*(partial|complete)\*\*\s*\|", re.IGNORECASE)
_FIXED_GROUP_RE = re.compile(r"\b(group\s*0|G\s*=\s*3|3-group)\b", re.IGNORECASE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_checklist() -> dict:
    return json.loads(_read(CHECKLIST_PATH))


def _dynamic_g_status(doc_text: str) -> str:
    m = _DYNAMIC_G_ROW_RE.search(doc_text)
    if not m:
        raise SystemExit("Dynamic G status row missing in fusion_project/MULTIGROUP.md")
    return m.group(1).lower()


def _git_text(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def _baseline_ref() -> str:
    for remote_ref in ("origin/main", "origin/master", "main", "master"):
        base = _git_text("merge-base", "HEAD", remote_ref)
        if base and base.strip():
            return base.strip()
    return "HEAD~1"


def _doc_changed_partial_to_complete() -> bool:
    current_status = _dynamic_g_status(_read(MULTIGROUP_DOC))
    if current_status != "complete":
        return False
    base_doc = _git_text("show", f"{_baseline_ref()}:fusion_project/MULTIGROUP.md")
    if base_doc is None:
        return True
    return _dynamic_g_status(base_doc) == "partial"


def _require_fallback_caveats(doc_text: str, checklist: dict) -> None:
    missing = [item for item in checklist["required_caveat_phrases"] if item not in doc_text]
    if missing:
        raise SystemExit("Dynamic G promotion blocked: fallback-only caveats missing: " + ", ".join(missing))


def _require_marker_taxonomy() -> None:
    if "g_matrix:" not in _read(PYTEST_INI) or "@pytest.mark.g_matrix" not in _read(TEST_LIB):
        raise SystemExit("Dynamic G promotion blocked: g_matrix marker taxonomy/evidence missing")


def _require_no_untagged_fixed_group_assumptions(checklist: dict) -> None:
    tags = tuple(checklist.get("required_legacy_annotations", ["legacy", "fallback"]))
    offenders: list[str] = []
    for path in (ROOT / "fusion_project").rglob("*.py"):
        if path.name in {"check_dynamic_g_promotion_policy.py"}:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if _FIXED_GROUP_RE.search(line):
                window = " ".join(lines[max(0, lineno - 2): min(len(lines), lineno + 1)]).lower()
                if not any(tag in window for tag in tags):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    if offenders:
        raise SystemExit(f"Dynamic G promotion blocked: untagged fixed-group assumptions remain: {offenders[:10]}")


def _collect_nodeids() -> set[str]:
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("Dynamic G promotion blocked: pytest --collect-only failed")
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


def _require_collected_evidence(checklist: dict) -> None:
    nodeids = _collect_nodeids()
    expected = []
    for _, ids in checklist.get("required_nodeids", {}).items():
        expected.extend(ids)
    missing = [nid for nid in expected if nid not in nodeids]
    if missing:
        raise SystemExit(f"Dynamic G promotion blocked: missing/deselected evidence tests: {missing[:12]}")


def _require_heavy_artifact_freshness(checklist: dict) -> None:
    if not HEAVY_ARTIFACT_META.exists():
        raise SystemExit("Dynamic G promotion blocked: heavy CI artifact status file missing")
    meta = json.loads(_read(HEAVY_ARTIFACT_META))
    if not meta.get("success", False):
        raise SystemExit("Dynamic G promotion blocked: most recent heavy CI lane failed")
    ts = datetime.fromisoformat(meta["timestamp_utc"].replace("Z", "+00:00"))
    max_age_days = int(checklist.get("max_heavy_age_days", 7))
    if datetime.now(timezone.utc) - ts > timedelta(days=max_age_days):
        raise SystemExit(f"Dynamic G promotion blocked: heavy CI artifact older than {max_age_days} days")



def _require_metadata_driven_default_apis() -> None:
    source_text = _read(ROOT / "fusion_project" / "fusion" / "source.py")
    tbr_text = _read(ROOT / "fusion_project" / "fusion" / "tbr.py")
    required_source = "make_dt_source requires energy_bounds or source_group_mapping"
    if required_source not in source_text:
        raise SystemExit("Dynamic G promotion blocked: make_dt_source default path is not metadata-driven")
    if "def make_dt_source_legacy_group0(" not in source_text:
        raise SystemExit("Dynamic G promotion blocked: explicit legacy source compatibility API missing")
    if "def compute_tbr_components_legacy_g3(" not in tbr_text:
        raise SystemExit("Dynamic G promotion blocked: explicit legacy TBR compatibility API missing")


def _require_legacy_surface_nonincreasing() -> None:
    baseline = json.loads(_read(LEGACY_BASELINE_PATH))["legacy_helper_callsites"]
    total = 0
    offenders: list[str] = []
    for path in (ROOT / "fusion_project").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "/test_" in rel or rel.endswith("/fusion/source.py") or rel.endswith("/fusion/tbr.py"):
            continue
        text = path.read_text(encoding="utf-8")
        count = text.count("make_dt_source_legacy_group0(") + text.count("compute_tbr_components_legacy_g3(")
        if count:
            offenders.append(f"{rel}:{count}")
            total += count
    if offenders:
        raise SystemExit("Dynamic G promotion blocked: legacy helper use in non-legacy runtime modules: " + ", ".join(offenders[:10]))
    if total > baseline:
        raise SystemExit(f"Dynamic G promotion blocked: legacy helper surface increased ({total}>{baseline})")


def _require_no_legacy_helper_imports_in_runtime() -> None:
    offenders: list[str] = []
    for path in (ROOT / "fusion_project").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "/test_" in rel or rel.endswith("/fusion/source.py") or rel.endswith("/fusion/tbr.py"):
            continue
        txt = path.read_text(encoding="utf-8")
        if "make_dt_source_legacy_group0" in txt or "compute_tbr_components_legacy_g3" in txt:
            offenders.append(rel)
    if offenders:
        raise SystemExit("Dynamic G promotion blocked: legacy helper imports/usages in runtime modules: " + ", ".join(offenders[:10]))

def _run_g_matrix_tests() -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-m", "g_matrix", "-q"], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit("Dynamic G promotion blocked: not all g_matrix tests passed")


def main() -> int:
    checklist = _load_checklist()
    if not _doc_changed_partial_to_complete():
        print("Dynamic G policy check: no partial→complete promotion detected; skipping enforcement gate.")
        return 0
    doc_text = _read(MULTIGROUP_DOC)
    _require_fallback_caveats(doc_text, checklist)
    _require_marker_taxonomy()
    _require_no_untagged_fixed_group_assumptions(checklist)
    _require_collected_evidence(checklist)
    _require_metadata_driven_default_apis()
    _require_legacy_surface_nonincreasing()
    _require_no_legacy_helper_imports_in_runtime()
    _require_heavy_artifact_freshness(checklist)
    _run_g_matrix_tests()
    print("Dynamic G policy check: promotion evidence satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
