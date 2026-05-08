from __future__ import annotations

from pathlib import Path

from sn_core import build_quadrature
from sn_multigroup import format_memory_report, load_multigroup_library
from sn_operators import _step_cell_numba


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MULTIGROUP_DIR = PROJECT_DIR / "data" / "multigroup"


def main() -> None:
    data_dir = DEFAULT_MULTIGROUP_DIR
    print("Multigroup Solver Capability Report")
    print("=" * 42)
    print("\nLibraries:")
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.json")):
            lib = load_multigroup_library(path)
            print(f"  {path}: G={lib.G}, materials={list(lib.materials)}")
    else:
        print(f"  {data_dir} not found")

    print("\nQuadrature:")
    for sn in (4, 8):
        dirs, _ = build_quadrature(sn)
        print(f"  S{sn}: {len(dirs)} directions")

    print("\nImplementation:")
    print("  blocked scattering: enabled")
    print(f"  optional numba cell kernel: {'enabled' if _step_cell_numba is not None else 'unavailable'}")
    print("  max regular solver smoke G: 27")
    print("  max optional heavy smoke G: 175")
    print("  full production 175g large-mesh solve: not yet practical with Python sweep")

    print("\n" + format_memory_report(50, 50, 50))

    print("\nRemaining limitations:")
    print("  - full sweep is still Python-loop dominated")
    print("  - real NJOY/OpenMC data import is schema-ready but not physics-data integrated")
    print("  - positivity rebalance is diagnostic/result-level, not conservative nonlinear fixup")


if __name__ == "__main__":
    main()
