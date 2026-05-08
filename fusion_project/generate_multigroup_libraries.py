from __future__ import annotations

from pathlib import Path

from sn_multigroup import make_synthetic_library, save_multigroup_library


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MULTIGROUP_DIR = PROJECT_DIR / "data" / "multigroup"


def main() -> None:
    out_dir = DEFAULT_MULTIGROUP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for G in (10, 27, 70):
        library = make_synthetic_library(G)
        save_multigroup_library(library, out_dir / f"synthetic_{G}g.json")
        save_multigroup_library(library, out_dir / f"synthetic_{G}g.npz")
    print(f"Wrote synthetic multigroup libraries to {out_dir}")


if __name__ == "__main__":
    main()
