from __future__ import annotations

import numpy as np

from sn_core import build_quadrature, build_reflection_map


FACE_SIGNS = {
    "xmin": np.array([-1.0, 1.0, 1.0]),
    "xmax": np.array([-1.0, 1.0, 1.0]),
    "ymin": np.array([1.0, -1.0, 1.0]),
    "ymax": np.array([1.0, -1.0, 1.0]),
    "zmin": np.array([1.0, 1.0, -1.0]),
    "zmax": np.array([1.0, 1.0, -1.0]),
}


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))


def _validate(sn: int) -> None:
    directions, weights = build_quadrature(sn)
    refl_map = build_reflection_map(directions)
    n_dir = directions.shape[0]

    _check(f"S{sn} face keys", set(refl_map) == set(FACE_SIGNS))

    for face, signs in FACE_SIGNS.items():
        mapping = refl_map[face]
        _check(f"S{sn} {face} mapping shape", mapping.shape == (n_dir,), str(mapping.shape))
        _check(f"S{sn} {face} integer dtype", np.issubdtype(mapping.dtype, np.integer), str(mapping.dtype))
        _check(f"S{sn} {face} valid indices", bool(np.all((mapping >= 0) & (mapping < n_dir))))

        reflected = directions[mapping]
        expected = directions * signs
        _check(
            f"S{sn} {face} reflection existence",
            bool(np.allclose(reflected, expected, rtol=0.0, atol=1.0e-12)),
        )
        _check(f"S{sn} {face} involution", bool(np.all(mapping[mapping] == np.arange(n_dir))))
        _check(
            f"S{sn} {face} weight symmetry",
            float(np.max(np.abs(weights - weights[mapping]))) < 1.0e-12,
        )
        _check(f"S{sn} {face} full coverage", set(mapping.tolist()) == set(range(n_dir)))


def main() -> None:
    for sn in (4, 8):
        _validate(sn)
    print("Reflection map validation complete.")


if __name__ == "__main__":
    main()
