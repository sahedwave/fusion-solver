from __future__ import annotations

import numpy as np

from sn_core import build_quadrature, integrate_J, integrate_moments


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))


def main() -> None:
    directions, weights = build_quadrature(8)
    nx, ny, nz, n_dir, G = 3, 4, 2, len(weights), 2

    psi_iso = np.full((nx, ny, nz, n_dir, G), 2.5, dtype=np.float64)
    phi_iso, J_iso = integrate_moments(psi_iso, directions, weights)
    _check("isotropic phi positive", bool(np.all(phi_iso > 0.0)))
    _check("isotropic current cancellation", float(np.max(np.abs(J_iso))) < 1.0e-12)

    psi_beam = np.zeros((nx, ny, nz, n_dir, G), dtype=np.float64)
    psi_beam[:, :, :, directions[:, 0] > 0.0, :] = 1.0
    _, J_beam = integrate_moments(psi_beam, directions, weights)
    _check("+X beam has positive Jx", float(np.mean(J_beam[..., 0])) > 0.0)
    _check("+X beam has zero Jy", float(np.max(np.abs(J_beam[..., 1]))) < 1.0e-12)
    _check("+X beam has zero Jz", float(np.max(np.abs(J_beam[..., 2]))) < 1.0e-12)

    rng = np.random.default_rng(123)
    psi = rng.random((nx, ny, nz, n_dir, G))
    phi, J = integrate_moments(psi, directions, weights)
    phi_ref = np.sum(psi * weights[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis], axis=3)
    _check("quadrature scalar-flux consistency", bool(np.allclose(phi, phi_ref, rtol=0.0, atol=1.0e-14)))

    J_only = integrate_J(psi, directions, weights)
    _check("phi shape", phi.shape == (nx, ny, nz, G), str(phi.shape))
    _check("J shape", J.shape == (nx, ny, nz, G, 3), str(J.shape))
    _check("integrate_J shape", J_only.shape == (nx, ny, nz, G, 3), str(J_only.shape))
    _check("integrate_J consistency", bool(np.allclose(J, J_only, rtol=0.0, atol=1.0e-14)))

    print("Moment integration validation complete.")


if __name__ == "__main__":
    main()
