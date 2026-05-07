"""
sn_solver.py — GMRES-DSA and Source-Iteration Solvers
======================================================

Uses the refactored operator classes from sn_operators.py.
SystemOperator.apply() now takes one argument (phi only).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sn_core import (
    Mesh, P1Material, BoundaryConditions,
)
from sn_operators import (
    TransportOperator, ScatteringOperator,
    SystemOperator, DSAPreconditioner,
)


# ================================================================
# DATA CLASSES
# ================================================================

@dataclass
class SolverConfig:
    tol:           float = 1e-8
    max_outer:     int   = 100
    gmres_restart: int   = 30
    inner_tol:     float = 1e-10
    verbose:       bool  = False


@dataclass
class SolverResult:
    phi:            np.ndarray
    J:              np.ndarray
    psi:            np.ndarray
    converged:      bool
    n_outer:        int
    n_gmres_total:  int = 0
    residuals:      list = field(default_factory=list)
    positivity_diagnostics: dict = field(default_factory=dict)


def _positivity_diagnostics(phi_raw: np.ndarray, mesh: Mesh) -> tuple[np.ndarray, dict]:
    negative = phi_raw < 0.0
    clipped = np.where(negative, -phi_raw, 0.0)
    vol = mesh.dx * mesh.dy * mesh.dz
    clipped_integral = float(clipped.sum()) * vol
    raw_abs_integral = float(np.abs(phi_raw).sum()) * vol
    diagnostics = {
        "negative_flux_before_floor": float(phi_raw.min()) if phi_raw.size else 0.0,
        "negative_cell_count": int(np.count_nonzero(negative)),
        "clipped_flux_integral": clipped_integral,
        "relative_balance_change": clipped_integral / max(raw_abs_integral, 1.0e-300),
        "rebalance_applied": False,
    }
    return np.maximum(phi_raw, 0.0), diagnostics


# ================================================================
# OPERATOR FACTORY
# ================================================================

def build_operators(
    mesh:       Mesh,
    mat:        P1Material,
    Q_ext:      np.ndarray,
    directions: np.ndarray,
    weights:    np.ndarray,
    bc:         BoundaryConditions,
    refl_map:   dict,
) -> tuple[TransportOperator, ScatteringOperator, SystemOperator, DSAPreconditioner]:
    """Construct all four operator objects for a given problem."""
    T = TransportOperator(mesh, mat, directions, weights, bc, refl_map)
    S = ScatteringOperator(mat, directions)
    A = SystemOperator(T, S, Q_ext)
    P = DSAPreconditioner(mesh, mat, bc)
    return T, S, A, P


def consistency_sweep(
    mesh:       Mesh,
    mat:        P1Material,
    Q_ext:      np.ndarray,
    directions: np.ndarray,
    weights:    np.ndarray,
    bc:         BoundaryConditions,
    refl_map:   dict,
) -> SolverResult:
    """Single sweep from zero — useful for operator checks."""
    T, S, A, _ = build_operators(mesh, mat, Q_ext, directions, weights, bc, refl_map)
    phi0 = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))
    phi_new, J_new = A.apply_phi_new(phi0)
    return SolverResult(
        phi=phi_new, J=J_new, psi=T.psi_ang.copy(),
        converged=True, n_outer=1,
    )


# ================================================================
# SOURCE ITERATION
# ================================================================

def solve_source_iteration(
    mesh:       Mesh,
    mat:        P1Material,
    Q_ext:      np.ndarray,
    directions: np.ndarray,
    weights:    np.ndarray,
    bc:         BoundaryConditions,
    refl_map:   dict,
    tol:        float = 1e-8,
    max_iter:   int   = 2000,
    verbose:    bool  = False,
) -> SolverResult:

    T, S, A, _ = build_operators(mesh, mat, Q_ext, directions, weights, bc, refl_map)

    phi = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))
    residuals = []

    for it in range(max_iter):
        phi_new, J_new = A.apply_phi_new(phi)
        norm_new = np.linalg.norm(phi_new)
        if norm_new < 1e-30:
            break
        res = np.linalg.norm(phi_new - phi) / norm_new
        residuals.append(res)
        if verbose:
            print(f"    SI iter {it+1:4d}  res = {res:.3e}")
        phi = phi_new
        if res < tol:
            break

    J_zero = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G, 3))
    T.reset_psi()
    psi_final, phi_final, J_final = T.sweep(Q_ext, phi, J_zero)
    phi_final, positivity = _positivity_diagnostics(phi_final, mesh)
    return SolverResult(
        phi=phi_final, J=J_final, psi=psi_final.copy(),
        converged=(residuals[-1] < tol if residuals else True),
        n_outer=it + 1,
        residuals=residuals,
        positivity_diagnostics=positivity,
    )


# ================================================================
# GMRES-DSA
# ================================================================

def solve_gmres_dsa(
    mesh:       Mesh,
    mat:        P1Material,
    Q_ext:      np.ndarray,
    directions: np.ndarray,
    weights:    np.ndarray,
    bc:         BoundaryConditions,
    refl_map:   dict,
    cfg:        SolverConfig = SolverConfig(),
) -> SolverResult:

    T, S, A, P = build_operators(mesh, mat, Q_ext, directions, weights, bc, refl_map)

    phi = np.zeros((mesh.nx, mesh.ny, mesh.nz, mat.G))
    G   = mat.G
    N   = mesh.nx * mesh.ny * mesh.nz * G
    residuals      = []
    n_gmres_total  = 0

    def matvec(x_flat):
        phi_x = x_flat.reshape(mesh.nx, mesh.ny, mesh.nz, G)
        return A.apply(phi_x).reshape(N)

    def precond(r_flat):
        r_phi = r_flat.reshape(mesh.nx, mesh.ny, mesh.nz, G)
        return P.apply(r_phi).reshape(N)

    from scipy.sparse.linalg import LinearOperator, gmres as sp_gmres

    LinOp  = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    PrecOp = LinearOperator((N, N), matvec=precond, dtype=np.float64)

    # Build RHS: b = φ - T(Q_ext)   [i.e. A·φ = b where b encodes Q_ext]
    # Actually we solve A·φ = 0 in the residual form; reformulate:
    # The fixed-point equation is φ = T·S·φ + T·Q_ext.
    # In the form A·φ = b:  (I - T·S)·φ = T·Q_ext
    # Compute b = T·Q_ext by one sweep with zero scattering source.
    phi_zero = np.zeros_like(phi)
    J_zero   = np.zeros((mesh.nx, mesh.ny, mesh.nz, G, 3))
    _, b_phi, _ = T.sweep(Q_ext, phi_zero, J_zero)
    T.reset_psi()                        # reset for proper GMRES start
    b = b_phi.reshape(N)

    iters_this_call = [0]

    def callback(rk):
        iters_this_call[0] += 1

    for outer in range(cfg.max_outer):
        x0 = phi.reshape(N)
        sol, info = sp_gmres(
            LinOp, b,
            x0=x0,
            M=PrecOp,
            restart=cfg.gmres_restart,
            rtol=cfg.inner_tol,
            maxiter=cfg.gmres_restart,
            callback=callback,
        )
        n_gmres_total += iters_this_call[0]
        iters_this_call[0] = 0

        phi_new = sol.reshape(mesh.nx, mesh.ny, mesh.nz, G)
        norm_new = np.linalg.norm(phi_new)
        if norm_new < 1e-30:
            phi = phi_new
            break
        res = np.linalg.norm(phi_new - phi) / norm_new
        residuals.append(res)
        phi = phi_new

        if cfg.verbose:
            print(f"    GMRES outer {outer+1:3d}  res={res:.3e}  "
                  f"gmres_its={n_gmres_total}")

        if info == 0 and res < cfg.tol:
            break

    J_zero = np.zeros((mesh.nx, mesh.ny, mesh.nz, G, 3))
    T.reset_psi()
    psi_final, phi_final, J_final = T.sweep(Q_ext, phi, J_zero)
    phi_final, positivity = _positivity_diagnostics(phi_final, mesh)
    return SolverResult(
        phi=phi_final, J=J_final, psi=psi_final.copy(),
        converged=(residuals[-1] < cfg.tol if residuals else True),
        n_outer=outer + 1,
        n_gmres_total=n_gmres_total,
        residuals=residuals,
        positivity_diagnostics=positivity,
    )
