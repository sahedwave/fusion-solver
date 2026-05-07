"""
sn_operators.py — PARTISN-Style Operator Classes
=================================================

All sweep and scattering logic lives here.  sn_sweep.py is no longer needed
and should be deleted — this module is its complete replacement.

Architecture
------------
  TransportOperator   T : Q_total → (ψ, φ, J)   (wraps the Sn sweep)
  ScatteringOperator  S : φ       → Q_scatter     (P0 + P1 scattering source)
  SystemOperator      A : φ       → φ - T·S·φ    (self-contained; derives J internally)
  DSAPreconditioner   P : r       → δφ            (diffusion-synthetic acceleration)

Operator purity contract (Phase 6 fix)
---------------------------------------
  SystemOperator.apply(phi) is a PURE function of φ:

      A(φ)  depends only on φ, never on ψ history.

  Enforcement mechanism
  ~~~~~~~~~~~~~~~~~~~~~
  Every call to apply() saves the current psi_ang, resets it to zero,
  runs the sweep, then restores the saved state.  This gives:

    • GMRES matvec calls: each Krylov vector sees an identical operator,
      with J=0 (vacuum P1 seed), independent of iteration order.

    • Outer SI / GMRES outer loop: psi_ang is restored after every
      operator evaluation so the reflective-boundary state that
      accumulates across outer iterations is never corrupted by
      the inner Krylov products.

  apply_phi_new() uses the same save/zero/restore discipline so the
  source-iteration loop is also stateless in the operator sense.

  The only callers that intentionally write psi_ang and expect it to
  persist are T.sweep() (called directly to build the RHS 'b' vector in
  solve_gmres_dsa) and the post-solve moment integration in the solver
  loop — both of which happen outside any operator evaluation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from sn_core import (
    Mesh, P1Material, BoundaryConditions,
    integrate_moments,
)
from mesh_builder import UnstructuredMesh
from mesh_geometry import _compute_sweep_order

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


# ================================================================
# INTERNAL SWEEP PRIMITIVES  (previously in sn_sweep.py)
# ================================================================

def _scattering_source(
    phi:        np.ndarray,    # (nx, ny, nz, G)
    J:          np.ndarray,    # (nx, ny, nz, G, 3)
    mat:        P1Material,
    directions: np.ndarray,    # (n_dir, 3)
) -> np.ndarray:               # (nx, ny, nz, n_dir, G)
    """
    P0 + P1 scattering source for all directions and groups.

        Q_s[i,j,k,m,g] = Σ_{g'} { Σ_s0[g',g] · φ[i,j,k,g']
                                  + 3 Σ_s1[g',g] · (Ω_m · J[i,j,k,g']) }
    """
    # P0 term: (...,G) × (G,G) → (...,G), broadcast over directions.
    q_p0 = np.tensordot(phi, mat.sigma_s0, axes=([-1], [0]))
    Q_s = np.expand_dims(q_p0, axis=-2)

    # P1 term
    omega_dot_J = np.einsum('mc,...gc->...mg', directions, J)
    q_p1 = 3.0 * np.tensordot(omega_dot_J, mat.sigma_s1, axes=([-1], [0]))

    return Q_s + q_p1


def _scattering_source_direction_group(
    phi:       np.ndarray,    # (nx, ny, nz, G)
    J:         np.ndarray,    # (nx, ny, nz, G, 3)
    mat:       P1Material,
    direction: np.ndarray,    # (3,)
    group:     int,
) -> np.ndarray:              # (nx, ny, nz)
    """P0+P1 scattering source for one outgoing direction and group."""
    q_p0 = np.tensordot(phi, mat.sigma_s0[:, group], axes=([-1], [0]))
    omega_dot_J = np.einsum("c,...gc->...g", direction, J)
    q_p1 = 3.0 * np.tensordot(omega_dot_J, mat.sigma_s1[:, group], axes=([-1], [0]))
    return q_p0 + q_p1


def _step_cell_python(
    psi_in_x: float, psi_in_y: float, psi_in_z: float,
    q_per_sr: float, sigma_t: float,
    amu: float, aeta: float, axi: float,
    inv_dx: float, inv_dy: float, inv_dz: float,
) -> Tuple[float, float, float, float]:
    """Linear diamond-differenced cell balance."""
    cx = amu * inv_dx
    cy = aeta * inv_dy
    cz = axi * inv_dz
    denom = sigma_t + 2.0 * (cx + cy + cz)
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, 0.0, 0.0, 0.0

    psi_cell = (
        2.0 * cx * psi_in_x
        + 2.0 * cy * psi_in_y
        + 2.0 * cz * psi_in_z
        + q_per_sr
    ) / denom
    if not np.isfinite(psi_cell):
        return 0.0, 0.0, 0.0, 0.0

    psi_out_x = 2.0 * psi_cell - psi_in_x
    psi_out_y = 2.0 * psi_cell - psi_in_y
    psi_out_z = 2.0 * psi_cell - psi_in_z

    if not np.isfinite(psi_cell + psi_out_x + psi_out_y + psi_out_z):
        return 0.0, 0.0, 0.0, 0.0

    return psi_cell, psi_out_x, psi_out_y, psi_out_z


if njit is not None:
    _step_cell_numba = njit(cache=True)(_step_cell_python)
else:
    _step_cell_numba = None


def _step_cell(
    psi_in_x: float, psi_in_y: float, psi_in_z: float,
    q_per_sr: float, sigma_t: float,
    amu: float, aeta: float, axi: float,
    inv_dx: float, inv_dy: float, inv_dz: float,
) -> Tuple[float, float, float, float]:
    """Diamond cell update; uses optional compiled scalar kernel when present."""
    if _step_cell_numba is not None:
        return _step_cell_numba(
            psi_in_x, psi_in_y, psi_in_z,
            q_per_sr, sigma_t,
            amu, aeta, axi,
            inv_dx, inv_dy, inv_dz,
        )
    return _step_cell_python(
        psi_in_x, psi_in_y, psi_in_z,
        q_per_sr, sigma_t,
        amu, aeta, axi,
        inv_dx, inv_dy, inv_dz,
    )


def _sweep_one_direction_group(
    psi_ang:       np.ndarray,          # (nx, ny, nz, n_dir, G) — modified in-place
    direction_idx: int,
    group:         int,
    amu: float, aeta: float, axi: float,
    mu:  float, eta:  float, xi:  float,
    mesh:       Mesh,
    sigma_t_g:  float,
    q_cell_g:   np.ndarray,             # (nx, ny, nz)
    bc:         BoundaryConditions,
    refl_map:   Dict[str, np.ndarray],
) -> None:
    """Sweep one discrete direction and one energy group in-place."""
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    inv_dx = 1.0 / mesh.dx
    inv_dy = 1.0 / mesh.dy
    inv_dz = 1.0 / mesh.dz
    m, g = direction_idx, group

    i_range = range(nx)          if mu  > 0 else range(nx - 1, -1, -1)
    j_range = range(ny)          if eta > 0 else range(ny - 1, -1, -1)
    k_range = range(nz)          if xi  > 0 else range(nz - 1, -1, -1)

    x_face = 'xmin' if mu  > 0 else 'xmax'
    y_face = 'ymin' if eta > 0 else 'ymax'
    z_face = 'zmin' if xi  > 0 else 'zmax'

    psi_x = np.zeros((ny, nz))
    psi_y = np.zeros((nx, nz))
    psi_z = np.zeros((nx, ny))

    if bc.is_reflective(x_face):
        m_ref = refl_map[x_face][m]
        psi_x[:, :] = (psi_ang[0,  :, :, m_ref, g] if mu  > 0
                       else psi_ang[-1, :, :, m_ref, g])

    if bc.is_reflective(y_face):
        m_ref = refl_map[y_face][m]
        psi_y[:, :] = (psi_ang[:, 0,  :, m_ref, g] if eta > 0
                       else psi_ang[:, -1, :, m_ref, g])

    if bc.is_reflective(z_face):
        m_ref = refl_map[z_face][m]
        psi_z[:, :] = (psi_ang[:, :, 0,  m_ref, g] if xi  > 0
                       else psi_ang[:, :, -1, m_ref, g])

    q_per_sr = q_cell_g * (1.0 / (4.0 * np.pi))

    for i in i_range:
        for j in j_range:
            for k in k_range:
                psi_c, pox, poy, poz = _step_cell(
                    psi_x[j, k], psi_y[i, k], psi_z[i, j],
                    q_per_sr[i, j, k], sigma_t_g,
                    amu, aeta, axi, inv_dx, inv_dy, inv_dz,
                )
                psi_ang[i, j, k, m, g] = psi_c
                psi_x[j, k] = pox
                psi_y[i, k] = poy
                psi_z[i, j] = poz


def _step_cell_unstructured(
    psi_inflow: np.ndarray,
    inflow_areas_cos: np.ndarray,
    outflow_area_cos_sum: float,
    q_per_sr: float,
    sigma_t: float,
    vol: float,
) -> tuple[float, float]:
    inflow_sum = float(np.dot(psi_inflow, inflow_areas_cos)) if psi_inflow.size else 0.0
    denom = sigma_t * vol + outflow_area_cos_sum
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, 0.0
    psi_cell = (q_per_sr * vol + inflow_sum) / denom
    if not np.isfinite(psi_cell):
        return 0.0, 0.0
    psi_cell = max(float(psi_cell), 0.0)
    return psi_cell, psi_cell


def _cartesian_proxy_from_unstructured(mesh: UnstructuredMesh):
    if mesh.cartesian_shape is None or mesh.cartesian_spacing is None:
        return None
    return Mesh(*mesh.cartesian_shape, *mesh.cartesian_spacing)


def _sweep_one_direction_group_unstructured(
    psi_ang:       np.ndarray,
    direction_idx: int,
    group:         int,
    direction:     np.ndarray,
    mesh:          UnstructuredMesh,
    sigma_t_g:     float,
    q_cell_g:      np.ndarray,
    bc:            BoundaryConditions,
    refl_map:      Dict[str, np.ndarray],
    sweep_order:   np.ndarray,
) -> None:
    # Cartesian-converted meshes preserve the legacy diamond-difference sweep exactly.
    cart = _cartesian_proxy_from_unstructured(mesh)
    if cart is not None:
        nx, ny, nz = mesh.cartesian_shape
        mu, eta, xi = direction
        _sweep_one_direction_group(
            psi_ang.reshape(nx, ny, nz, psi_ang.shape[-2], psi_ang.shape[-1]),
            direction_idx, group, abs(mu), abs(eta), abs(xi), mu, eta, xi,
            cart, sigma_t_g, q_cell_g.reshape(nx, ny, nz), bc, refl_map,
        )
        return

    m, g = direction_idx, group
    face_flux: dict[int, float] = {}
    q_per_sr = np.asarray(q_cell_g, dtype=np.float64) * (1.0 / (4.0 * np.pi))
    for c in sweep_order:
        c = int(c)
        psi_in = []
        inflow = []
        outflow_sum = 0.0
        for f in mesh.cell_to_faces[c]:
            f = int(f)
            cL, cR = map(int, mesh.face_to_cells[f])
            sign = 1.0 if c == cL else -1.0
            normal = sign * mesh.face_normal[f]
            dot = float(np.dot(direction, normal))
            area_cos = abs(dot) * float(mesh.face_area[f])
            if dot > 1.0e-14:
                outflow_sum += area_cos
            elif dot < -1.0e-14:
                other = cR if c == cL else cL
                psi_in.append(float(face_flux.get(f, 0.0)) if other != -1 else 0.0)
                inflow.append(area_cos)
        psi_c, psi_out = _step_cell_unstructured(
            np.asarray(psi_in, dtype=np.float64),
            np.asarray(inflow, dtype=np.float64),
            outflow_sum, float(q_per_sr[c]), sigma_t_g, float(mesh.cell_volume[c]),
        )
        psi_ang[c, m, g] = psi_c
        for f in mesh.cell_to_faces[c]:
            f = int(f)
            cL, cR = map(int, mesh.face_to_cells[f])
            sign = 1.0 if c == cL else -1.0
            if float(np.dot(direction, sign * mesh.face_normal[f])) > 1.0e-14:
                face_flux[f] = psi_out


# ================================================================
# TransportOperator  —  T : Q_total → (ψ, φ, J)
# ================================================================

class TransportOperator:
    """
    Encapsulates one complete multi-group Sn sweep.

    T.sweep(Q_total, phi_in, J_in) → (psi, phi_new, J_new)

    The operator holds persistent angular flux storage (psi_ang) so that
    reflective-boundary state carries across outer iterations.
    """

    def __init__(
        self,
        mesh:       Mesh,
        mat:        P1Material,
        directions: np.ndarray,
        weights:    np.ndarray,
        bc:         BoundaryConditions,
        refl_map:   Dict[str, np.ndarray],
    ) -> None:
        self.mesh       = mesh
        self.mat        = mat
        self.directions = directions
        self.weights    = weights
        self.bc         = bc
        self.refl_map   = refl_map
        n_dir = len(weights)
        if isinstance(mesh, UnstructuredMesh):
            self.psi_ang = np.zeros((mesh.N_cells, n_dir, mat.G), dtype=np.float64)
            self._sweep_orders = [_compute_sweep_order(mesh, directions[m]) for m in range(n_dir)]
        else:
            self.psi_ang = np.zeros(
                (mesh.nx, mesh.ny, mesh.nz, n_dir, mat.G), dtype=np.float64
            )
            self._sweep_orders = None

    def reset_psi(self) -> None:
        """Zero the angular flux (call between independent solves)."""
        self.psi_ang[:] = 0.0

    def sweep(
        self,
        Q_total: np.ndarray,    # (nx, ny, nz, G)  — external + scatter
        phi_in:  np.ndarray,    # (nx, ny, nz, G)
        J_in:    np.ndarray,    # (nx, ny, nz, G, 3)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One complete sweep:
            ψ_new  = L⁻¹ [ S(φ_in, J_in) + Q_total ]
            φ_new  = D ψ_new
            J_new  = D₁ ψ_new

        Returns (psi_ang, phi_new, J_new).
        """
        mesh       = self.mesh
        mat        = self.mat
        directions = self.directions
        weights    = self.weights
        bc         = self.bc
        refl_map   = self.refl_map

        has_reflective_bc = any(
            bc.is_reflective(face)
            for face in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        )

        for g in range(mat.G):
            sigma_t_g = mat.sigma_t[g]
            max_reflect_iters = 30 if has_reflective_bc else 1
            for _ in range(max_reflect_iters):
                psi_prev_g = self.psi_ang[..., :, g].copy() if has_reflective_bc else None
                for m in range(len(weights)):
                    mu, eta, xi = directions[m, 0], directions[m, 1], directions[m, 2]
                    q_scatter_g = _scattering_source_direction_group(
                        phi_in, J_in, mat, directions[m], g
                    )
                    q_cell_g = Q_total[..., g] + q_scatter_g
                    if isinstance(mesh, UnstructuredMesh):
                        _sweep_one_direction_group_unstructured(
                            psi_ang=self.psi_ang, direction_idx=m, group=g,
                            direction=directions[m], mesh=mesh, sigma_t_g=sigma_t_g,
                            q_cell_g=q_cell_g, bc=bc, refl_map=refl_map,
                            sweep_order=self._sweep_orders[m],
                        )
                    else:
                        _sweep_one_direction_group(
                            psi_ang=self.psi_ang,
                            direction_idx=m, group=g,
                            amu=abs(mu), aeta=abs(eta), axi=abs(xi),
                            mu=mu, eta=eta, xi=xi,
                            mesh=mesh, sigma_t_g=sigma_t_g, q_cell_g=q_cell_g,
                            bc=bc, refl_map=refl_map,
                        )
                if not has_reflective_bc:
                    break
                delta = np.linalg.norm(self.psi_ang[..., :, g] - psi_prev_g)
                scale = max(np.linalg.norm(self.psi_ang[..., :, g]), 1.0)
                if delta / scale < 1.0e-11:
                    break

        phi_new, J_new = integrate_moments(self.psi_ang, directions, weights)
        return self.psi_ang, phi_new, J_new


# ================================================================
# ScatteringOperator  —  S : φ → Q_scatter
# ================================================================

class ScatteringOperator:
    """
    Maps scalar flux (and derived current) to the scattering source.

        Q_s = S(φ, J)

    where J is the first angular moment of ψ.  In the PARTISN-style
    framework J is obtained from the transport operator; this class
    only computes the arithmetic.
    """

    def __init__(self, mat: P1Material, directions: np.ndarray) -> None:
        self.mat        = mat
        self.directions = directions

    def apply(
        self,
        phi: np.ndarray,    # (nx, ny, nz, G)
        J:   np.ndarray,    # (nx, ny, nz, G, 3)
    ) -> np.ndarray:        # (nx, ny, nz, n_dir, G)
        """Compute P0+P1 scattering source."""
        return _scattering_source(phi, J, self.mat, self.directions)


# ================================================================
# SystemOperator  —  A : φ → (I - T·S) φ
# ================================================================

class SystemOperator:
    """
    The fixed-point residual operator for source iteration.

        A(φ) = φ - T[S(φ) + Q_ext]

    Operator purity guarantee
    -------------------------
    apply(phi) is a PURE function of φ.  It must satisfy:

        same φ  →  identical A(φ)  (bitwise reproducible)

    Enforcement: every apply() call
      1. saves   psi_ang  (captures outer-iteration reflective state)
      2. zeros   psi_ang  (P1 seed J = 0 — no history dependence)
      3. runs    the sweep
      4. restores psi_ang (outer-iteration state is never corrupted)

    This makes each GMRES matvec see the same fixed linear operator
    regardless of which Krylov vector was evaluated previously.

    Usage
    -----
        A = SystemOperator(T, S, Q_ext)
        residual = A.apply(phi)   # pure: no external state management needed
    """

    def __init__(
        self,
        transport:  TransportOperator,
        scattering: ScatteringOperator,
        Q_ext:      np.ndarray,          # (nx, ny, nz, G)
    ) -> None:
        self.T      = transport
        self.S      = scattering
        self.Q_ext  = Q_ext

    # ------------------------------------------------------------------
    # PURE operator evaluation — save / zero / restore pattern
    # ------------------------------------------------------------------

    def _pure_sweep(
        self,
        phi: np.ndarray,      # (nx, ny, nz, G)
        Q_ext: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal helper: evaluate T·S(φ) with J = 0 seed (pure in φ).

        Save psi_ang → zero psi_ang → sweep → restore psi_ang.
        Returns (phi_new, J_new) where phi_new = T·(S(φ) + Q_ext).

        The save/restore means the outer-iteration reflective state in
        psi_ang is never overwritten by a Krylov sub-iteration product.
        """
        # ── 1. Save outer-iteration angular flux ──────────────────────
        psi_saved = self.T.psi_ang.copy()

        # ── 2. Zero: P1 seed J = 0  (no history) ─────────────────────
        self.T.psi_ang[:] = 0.0
        J_zero = np.zeros(
            (self.T.mesh.nx, self.T.mesh.ny, self.T.mesh.nz, self.T.mat.G, 3),
            dtype=np.float64,
        )

        # ── 3. One transport sweep ────────────────────────────────────
        # T.sweep owns scattering-source construction; passing pre-scattered
        # source here would double count scattering and break the fixed point.
        _, phi_new, J_new = self.T.sweep(Q_ext, phi, J_zero)

        # ── 4. Restore outer-iteration angular flux ───────────────────
        self.T.psi_ang[:] = psi_saved

        return phi_new, J_new

    def apply(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute A(φ) = φ - T·S(φ).

        Pure function of φ: identical input → identical output,
        regardless of call order or GMRES iteration history.
        """
        phi_new, _ = self._pure_sweep(phi, np.zeros_like(self.Q_ext))
        return phi - phi_new

    # ------------------------------------------------------------------
    # Convenience: one full application returning new φ (for SI loop)
    # ------------------------------------------------------------------
    def apply_phi_new(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (phi_new, J_new) after one T·S application.

        Uses the same pure save/zero/restore contract as apply().
        """
        return self._pure_sweep(phi, self.Q_ext)


# ================================================================
# DSAPreconditioner  —  P : r → δφ
# ================================================================

class DSAPreconditioner:
    """
    Diffusion Synthetic Acceleration preconditioner.

    Solves the multi-group diffusion equation:

        [ -∇·D∇ + Σ_a ] δφ = r_scalar

    where Σ_a uses the P0 absorption cross section (sigma_t - Σ_s0.sum(axis=1))
    *not* the removal cross section.  This is the standard DSA formulation.

    The matrix is assembled once and stored as a sparse LU factorisation.
    """

    def __init__(
        self,
        mesh: Mesh,
        mat:  P1Material,
        bc:   BoundaryConditions,
        verbose: bool = False,
    ) -> None:
        self.mesh    = mesh
        self.mat     = mat
        self.bc      = bc
        self.verbose = verbose
        self._A      = self._assemble(mesh, mat, bc)

    # ------------------------------------------------------------------
    def apply(self, residual_phi: np.ndarray) -> np.ndarray:
        """Solve the DSA system for δφ given scalar-flux residual."""
        mesh = self.mesh
        G    = self.mat.G
        if isinstance(mesh, UnstructuredMesh):
            shape = (mesh.N_cells, G)
            N = mesh.N_cells * G
        else:
            shape = (mesh.nx, mesh.ny, mesh.nz, G)
            N = mesh.nx * mesh.ny * mesh.nz * G
        rhs  = residual_phi.reshape(N)
        sol  = spsolve(self._A.tocsr(), rhs)
        return sol.reshape(shape)

    # ------------------------------------------------------------------
    def _assemble(
        self,
        mesh: Mesh,
        mat:  P1Material,
        bc:   BoundaryConditions,
    ):
        if isinstance(mesh, UnstructuredMesh):
            return self._assemble_unstructured(mesh, mat, bc)
        return self._assemble_cartesian(mesh, mat, bc)

    def _assemble_cartesian(self, mesh: Mesh, mat: P1Material, bc: BoundaryConditions):
        nx, ny, nz, G = mesh.nx, mesh.ny, mesh.nz, mat.G
        dx, dy, dz    = mesh.dx, mesh.dy, mesh.dz
        D             = mat.D
        sigma_a       = mat.sigma_a

        def idx(i, j, k, g):
            return ((i * ny + j) * nz + k) * G + g

        N = nx * ny * nz * G
        A = lil_matrix((N, N), dtype=np.float64)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for g in range(G):
                        row = idx(i, j, k, g)
                        Dg  = D[g]
                        diag = sigma_a[g]

                        cx = Dg / dx**2
                        if i > 0:
                            A[row, idx(i-1, j, k, g)] -= cx; diag += cx
                        elif not bc.xmin:
                            diag += 2.0 * Dg / dx
                        if i < nx - 1:
                            A[row, idx(i+1, j, k, g)] -= cx; diag += cx
                        elif not bc.xmax:
                            diag += 2.0 * Dg / dx

                        cy = Dg / dy**2
                        if j > 0:
                            A[row, idx(i, j-1, k, g)] -= cy; diag += cy
                        elif not bc.ymin:
                            diag += 2.0 * Dg / dy
                        if j < ny - 1:
                            A[row, idx(i, j+1, k, g)] -= cy; diag += cy
                        elif not bc.ymax:
                            diag += 2.0 * Dg / dy

                        cz = Dg / dz**2
                        if k > 0:
                            A[row, idx(i, j, k-1, g)] -= cz; diag += cz
                        elif not bc.zmin:
                            diag += 2.0 * Dg / dz
                        if k < nz - 1:
                            A[row, idx(i, j, k+1, g)] -= cz; diag += cz
                        elif not bc.zmax:
                            diag += 2.0 * Dg / dz

                        A[row, row] = diag
        return A

    def _assemble_unstructured(self, mesh: UnstructuredMesh, mat: P1Material, bc: BoundaryConditions):
        G = mat.G
        N = mesh.N_cells * G
        A = lil_matrix((N, N), dtype=np.float64)
        D = mat.D
        sigma_a = mat.sigma_a

        def idx(c, g):
            return int(c) * G + int(g)

        for f in range(mesh.N_faces):
            cL, cR = map(int, mesh.face_to_cells[f])
            if cR == -1:
                continue
            area = float(mesh.face_area[f])
            hf = (float(mesh.cell_volume[cL]) + float(mesh.cell_volume[cR])) / (2.0 * area)
            for g in range(G):
                Dg = float(D[g])
                coeff = Dg * area / hf
                rowL, rowR = idx(cL, g), idx(cR, g)
                A[rowL, rowL] += coeff
                A[rowR, rowR] += coeff
                A[rowL, rowR] -= coeff
                A[rowR, rowL] -= coeff

        boundary = set(int(f) for arr in mesh.boundary_faces.values() for f in arr)
        for c in range(mesh.N_cells):
            for g in range(G):
                row = idx(c, g)
                A[row, row] += sigma_a[g] * float(mesh.cell_volume[c])
                for f in mesh.cell_to_faces[c]:
                    f = int(f)
                    if f not in boundary:
                        continue
                    area = float(mesh.face_area[f])
                    hf = float(mesh.cell_volume[c]) / area
                    A[row, row] += 2.0 * float(D[g]) * area / hf
        return A



# ================================================================
# MODULE INTEGRITY ASSERTIONS
# ================================================================
# Checked once at import time.  If any class or method is accidentally
# removed during future refactors, the import itself will fail with a
# clear message rather than a cryptic AttributeError deep in the solver.

assert hasattr(TransportOperator, "sweep"),      \
    "sn_operators: TransportOperator.sweep() is missing — file is incomplete."
assert hasattr(TransportOperator, "reset_psi"),  \
    "sn_operators: TransportOperator.reset_psi() is missing — file is incomplete."
assert hasattr(ScatteringOperator, "apply"),     \
    "sn_operators: ScatteringOperator.apply() is missing — file is incomplete."
assert hasattr(SystemOperator, "apply"),         \
    "sn_operators: SystemOperator.apply() is missing — file is incomplete."
assert hasattr(SystemOperator, "apply_phi_new"), \
    "sn_operators: SystemOperator.apply_phi_new() is missing — file is incomplete."
assert hasattr(DSAPreconditioner, "apply"),      \
    "sn_operators: DSAPreconditioner.apply() is missing — file is incomplete."
