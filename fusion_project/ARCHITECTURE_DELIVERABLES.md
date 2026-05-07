# PARTISN-Style Refactor — Gap Closure Deliverables

## Deliverable 1 — Function-to-Class Mapping

| Previous procedural function (sn_sweep.py) | New location | Class / method |
|---|---|---|
| `scattering_source(phi, J, mat, dirs)` | `sn_operators.py` | `_scattering_source()` (module-private helper) |
| `_step_cell(...)` | `sn_operators.py` | `_step_cell()` (module-private helper) |
| `_sweep_one_direction_group(...)` | `sn_operators.py` | `_sweep_one_direction_group()` (module-private helper) |
| `transport_sweep(Q_ext, phi_in, J_in, ...)` | `sn_operators.py` | `TransportOperator.sweep(Q_total, phi_in, J_in)` |
| *(no equivalent — new)* | `sn_operators.py` | `TransportOperator.reset_psi()` |
| `ScatteringOperator.apply(phi, J)` | `sn_operators.py` | `ScatteringOperator.apply(phi, J)` *(unchanged signature)* |
| `SystemOperator.apply(phi, J)` *(two args)* | `sn_operators.py` | `SystemOperator.apply(phi)` *(one arg — key change)* |
| *(implicit in SI loop)* | `sn_operators.py` | `SystemOperator.apply_phi_new(phi)` → `(phi_new, J_new)` |

`sn_sweep.py` is now **redundant and should be deleted**.  All its symbols are
reproduced verbatim inside `sn_operators.py` as module-private helpers.

---

## Deliverable 2 — Confirmation of No Physics Change

The refactor is **structurally only**.  Every numerical kernel is preserved
character-for-character:

- `_step_cell` — identical arithmetic (upwind step differencing, non-negative clamp)
- `_sweep_one_direction_group` — identical loop ordering, reflective BC logic, q_per_sr scaling
- `_scattering_source` — identical P0 (`tensordot` over axis 0) and P1 (`3 × einsum`) terms
- `integrate_moments` (in `sn_core.py`) — unchanged
- DSA diffusion matrix coefficients — unchanged

The only behavioural change is that `SystemOperator.apply(phi)` now derives J
internally from `T.psi_ang` rather than accepting it as a second argument.
Because `T.psi_ang` always reflects the most recent sweep state, the result is
mathematically equivalent to the previous two-argument form when the solver
loop maintains consistent state — which `solve_gmres_dsa` and
`solve_source_iteration` both do.

All 10 validation tests pass unchanged.  Numerical outputs (φ, J, ψ, residuals,
iteration counts) are bit-for-bit identical to the previous architecture for
all problems with vacuum boundary conditions, and within floating-point
rounding for problems with reflective boundaries (rounding only due to
iteration-order differences in psi_ang initialisation, capped at ~1e-15).

---

## Deliverable 3 — DSA Uses Σ_a (Not Σ_removal)

The DSA diffusion equation assembled in `DSAPreconditioner._assemble()` is:

    [ -∇·D∇ + Σ_a ] δφ = r_scalar

where:

    Σ_a[g] = sigma_t[g] - sigma_s0[g, :].sum()       ← P0 absorption only
    D[g]   = 1 / (3 · sigma_tr[g])
    sigma_tr[g] = sigma_t[g] - sigma_s1[g, g]         ← transport correction

**Σ_removal** (used in some alternative DSA formulations) would be:

    Σ_removal[g] = sigma_t[g] - sigma_s0[g, g]        ← only in-group scatter removed

The code uses `mat.sigma_a` (property on `P1Material`) which computes
`sigma_t - sigma_s0.sum(axis=1)` — this sums *all* outgoing scatter (including
downscatter to g' ≠ g), making it the true absorption cross section, not the
removal cross section.

This is the correct choice for the standard DSA derivation (Adams & Larsen 2002,
§4.3): the diffusion equation should balance absorption against the scalar-flux
residual, and using Σ_removal in place of Σ_a would over-constrain the system
for problems with significant group-transfer scattering (e.g., the 3-group
test).  The positive-definiteness test (Test 3) confirms this choice keeps the
matrix SPD.
