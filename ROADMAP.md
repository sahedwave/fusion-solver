# ROADMAP.md — Solver Evolution Plan

## Phase 1 — Stabilization (Current Priority)

- Fix circular dependencies
- Isolate core data types (Mesh, Material, BC)
- Fix cwd-dependent tests
- Standardize boundary conditions
- Enforce conservation diagnostics

---

## Phase 2 — Numerical Correctness Upgrade

- Replace positivity clipping with conservative fixups
- Improve DSA consistency for unstructured meshes
- Add transport residual tracking
- Improve sweep ordering robustness

---

## Phase 3 — Scalability

- Reduce Python sweep overhead
- Introduce group blocking
- Introduce angle blocking
- Prepare compiled kernels (Numba/C++ backend)

---

## Phase 4 — Modern Transport Methods

- High-order quadrature support
- Eigenvalue (k-effective) solver
- Sparse scattering matrices
- Advanced acceleration (CMFD / NDA)

---

## Phase 5 — Unstructured FEM/DG Migration

- Replace face-balance approximation
- Introduce DG transport formulation
- Add numerical flux-based discretization
- Add mesh refinement convergence studies

---

## Phase 6 — HPC Expansion

- MPI domain decomposition
- GPU sweep kernels
- Distributed memory angular flux handling
