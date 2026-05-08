# PERFORMANCE.md — HPC and Scaling Constraints

## Major bottlenecks

- Python-level sweep loops
- Full angular flux storage
- Dense scattering matrices
- GMRES matvec sweep cost

---

## Scaling laws

Let:
- N = mesh cells
- G = energy groups
- D = directions

Cost scales as:

O(N × G × D)

Memory scales as:

O(N × G × D)

---

## Required optimizations

- Group blocking
- Angle blocking
- Sweep kernel compilation (Numba/C++/CUDA)
- Sparse scattering representation
- Reduced angular storage (moment-based or sweep buffers)

---

## Forbidden patterns

- Full 4D angular flux storage for large problems
- Dense GxG scattering without blocking
- Pure Python nested sweep loops for production runs

---

## Target HPC direction

- MPI domain decomposition
- GPU sweep kernels
- Task-based sweep scheduling
- Cache-aware transport kernels
