# ARCHITECTURE.md — Deterministic Transport System Design

## System Overview

This project implements a deterministic neutron transport solver based on:

- Discrete Ordinates (Sn)
- Structured Cartesian and Unstructured meshes
- Diamond difference and step-characteristic schemes
- Multigroup neutron transport
- Krylov-based linear solvers
- DSA acceleration

Target reference systems:
- PARTISN
- Denovo
- ATTILA
- OpenSn
- Shift deterministic modules

---

## Core Subsystems

### 1. Mesh System
- Structured Cartesian mesh
- Unstructured tetra/hexa mesh support
- Face connectivity and sweep ordering

### 2. Transport Core
- Angular flux ψ(x, Ω, g)
- Scalar flux φ(x, g)
- Current J(x, g)

### 3. Operators
- TransportOperator (streaming + removal)
- ScatteringOperator
- SystemOperator (matrix-free assembly)

### 4. Acceleration
- Diffusion Synthetic Acceleration (DSA)
- GMRES + preconditioning

---

## Design Principles

- Strict separation of:
  mesh / materials / operators / solvers / acceleration
- No global hidden state in solvers
- Operators must be composable and testable
- State ownership must be explicit

---

## Numerical Philosophy

- Conservation must be explicit, not assumed
- Positivity must not violate balance
- Unstructured transport must remain consistent with discretization
- Any approximation must be documented

---

## Evolution Path

- Replace Python sweeps with compiled kernels
- Introduce DG/FEM transport for unstructured meshes
- Add MPI domain decomposition
- Move toward GPU-compatible sweep scheduling
