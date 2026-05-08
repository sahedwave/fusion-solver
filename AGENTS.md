# AGENTS.md — Operational Rules

You are working on a deterministic neutron transport solver evolving toward production-grade HPC infrastructure.

## Non-negotiable rules

- Do not modify solver numerics without explaining conservation and stability impact.
- Do not introduce silent flux clipping or hidden normalization.
- Do not add dense allocations in angular flux or scattering without justification.
- Do not tighten module coupling (no circular dependencies).

## Required workflow before any code change

1. Inspect relevant modules first.
2. Identify numerical and architectural risks.
3. Explain impact on:
   - transport conservation
   - stability
   - scalability
4. Implement incrementally.
5. Ensure regression compatibility.

## Mandatory tests after changes

Run appropriate tests:

- pytest fusion_project/sn_validation.py
- pytest fusion_project/test_phase8.py
- pytest fusion_project/test_unstructured_mesh.py
- pytest fusion_project/test_multigroup_library.py

## Core principle

This is a production-oriented deterministic transport framework, not a research prototype.

Correctness > convenience.
Scalability > simplicity.
Explicit numerics > hidden heuristics.
