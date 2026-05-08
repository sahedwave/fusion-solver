# VALIDATION.md — Physics and Numerical Verification

## Core requirement

No solver change is valid without validation evidence.

---

## Required test categories

### Conservation
- Global particle balance
- Cell-wise leakage consistency

### Positivity
- Detect negative flux regions
- Ensure fixups do not break conservation

### Benchmark suites
- 1D slab transport
- Streaming tests
- Void region tests
- Thick diffusion limit
- Ray effect sensitivity
- Shielding benchmarks

---

## Regression requirements

- Any solver modification must be compared against:
  - previous solution
  - reference analytical or benchmark solution

---

## Unstructured validation

- Mesh refinement convergence
- Conservation under irregular geometry
- Sweep ordering correctness

---

## Acceptance rule

A solver change is INVALID if:
- conservation is not tested
- numerical stability is not discussed
- benchmark comparison is missing
