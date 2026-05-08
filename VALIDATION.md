# VALIDATION.md — Physics and Numerical Verification

## Core requirement

No solver change is valid without validation evidence.

---

## Required test categories

### Validation tiers and scope

- **Tier A — Synthetic software regression:** deterministic CI drift checks
  against repository golden artifacts.
- **Tier B — Manufactured/analytic checks:** compare to known analytic trends
  or closed-form references where available.
- **Tier C — External physics validation:** independent benchmark/experimental
  comparisons (separate data ownership, review, and tolerances).

Tier A/B are necessary for software quality but do **not** replace Tier C for
claiming external physics validation.

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

### External benchmark slots (placeholder, no fake data)
- `external_validation/criticality/` (planned)
- `external_validation/shielding/` (planned)
- `external_validation/fusion_blanket/` (planned)

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
