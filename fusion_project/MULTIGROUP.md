# Multigroup Library Support

This project now has a small production-oriented multigroup data layer in
`sn_multigroup.py`.  It is intentionally separate from the sweep and solver
modules so existing transport APIs remain stable.

## Data Model

`MaterialXS` requires:

- `sigma_t`: shape `(G,)`
- `sigma_s0`: shape `(G, G)` dense array or SciPy sparse matrix
- `sigma_s1`: shape `(G, G)` dense array or SciPy sparse matrix

The public `sigma_s0` and `sigma_s1` attributes remain dense NumPy arrays for
legacy API compatibility.  Each material also exposes `sigma_s0_sparse` and
`sigma_s1_sparse` as canonical CSC matrices.  Sweep-time scattering evaluates
outgoing group columns from these sparse matrices, so sparse libraries touch
only nonzero group-coupling entries and do not require constructing a full
`(nx, ny, nz, n_dir, G)` scattering source.

Optional data:

- `reactions`: dictionary of reaction-name to `(G,)` array
- `heating`: shape `(G,)`
- `chi`: fission emission spectrum, shape `(G,)`
- `nu_sigma_f`: neutron production cross section, shape `(G,)`
- `metadata`: dictionary

Fission data are first-class schema fields rather than entries hidden in
`reactions` or `metadata`.  When present, `chi` and `nu_sigma_f` must be finite
and nonnegative.  `chi` is required to be supplied already normalized, with
`np.isclose(chi.sum(), 1.0)` true; the loader rejects unnormalized spectra
rather than silently renormalizing them.  This preserves explicit source
strength accounting for future fixed-source fission and eigenvalue workflows.

Non-fission materials should omit `chi` and `nu_sigma_f` or set them to `null`
in JSON.  A non-fission material may provide an all-zero `nu_sigma_f` when a
code path expects the field to exist, but `chi` should remain omitted unless a
valid normalized fission spectrum is physically meaningful for that material.

`MultigroupLibrary` requires:

- `energy_bounds`: shape `(G + 1,)`, strictly monotonic
- `materials`: dictionary of material key to `MaterialXS`

## File Formats

Use:

- JSON (`.json`) for readable small libraries
- NPZ (`.npz`) for dense numerical arrays with only NumPy required
- HDF5 (`.h5`, `.hdf5`) for portable hierarchical libraries when `h5py` is
  installed; `h5py` is included in `requirements.txt` for standard project
  environments

All formats round-trip the first-class `chi` and `nu_sigma_f` arrays when they
are present.  Scattering is serialized in both dense form (`sigma_s0`,
`sigma_s1`) and sparse COO triplet form (`sigma_s0_sparse`, `sigma_s1_sparse`)
for JSON, NPZ, and HDF5.  The sparse schema is `format=coo_triplet_v1`,
`shape=[G, G]`, and parallel `row`, `col`, `data` arrays using the convention
`[source_group, outgoing_group]`.  Loaders prefer the sparse triplets when
present and fall back to the dense arrays for older files.  Older JSON and NPZ
libraries that omit these fields remain valid and load with
`MaterialXS.chi is None` and `MaterialXS.nu_sigma_f is None`.
`sn_multigroup.py` still does not import `h5py` at module import time, so
source checkouts or constrained environments that have not installed the full
requirements can continue to import the multigroup module.  Attempting to read
or write `.h5`/`.hdf5` without `h5py` raises a clear installation error.  For
standard environments, install the project requirements:

```bash
pip install -r requirements.txt
```

### Constrained processed-XS importer (schema integration only)

`sn_multigroup.import_processed_fusion_xs_json(path)` loads a narrow JSON
schema intended for processed multigroup payload integration:

- required: `energy_bounds`, `materials`
- per material required: `name`, `sigma_t`, `sigma_s0`, `sigma_s1`
- per material optional: `reactions`, `heating`, `chi`, `nu_sigma_f`, `metadata`

Validation remains strict through `MaterialXS`/`MultigroupLibrary` checks
(shape, monotonic bounds, finite/nonnegative XS, normalized `chi` when
provided). If top-level `metadata.provenance` is present, library metadata is
tagged `processed_external_format=true`; otherwise a warning is emitted and
the tag is set false.

This importer is **schema wiring only**. It does **not** imply FENDL/NJOY/OpenMC
benchmark validation by itself.

The HDF5 schema is:

```text
/energy_bounds
/metadata_json
/material_keys_json
/materials/<material-key>/name
/materials/<material-key>/sigma_t
/materials/<material-key>/sigma_s0
/materials/<material-key>/sigma_s1
/materials/<material-key>/scattering_sparse/sigma_s0/{shape,row,col,data}
/materials/<material-key>/scattering_sparse/sigma_s1/{shape,row,col,data}
/materials/<material-key>/chi                 # optional
/materials/<material-key>/nu_sigma_f          # optional
/materials/<material-key>/heating             # optional
/materials/<material-key>/metadata_json
/materials/<material-key>/reaction_keys_json
/materials/<material-key>/reactions/<reaction-name>
```

```python
from sn_multigroup import load_multigroup_library

library = load_multigroup_library("data/multigroup/synthetic_27g.json")
material = library.materials["synthetic_27g"].to_p1_material()
```

## Spectrum Sources

Arbitrary group sources are built with:

```python
from sn_core import dt_source_spectrum, make_spectrum_source

spectrum = dt_source_spectrum(library.energy_bounds)
Q_ext = make_spectrum_source(mesh, spectrum, strength=1.0, geometry="point")
```

The source is normalized so:

```text
sum(Q_ext) * dx * dy * dz == strength
```

## Known group-semantics assumptions

The solver core and multigroup data layer are dynamic in group count: `P1Material`
and `MaterialXS` derive `G` from array lengths, validate `(G,)` and `(G, G)`
shapes, and the sparse sweep path iterates over `mat.G`.  That does **not** mean
all fusion convenience helpers are physically group-agnostic.  The following
assumptions are intentionally documented so production work does not mistake
software-compatible fallbacks for processed multigroup physics data.

### Acceptable synthetic/software-regression assumptions

These are acceptable for current regression tests and examples, but are not a
claim of production physics validity:

- `fusion/materials.py` provides reference values for a **3-group**
  fast/epithermal/thermal ordering: `g=0 fast`, `g=1 epi`, `g=2 thermal`.
- For `G != 3`, fusion material factories use `_uniform_fill`, a synthetic
  `1/(1+g)` decay from the fast-group value.  This is a compatibility fallback
  that preserves array shape and deterministic behavior; it is not a real group
  collapse from FENDL/NJOY/OpenMC data.
- `test_phase8.py` is a 3-group fusion post-processing validation suite.  Its
  point-source and D-T-source checks intentionally verify the legacy convention
  that group 0 is the fast/D-T source group and groups `> 0` receive no direct
  D-T source when no `energy_bounds` are provided.
- Synthetic multigroup tests in `test_multigroup_library.py` validate arbitrary
  source spectra and energy-bound D-T mapping, but those libraries remain
  software-validation data rather than nuclear design data.

### Production blockers before claiming truly dynamic fusion group semantics

The following must be replaced by explicit metadata or reaction channels before
claiming production-grade, group-agnostic fusion post-processing:

- `make_dt_source(..., energy_bounds=None)` places 14.1 MeV neutrons in group 0
  by convention.  Production workflows should provide energy bounds or an
  explicit source-group mapping so the D-T source group is data-driven.
- `compute_tbr_components()` legacy split (Li-7 in group 0, Li-6 in groups 1--2)
  is compatibility-only and opt-in (`legacy_group_semantics=True`, `G==3`).
  It is not external-physics validated for arbitrary `G`, and production claims
  require explicit `breeding_channels` metadata.
- Li-bearing material factories encode enrichment effects through the current
  3-group fast/epi/thermal arrays or the synthetic `_uniform_fill` fallback.
  Production TBR splitting should instead consume explicit `Li6`/`Li7`
  reaction-channel vectors with shape `(G,)`.
- Heating, damage, and reaction post-processing are shape-generic once supplied
  with `(G,)` material vectors, but physical validity depends on those vectors
  being generated for the same energy structure as the transport solve.

In short: **solver-core dynamic `G` support is present; production fusion
group semantics are only partially dynamic until source mappings and reaction
channels become metadata-driven.**

## Synthetic Libraries

Generated examples live under:

```text
data/multigroup/
```

Current generated sizes:

- 10 groups
- 27 groups
- 70 groups

They are synthetic, downscatter-dominant, and meant for software validation,
not nuclear design.

## Multigroup roadmap status

| Area | Status | Evidence | Remaining work |
|---|---|---|---|
| Dynamic G | **partial** | `sn_multigroup.py` data model + `make_synthetic_library`/`make_sparse_synthetic_library`; multigroup tests in `test_multigroup_library.py` and `test_multigroup_heavy.py`. | Keep parity for larger `G` across all physics/postprocessing paths; reduce Python sweep overhead for large `G` (ROADMAP Phase 3, PERFORMANCE bottlenecks). |
| XS schema | **partial** | `MaterialXS` + `MultigroupLibrary` validation in `sn_multigroup.py`; schema tests in `test_multigroup_library.py`. | Add formally versioned schema docs and migration tooling for future fields; tighten external-import contract evolution. |
| JSON/NPZ/HDF5 loaders | **partial** | `save_multigroup_library`/`load_multigroup_library` and HDF5 helpers in `sn_multigroup.py`; round-trip tests (`json/npz/h5`) in `test_multigroup_library.py`. | Add broader compatibility tests against archived historical files and stricter schema-version gating. |
| Group metadata | **partial** | `group_names`, `lethargy_widths`, `source_group_mapping` in `MultigroupLibrary`; metadata tests in `test_multigroup_library.py`. | Expand metadata conventions and controlled vocab for downstream tooling; add compatibility policy docs. |
| Sparse scattering | **partial** | Sparse CSC support in `MaterialXS`; sweep-time directional/group scattering in `sn_operators.py`; guardrail tests in `test_sparse_scattering.py`. | Backend kernel work remains future (ROADMAP Phase 4 note); optimize large-scale runtime beyond Python loops. |
| Arbitrary source spectra | **partial** | `make_spectrum_source`, `dt_source_spectrum`, named-source helper in `sn_multigroup.py`; source tests in `test_multigroup_library.py`. | Add validated external source libraries/workflows; integrate richer source provenance and uncertainty metadata. |
| 10/27/70/175 group tests | **partial** | `test_multigroup_heavy.py` fast/heavy tiers and `multigroup_benchmarks.py` report harness. | Keep heavy tier opt-in; extend coverage for larger meshes and stricter scalability gates after backend acceleration. |
| Performance pass | **not started** (production-scale) | Current docs explicitly list Python sweep and memory bottlenecks in `PERFORMANCE.md`; benchmark artifacts are regression sentinels only. | Implement Phase 3+ roadmap items: group/angle blocking, compiled kernels, reduced storage, MPI/GPU pathways. |
| Real physics-library integration | **partial** (schema only) | `import_processed_fusion_xs_json` importer in `sn_multigroup.py`; importer section above; mock `example_real_schema.json`. | Integrate real processed libraries with audited provenance and unit consistency checks; no placeholder physics accepted. |
| Production validation | **not started** (external benchmark level) | `VALIDATION.md` Tier A/B vs Tier C distinction; current golden/multigroup suites are software regression + manufactured checks. | Add independent benchmark/experimental datasets, acceptance tolerances, and review workflow for Tier C external validation. |

**Important:** current multigroup CI/golden suites are software-regression and
manufactured-check infrastructure. They are not external benchmark validation
or licensing-grade neutronics qualification.

## Memory Scaling Warning

The legacy all-direction scattering API still returns a dense array with shape:

```text
(nx, ny, nz, n_dir, G)
```

For a `50x50x50`, S8, `G=175` problem, angular flux plus dense scattering
source alone is roughly 26 GiB before solver overhead.  The sweep path now uses
per-direction/per-group sparse scattering columns and a single cell-shaped
source buffer, avoiding that full dense scattering-source allocation.

## Multigroup Benchmark / Regression Tiers

The large-group checks are synthetic software benchmarks, not physics
validation benchmarks.  They use `make_sparse_synthetic_library()` with a
deterministic sparse upscatter/downscatter pattern, an isotropic normalized
D-T spectrum source, vacuum Cartesian boundaries, and the existing GMRES-DSA
solver.  No transport numerics are changed by the benchmark harness.

### Fast CI tier

The default fast tier runs 10-group and 27-group strict numerical regressions
on a `2x2x2` mesh with S4 quadrature.  It checks more than finite/nonnegative
flux: convergence status, outer iterations, total GMRES iterations, scalar
flux norms/summaries, source-strength conservation, positivity diagnostics,
mesh/quadrature metadata, and sparse scattering nonzero counts are compared
against the baseline artifact.

```bash
python -m pytest fusion_project/test_multigroup_heavy.py -q
```

### Optional heavy tier

The heavy tier is opt-in so routine CI is not destabilized by Python sweep
performance.  It runs 70 groups on a `3x3x3` mesh and 175 groups on a `2x2x2`
mesh, both with S4 quadrature.  These cases record wall time and peak RSS
memory and enforce the same numerical/convergence/source checks as the fast
tier.  Environment-sensitive performance thresholds are not enforced unless
`FUSION_BENCHMARK_STRICT_PERF=1` is set.

```bash
python -m pytest fusion_project/test_multigroup_heavy.py -q -m heavy --run-heavy
```

Strict local performance comparison against the recorded baseline can be
requested with:

```bash
FUSION_BENCHMARK_STRICT_PERF=1 python -m pytest fusion_project/test_multigroup_heavy.py -q -m heavy --run-heavy
```

### Benchmark report generation

Machine-readable reports are generated by the harness in
`multigroup_benchmarks.py`.  The report records:

- group count, mesh size, and quadrature size
- wall time and peak RSS memory when enabled
- convergence status, outer iterations, and total GMRES iterations
- residual history and final residual
- scalar flux `l1`, `l2`, `linf`, sum, integral, min, max, and mean
- source-strength conservation diagnostics
- sparse scattering nonzero counts

Fast report generation through pytest:

```bash
python -m pytest fusion_project/test_multigroup_heavy.py -q -m benchmark --run-benchmark
```

Direct CLI report generation:

```bash
python fusion_project/multigroup_benchmarks.py --tier fast --output fusion_project/data/benchmarks/latest_fast_report.json
python fusion_project/multigroup_benchmarks.py --tier heavy --output fusion_project/data/benchmarks/latest_heavy_report.json
python fusion_project/multigroup_benchmarks.py --tier all --output fusion_project/data/benchmarks/latest_all_report.json
```

Use `--no-memory` to disable peak RSS recording for timing-only runs.

### Baseline artifact and tolerances

The golden software-regression artifact is stored at:

```text
fusion_project/data/benchmarks/synthetic_multigroup_benchmark_baseline.json
```

It contains reference numerical summaries and reference runtime values for the
synthetic cases.  Runtime and memory are recorded for visibility, but regular
CI only enforces numerical/convergence/source regressions because wall time and
peak RSS vary substantially across developer laptops, containers, and shared
CI runners.

Reference baseline in this repository was produced in a Linux container with
Python-level sweeps.  Typical observed runtimes in that environment were about
0.5--1.5 seconds for each fast case and about 8--9 seconds for each heavy case
when peak RSS instrumentation used `resource.getrusage()` rather than Python
allocation tracing.  These numbers are scaling sentinels for software
regression only; they are not production HPC performance targets.

## Golden Fusion Regression Benchmarks

Production-oriented golden regressions live under `fusion_project/data/golden/`
and are validated by `fusion_project/test_golden_benchmarks.py`.  These cases
are deterministic fixed-source problems or analytic manufactured references for
routine drift detection; they are not a substitute for licensed benchmark or
experimental validation data.

### Validation taxonomy (what these tests mean)

1. **Synthetic software regression (golden CI drift checks)**  
   Deterministic fixed-source snapshots used to detect numerical/software
   drift in this repository.
2. **Manufactured / analytic checks**  
   Problems with known analytic behavior (for example slab attenuation trends)
   used to verify discretization/implementation consistency.
3. **External physics validation (future, separate suite)**  
   Independent benchmark or experimental comparisons. Not included in the
   current golden artifacts and must be tracked with separate datasets,
   tolerances, and test markers.

Covered fast CI categories:

- **Shielding** — a single-group fixed-source Cartesian box with a smooth
  source biased toward the inlet side and detector `x`-plane averages/ratios.
- **Slab attenuation** — a 1-D-like pure absorber manufactured exponential
  attenuation profile.  The reference stores the analytic cell-center flux,
  adjacent attenuation ratios, and the tolerance documenting discretization
  expectations.
- **Downscatter spectrum** — a three-group sparse downscatter material with a
  fast fixed source and group-integrated scalar-flux spectrum checks.
- **TBR** — a Li-bearing post-processing benchmark that validates total TBR,
  Li-6 contribution, Li-7 contribution, Li-6 fraction, and breeding-map
  integral.
- **Heating** — an SS316 kerma benchmark that validates integrated power,
  peak/mean volumetric heating, peak-to-mean ratio, and a boundary heat-flux
  proxy.

Each JSON artifact includes versioned metadata:

- `schema_version`
- mesh dimensions and spacing
- synthetic material/library name
- solver configuration and boundary-condition summary
- tolerances used by the test comparison
- convergence metadata, scalar-flux norms, selected detector/cell metrics, and
  post-processing integrals where applicable

Routine validation command:

```bash
python -m pytest fusion_project/test_golden_benchmarks.py -q
```

Opt-in larger production-style checks use the existing `heavy` marker and are
not run by normal CI unless requested:

```bash
python -m pytest fusion_project/test_golden_benchmarks.py -q -m heavy --run-heavy
```

Golden files are never regenerated by tests.  To intentionally regenerate after
a reviewed numerical or benchmark-definition change, run the explicit CLI and
commit the JSON diff with the code review rationale:

```bash
python fusion_project/golden_benchmarks.py --tier fast --write-golden
python fusion_project/golden_benchmarks.py --tier heavy --write-golden
# or regenerate both tiers explicitly:
python fusion_project/golden_benchmarks.py --tier all --write-golden
```

Regeneration should be treated as a numerical-governance event: document why the
reference moved, confirm conservation/stability impact, and verify no hidden
normalization or flux clipping was introduced.


## Dynamic-G promotion checklist and CI gates

Dynamic-G status promotion (`partial` -> `complete`) is governed by a machine-readable
checklist at `fusion_project/dynamic_g_promotion_checklist.yaml` and enforced by
`fusion_project/check_dynamic_g_promotion_policy.py`.

CI lanes:

- `ci-fast` (per-PR required): fast deterministic checks + policy checks.
- `ci-heavy-nightly` (promotion-gating required): heavy deterministic matrix for
  `G={70,175}` with strict performance mode and artifact retention.

Promotion to `complete` requires a recent successful heavy artifact status file
(`fusion_project/data/benchmarks/heavy_ci_status.json`) no older than the
configured freshness window (`max_heavy_age_days`, currently 7).

This policy hardening does not alter transport kernels or solver numerics;
conservation/stability behavior remains governed by existing solver and
validation suites.
