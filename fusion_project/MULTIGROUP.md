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
