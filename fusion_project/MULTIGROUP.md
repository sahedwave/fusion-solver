# Multigroup Library Support

This project now has a small production-oriented multigroup data layer in
`sn_multigroup.py`.  It is intentionally separate from the sweep and solver
modules so existing transport APIs remain stable.

## Data Model

`MaterialXS` requires:

- `sigma_t`: shape `(G,)`
- `sigma_s0`: shape `(G, G)`
- `sigma_s1`: shape `(G, G)`

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
are present.  Older JSON and NPZ libraries that omit these fields remain valid
and load with `MaterialXS.chi is None` and `MaterialXS.nu_sigma_f is None`.
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

The current dense scattering source has shape:

```text
(nx, ny, nz, n_dir, G)
```

For a `50x50x50`, S8, `G=175` problem, angular flux plus dense scattering
source alone is roughly 26 GiB before solver overhead.  The next production
step is group-blocked or angle-blocked scattering/sweep evaluation.

## Current Tested Scale

Standalone validation covers schema, load/save, source normalization, D-T
mapping, and solver smoke tests at 10 and 27 groups.  The 70-group synthetic
library is generated for loader/memory work but is not yet part of the regular
solver smoke suite because the current Python sweep is too slow for routine
large-group regression.
