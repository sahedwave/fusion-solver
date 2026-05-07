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
- `metadata`: dictionary

`MultigroupLibrary` requires:

- `energy_bounds`: shape `(G + 1,)`, strictly monotonic
- `materials`: dictionary of material key to `MaterialXS`

## File Formats

Use:

- JSON for readable small libraries
- NPZ for dense numerical arrays

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
