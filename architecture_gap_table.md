# Architecture Gap Table

| Area | Previous Risk | Current Status | Evidence | Remaining Limitations |
|---|---:|---|---|---|
| Spatial Discretization | 🔴 High | ✅ Resolved | Migration validation passed for the structured validation suite, Phase 8 suite, unstructured mesh geometry/Gmsh import tests, and FEM/unstructured sweep tests. The structured blanket example runs successfully. An equivalent `MeshBuilder.from_cartesian(...)` blanket solve matches the original structured solve with L2 relative difference `0.000000e+00`, satisfying the `1e-5` comparison target. | True non-Cartesian unstructured sweeps use the documented face-balance step-characteristic update, not Cartesian diamond-difference algebra. Reflective unstructured boundaries require the quadrature set to contain the specular reflected direction for the boundary normal; otherwise the solver raises a clear `ValueError`. The fallback Gmsh reader intentionally supports only the ASCII v2 subset used by deterministic test meshes; install `meshio>=5.3` for broader Gmsh support. |
| Gmsh Mesh Import | 🟡 Medium | ✅ Resolved | `MeshBuilder.from_gmsh()` loads the checked-in `box_8x8x8.msh`, maps physical boundary groups, and validates positive volumes and unit normals in tests. | Binary/newer Gmsh variants should use `meshio`; the internal fallback is deliberately minimal. |
| Unstructured Boundary Conditions | 🔴 High | ✅ Resolved | Boundary tags now affect unstructured sweeps: vacuum inflow is zero and reflective inflow uses reflected angular directions. Tests cover vacuum tet boxes, Cartesian-derived reflective boxes, and tagged reflective unstructured boundaries. | Named physical tags default to vacuum unless explicitly mapped with `BoundaryConditions(boundary_types={...})` or named `"vacuum"`/`"reflective"`. |
| Cartesian Reduction | 🟡 Medium | ✅ Resolved | Cartesian-derived unstructured meshes carry metadata used for deterministic lexicographic sweep ordering, Cartesian DD sweep reduction, and Cartesian FD DSA assembly; numerical tests compare sweep order, DSA matrices/apply results, total volume, and flux integrals. | This exact reduction applies only to meshes produced by `MeshBuilder.from_cartesian()` with intact `cartesian_shape` and `cartesian_spacing` metadata. |

## Final Migration Validation

Commands run with `/root/.pyenv/versions/3.14.4/bin/python`:

- `python -m pytest fusion_project/sn_validation.py`
- `python -m pytest fusion_project/test_phase8.py`
- `python -m pytest fusion_project/test_unstructured_mesh.py`
- `python -m pytest fusion_project/test_fem_sweep.py`
- `python fusion_project/run_blanket_example.py`
- Structured-vs-`from_cartesian` blanket comparison script using the same mesh size, material, source, quadrature, boundary conditions, and solver configuration as `run_blanket_example.py`.

The `from_cartesian` blanket comparison produced:

```text
structured True 2 26
unstructured True 2 26
L2 relative difference: 0.000000e+00
max absolute difference: 0.000000e+00
1e-5 equivalence pass: True
```
