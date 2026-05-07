# Prompt to solve failing migration-plan tests 10 and 11

You are fixing unstructured sweep migration tests in `fusion_project/test_fem_sweep.py`.

## Failing commands
- `timeout 120 pytest -q fusion_project/test_fem_sweep.py::test_10_fully_reflective_uniform_flux_unstructured_cartesian -q`
- `timeout 120 pytest -q fusion_project/test_fem_sweep.py::test_11_global_conservation_unstructured_tet_box -q`

## Context
- The project has already chosen **Path B** (explicit step-characteristic migration), not exact diamond-difference equivalence.
- Therefore, tests must not claim strict uniformity/equality behavior that requires structured diamond-difference.
- Keep the tests meaningful for the face-based unstructured discretization.

## Required actions
1. Update Test 10 so it validates realistic behavior for face-based unstructured sweep under fully reflective, uniform source conditions:
   - finite and nonnegative scalar flux,
   - bounded spatial variation using a documented tolerance that reflects current discretization behavior (not 1%).
2. Keep Test 11 as a conservation check on tet mesh, and ensure it runs reliably under timeout in this environment.
3. Add concise in-test comments documenting why strict `<1%` uniformity is not required under Path B.
4. Run both failing pytest commands and ensure they pass.

## Output format
- Provide exact commands run and outcomes.
- If tolerance is changed, explain the numerical reason briefly.
