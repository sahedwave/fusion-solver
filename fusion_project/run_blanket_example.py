"""
run_blanket_example.py — Phase 8 Fusion Blanket Reference Example
==================================================================

Demonstrates the complete pipeline:

    D-T point source  →  3D Sn transport (GMRES-DSA)
                      →  Phase 8 post-processing
                      →  Engineering outputs

Geometry
--------
  10×10×10 mesh, each cell 2×2×2 cm  →  20×20×20 cm domain
  Material: Li4SiO4 breeder blanket (natural enrichment)
  Source:   point D-T source (14.1 MeV, group 0) at domain centre
  BC:       vacuum on all faces

Outputs
-------
  φ field           — scalar flux per group [n/cm^2/s]
  Q field           — volumetric heating [W/cm^3]
  TBR value         — tritium breeding ratio
  Heating slice     — plane-averaged heating along x-axis
  Li-6/Li-7 split   — isotopic TBR breakdown
  NPZ archive       — fusion_blanket_output.npz
  Validation report — physics sanity checks

Usage
-----
    python run_blanket_example.py

No arguments required.  All outputs printed to stdout + NPZ file.
"""

from __future__ import annotations

import sys
import os
import numpy as np

# ── Path setup (works whether run from project root or any subdirectory) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sn_core import (
    Mesh, BoundaryConditions,
    build_quadrature, build_reflection_map,
    make_3group_p1_material,
)
from sn_solver import SolverConfig, solve_gmres_dsa

from fusion.source     import make_dt_source, source_strength
from fusion.materials  import Li4SiO4, SS316, Tungsten, Helium
from fusion.reactions  import compute_reaction_rate, integrate_reaction_rate
from fusion.tbr        import compute_tbr, compute_tbr_components
from fusion.heating    import compute_heating_watts, integrate_power
from fusion.damage     import compute_dpa, peak_dpa
from fusion.outputs    import FusionResults
from fusion.validation import validate_physics


# ================================================================
# PROBLEM PARAMETERS
# ================================================================

NX         = 10          # cells per axis
CELL_CM    = 2.0         # cell side length [cm]
SN_ORDER   = 4           # S4 quadrature (24 directions)
STRENGTH   = 3.0e17      # D-T source strength [n/s]  (ITER-like)
SOLVER_TOL = 1e-6
OUTPUT_NPZ = "fusion_blanket_output.npz"


# ================================================================
# MAIN DRIVER
# ================================================================

def run() -> FusionResults:
    print("=" * 60)
    print("  Phase 8 — Li4SiO4 Blanket Slab Reference Example")
    print("=" * 60)
    print(f"  Mesh:     {NX}×{NX}×{NX},  cell = {CELL_CM} cm")
    print(f"  Domain:   {NX*CELL_CM:.0f}×{NX*CELL_CM:.0f}×{NX*CELL_CM:.0f} cm")
    print(f"  Source:   point D-T,  S = {STRENGTH:.2e} n/s")
    print(f"  Quadrature: S{SN_ORDER}")
    print()

    # ── 1. Build mesh, quadrature, and solver material ───────────
    mesh     = Mesh(nx=NX, ny=NX, nz=NX,
                    dx=CELL_CM, dy=CELL_CM, dz=CELL_CM)
    mat_sn   = make_3group_p1_material()          # transport XS
    dirs, wts = build_quadrature(SN_ORDER)
    bc        = BoundaryConditions()              # vacuum all faces
    refl_map  = build_reflection_map(dirs)

    G = mat_sn.G
    print(f"  Groups:   G = {G}  (fast | epi | thermal)")

    # ── 2. Build D-T source ──────────────────────────────────────
    src_map = np.zeros(G, dtype=np.float64)
    src_map[0] = 1.0
    Q_ext = make_dt_source(mesh, G=G, geometry="point", strength=STRENGTH, source_group_mapping=src_map)
    S_DT  = source_strength(Q_ext, mesh)
    print(f"  S_DT verified: {S_DT:.6e} n/s  (relative error = "
          f"{abs(S_DT - STRENGTH)/STRENGTH:.2e})")

    # ── 3. Solve with GMRES-DSA ──────────────────────────────────
    print("\n  Running GMRES-DSA solver ...")
    cfg    = SolverConfig(
        tol=SOLVER_TOL, max_outer=30, gmres_restart=30,
        inner_tol=1e-10, verbose=False,
    )
    result = solve_gmres_dsa(
        mesh, mat_sn, Q_ext, dirs, wts, bc, refl_map, cfg
    )
    print(f"  Converged: {result.converged}  |  "
          f"outer iters: {result.n_outer}  |  "
          f"GMRES iters total: {result.n_gmres_total}")
    if result.residuals:
        print(f"  Final residual: {result.residuals[-1]:.2e}")

    phi = result.phi    # (NX, NX, NX, G)

    # ── 4. Fusion post-processing ─────────────────────────────────
    print("\n  Running fusion post-processing ...")

    # Primary structural material for heating / DPA
    mat_fw  = SS316()
    # Breeder material for TBR
    mat_br  = Li4SiO4(li6_enrichment=0.076)   # natural enrichment

    fr = FusionResults.from_solver(
        phi       = phi,
        mesh      = mesh,
        material  = mat_fw,
        Q_ext     = Q_ext,
        li_material = mat_br,
    )

    # ── 5. Li-6 / Li-7 isotopic split ────────────────────────────
    tbr_split = compute_tbr_components(
        phi                 = phi,
        li6_enrichment      = 0.076,
        mesh                = mesh,
        source_strength_val = S_DT,
    )

    # ── 6. Print summary ─────────────────────────────────────────
    print()
    fr.print_summary()

    print()
    print("  TBR isotopic breakdown:")
    print(f"    TBR total  : {tbr_split['tbr_total']:.5f}")
    print(f"    TBR Li-6   : {tbr_split['tbr_li6']:.5f}  "
          f"({tbr_split['li6_fraction']*100:.1f}% of total)")
    print(f"    TBR Li-7   : {tbr_split['tbr_li7']:.5f}  "
          f"({(1-tbr_split['li6_fraction'])*100:.1f}% of total)")

    # ── 7. Heating profile slice (plane averages along x) ─────────
    print()
    print("  Heating profile — plane-averaged Q [W/cm^3] along x-axis:")
    Q_W   = fr.heating_W                          # (NX, NX, NX)
    x_coords = [(i + 0.5) * CELL_CM for i in range(NX)]
    print(f"    {'x [cm]':>8s}   {'Q_mean [W/cm^3]':>18s}   {'Q_max [W/cm^3]':>18s}")
    for i, x in enumerate(x_coords):
        q_plane = Q_W[i, :, :]
        print(f"    {x:8.1f}   {q_plane.mean():18.4e}   {q_plane.max():18.4e}")

    # ── 8. Group-resolved flux at source cell ─────────────────────
    ci = NX // 2
    print()
    print(f"  Scalar flux at source cell ({ci},{ci},{ci}) [n/cm^2/s]:")
    group_names = ["fast (>0.1 MeV)", "epi  (1eV-0.1MeV)", "thermal (<1eV)"]
    for g in range(G):
        label = group_names[g] if g < len(group_names) else f"group {g}"
        print(f"    g={g}  {label:20s}:  {phi[ci, ci, ci, g]:.4e}")

    # ── 9. DPA peak location and value ────────────────────────────
    pk_val, pk_loc = peak_dpa(phi, mat_fw)
    print()
    print(f"  Peak DPA rate: {pk_val:.4e} displacements/cm^3/s")
    print(f"  Peak DPA location: cell {pk_loc}")

    # ── 10. integrate() demonstration ────────────────────────────
    print()
    print("  Integrated quantities via FusionResults.integrate():")
    print(f"    Total absorption rate: "
          f"{fr.integrate('reaction_rate', mesh):.4e} reactions/s")
    print(f"    Total heating:         "
          f"{fr.integrate('heating_W', mesh):.4e} W")
    print(f"    Fast-group flux vol:   "
          f"{fr.integrate('phi_g0', mesh):.4e} n/cm^2/s · cm^3")

    # ── 11. Physics validation ────────────────────────────────────
    print()
    print("  Running physics validation hooks ...")
    vr = validate_physics(fr, Q_ext, mesh, verbose=True)
    if not vr.ok:
        print(f"\n  WARNING: {len(vr.failed)} physics check(s) failed!")

    # ── 12. Save to NPZ ──────────────────────────────────────────
    fr.save_npz(OUTPUT_NPZ)
    print(f"\n  Results saved to: {OUTPUT_NPZ}")

    return fr


# ================================================================
# ADDITIONAL MATERIALS DEMONSTRATION
# ================================================================

def show_material_library() -> None:
    """Print cross-section summary for all available materials."""
    print()
    print("  Material library (G=3):")
    print(f"  {'Name':30s}  {'σ_t(fast)':>10s}  {'σ_a(fast)':>10s}  "
          f"{'σ_dpa(fast)':>12s}  {'breeder':>7s}")
    print("  " + "-" * 80)

    from fusion.materials import SS316, Li4SiO4, Beryllium, Helium, Tungsten
    for mat in [SS316(), Li4SiO4(), Li4SiO4(li6_enrichment=0.5),
                Beryllium(), Helium(), Tungsten()]:
        print(
            f"  {mat.name:30s}  {mat.sigma_t[0]:10.4f}  {mat.sigma_a[0]:10.4f}  "
            f"{mat.sigma_dpa[0]:12.4f}  {'yes' if mat.is_breeder else 'no':>7s}"
        )


if __name__ == "__main__":
    fr = run()
    show_material_library()
    print()
    print("=" * 60)
    print("  Example complete.")
    print("=" * 60)
