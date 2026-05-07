"""
fusion/validation.py — Physics sanity hooks for Phase 8 outputs
================================================================

Standalone callable checks that can be run after any solve + post-process
call to verify that the fusion outputs are physically self-consistent.

These are NOT unit tests (those live in test_phase8.py).
They are lightweight guard rails that can be embedded in production
pipelines to catch obviously wrong results before downstream use.

Usage
-----
    from fusion.validation import validate_physics
    results = FusionResults.from_solver(phi, mesh, mat_fw, Q_ext,
                                        li_material=mat_br)
    validate_physics(results, Q_ext, mesh, verbose=True)

Each check raises PhysicsViolationError on failure, or prints a PASS
line when verbose=True.  All checks are run in sequence; failures are
collected and reported together so the caller sees the full picture.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


class PhysicsViolationError(RuntimeError):
    """Raised when a fusion physics sanity check fails."""


@dataclass
class ValidationResult:
    """Return value of validate_physics()."""
    passed:   list[str]
    failed:   list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def __str__(self) -> str:
        lines = ["=== Fusion Physics Validation ==="]
        for msg in self.passed:
            lines.append(f"  [PASS] {msg}")
        for msg in self.warnings:
            lines.append(f"  [WARN] {msg}")
        for msg in self.failed:
            lines.append(f"  [FAIL] {msg}")
        status = "OK" if self.ok else f"FAILED ({len(self.failed)} violations)"
        lines.append(f"  Status: {status}")
        return "\n".join(lines)


def validate_physics(
    results,                        # FusionResults
    Q_ext:   np.ndarray,            # (nx, ny, nz, G)  external source
    mesh,                           # Mesh
    tbr_max: float = 2.0,
    verbose: bool  = False,
    raise_on_failure: bool = False,
) -> ValidationResult:
    """
    Run all physics sanity checks on a FusionResults object.

    Parameters
    ----------
    results : FusionResults
        Populated by FusionResults.from_solver().
    Q_ext : np.ndarray (nx, ny, nz, G)
        External source used to produce results.phi.
    mesh : Mesh
        Solver mesh for spatial integration.
    tbr_max : float
        Physical upper bound for TBR.  Default 2.0 (no blanket can breed
        more than ~1.5× in practice; 2.0 allows ample margin for tests).
    verbose : bool
        Print each check result to stdout.
    raise_on_failure : bool
        If True, raise PhysicsViolationError on the first failed check.

    Returns
    -------
    ValidationResult
        .ok is True iff all checks passed.
    """
    passed:   list[str] = []
    failed:   list[str] = []
    warnings: list[str] = []

    def _ok(msg: str) -> None:
        passed.append(msg)
        if verbose:
            print(f"  [PASS] {msg}")

    def _fail(msg: str) -> None:
        failed.append(msg)
        if verbose:
            print(f"  [FAIL] {msg}")
        if raise_on_failure:
            raise PhysicsViolationError(msg)

    def _warn(msg: str) -> None:
        warnings.append(msg)
        if verbose:
            print(f"  [WARN] {msg}")

    # ── 1. Non-negativity of flux ─────────────────────────────────
    phi_min = results.phi.min()
    if phi_min >= 0.0:
        _ok(f"Scalar flux non-negative everywhere (min={phi_min:.2e})")
    else:
        _fail(f"Scalar flux has negative values: min={phi_min:.2e}")

    # ── 2. Non-negativity of energy deposition ────────────────────
    heat_min = results.heating_W.min()
    if heat_min >= 0.0:
        _ok(f"Energy deposition non-negative everywhere (min={heat_min:.2e} W/cm^3)")
    else:
        _fail(f"Negative energy deposition: min={heat_min:.2e} W/cm^3")

    # ── 3. Non-negativity of DPA ──────────────────────────────────
    dpa_min = results.dpa_rate.min()
    if dpa_min >= 0.0:
        _ok(f"DPA rate non-negative everywhere (min={dpa_min:.2e})")
    else:
        _fail(f"Negative DPA rate: min={dpa_min:.2e}")

    # ── 4. Non-negativity of breeding map ─────────────────────────
    breed_min = results.breeding_map.min()
    if breed_min >= 0.0:
        _ok(f"Breeding map non-negative everywhere (min={breed_min:.2e} T/cm^3/s)")
    else:
        _fail(f"Negative breeding map: min={breed_min:.2e} T/cm^3/s")

    # ── 5. TBR in physical range (0, tbr_max) ─────────────────────
    if results.tbr <= 0.0:
        _fail(f"TBR is non-positive: {results.tbr:.4f}  (must be > 0)")
    elif results.tbr > tbr_max:
        _fail(f"TBR={results.tbr:.4f} exceeds physical bound {tbr_max:.1f}")
    else:
        _ok(f"TBR in physical range: {results.tbr:.4f}  (bound: 0 < TBR <= {tbr_max})")

    # ── 6. Heating localises at source ────────────────────────────
    # Source peak is in group 0 (fast).  Find the spatial cell with
    # the largest Q_ext[...,0] and check that heating is also maximum
    # in the same region (nearest-cell criterion, ±1 cell tolerance).
    src_idx = np.unravel_index(
        np.argmax(Q_ext[:, :, :, 0]), Q_ext[:, :, :, 0].shape
    )
    heat_idx = np.unravel_index(
        np.argmax(results.heating_W), results.heating_W.shape
    )
    dist = int(max(abs(src_idx[d] - heat_idx[d]) for d in range(3)))
    if dist <= 1:
        _ok(
            f"Heating peak at/adjacent to source cell "
            f"(src={src_idx}, heat_peak={heat_idx}, dist={dist})"
        )
    else:
        _fail(
            f"Heating peak ({heat_idx}) is {dist} cells from source ({src_idx}). "
            f"Expected dist <= 1."
        )

    # ── 7. Reaction rates correlate with flux ─────────────────────
    # Pearson-like check: spatial correlation between group-summed flux
    # and reaction rate must be positive (both derived from the same phi).
    phi_sum  = results.phi.sum(axis=-1).ravel()    # (nx*ny*nz,)
    rr_flat  = results.reaction_rate.ravel()
    if phi_sum.std() > 1e-30 and rr_flat.std() > 1e-30:
        corr = float(np.corrcoef(phi_sum, rr_flat)[0, 1])
        if corr > 0.99:
            _ok(f"Reaction rate strongly correlated with flux (r={corr:.4f})")
        elif corr > 0.90:
            _warn(f"Reaction rate correlation with flux is moderate (r={corr:.4f})")
        else:
            _fail(
                f"Reaction rate poorly correlated with flux (r={corr:.4f}). "
                f"Expected r > 0.90 — cross-section or flux issue."
            )
    else:
        _warn("Flux or reaction rate has no spatial variation — correlation skipped.")

    # ── 8. Total power is positive ────────────────────────────────
    if results.total_power_MW > 0.0:
        _ok(f"Total power positive: {results.total_power_MW:.4e} MW")
    else:
        _fail(f"Total power non-positive: {results.total_power_MW:.4e} MW")

    # ── 9. Source group-0 only ────────────────────────────────────
    G = Q_ext.shape[-1]
    if G > 1:
        non_fast_sum = Q_ext[:, :, :, 1:].sum()
        if non_fast_sum == 0.0:
            _ok("D-T source correctly confined to group 0 (fast) only")
        else:
            _fail(
                f"D-T source has non-zero values in groups 1..{G-1}: "
                f"sum={non_fast_sum:.2e}  (source must be in group 0 only)"
            )

    # ── 10. Global balance warning (vacuum BC expected to leak) ───
    from fusion.source import source_strength
    S_DT = source_strength(Q_ext, mesh)
    if S_DT > 0:
        leakage_frac = results.leakage / S_DT
        if 0.0 <= leakage_frac < 1.0:
            _ok(
                f"Global balance: leakage/S_DT = {leakage_frac:.3f} "
                f"(absorption fraction = {1-leakage_frac:.3f})"
            )
        elif leakage_frac >= 1.0:
            _warn(
                f"Leakage fraction = {leakage_frac:.3f} >= 1 — "
                "this is expected only for very optically thin domains."
            )

    if verbose:
        print(f"  --- {len(passed)} passed, {len(warnings)} warnings, "
              f"{len(failed)} failed ---")

    return ValidationResult(passed=passed, failed=failed, warnings=warnings)
