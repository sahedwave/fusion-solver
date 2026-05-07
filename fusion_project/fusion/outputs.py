"""
fusion/outputs.py — Fusion post-processing results container
=============================================================

FusionResults stores all computed engineering observables and provides
output methods (NumPy save, optional VTK export for ParaView).

Design rule: FusionResults is a PURE DATA CONTAINER.
It has no dependency on the transport solver internals.
It stores only NumPy arrays and scalar quantities.

Required output formats (per spec):
    - NumPy arrays (mandatory)  -> .npz archive via save_npz()
    - VTK export (optional)     -> .vts structured grid via export_vtk()
      (requires vtk or pyevtk package; skips gracefully if absent)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FusionResults:
    """
    Container for all fusion physics post-processing outputs.

    All spatial arrays have shape (nx, ny, nz) or (nx, ny, nz, G).

    Attributes
    ----------
    phi : np.ndarray (nx, ny, nz, G)
        Converged scalar flux  [n/cm^2/s].
    reaction_rate : np.ndarray (nx, ny, nz)
        Group-summed absorption rate  [reactions/cm^3/s].
    tbr : float
        Scalar Tritium Breeding Ratio (dimensionless).
    breeding_map : np.ndarray (nx, ny, nz)
        Spatial tritium production rate  [T-atoms/cm^3/s].
    heating_MeV : np.ndarray (nx, ny, nz)
        Volumetric heat deposition  [MeV/cm^3/s].
    heating_W : np.ndarray (nx, ny, nz)
        Volumetric heat deposition  [W/cm^3].
    dpa_rate : np.ndarray (nx, ny, nz)
        Radiation damage rate (DPA proxy)  [displacements/cm^3/s].
    total_power_MW : float
        Domain-integrated power deposition  [MW].
    peak_dpa_rate : float
        Maximum DPA rate in domain  [displacements/cm^3/s].
    mesh : optional Mesh object (for VTK coordinate generation)
    material_name : str
        Name of the material used in post-processing.
    """

    phi:            np.ndarray                  # (nx,ny,nz,G)
    reaction_rate:  np.ndarray                  # (nx,ny,nz)
    tbr:            float
    breeding_map:   np.ndarray                  # (nx,ny,nz)
    heating_MeV:    np.ndarray                  # (nx,ny,nz)
    heating_W:      np.ndarray                  # (nx,ny,nz)
    dpa_rate:       np.ndarray                  # (nx,ny,nz)
    total_power_MW: float
    peak_dpa_rate:  float
    leakage:        float  = 0.0               # energy-integrated leakage [n/s]
    mesh:           object = field(default=None, repr=False)
    material_name:  str    = "unknown"

    # ----------------------------------------------------------------
    # Factory: build from solver output + fusion modules
    # ----------------------------------------------------------------

    @classmethod
    def from_solver(
        cls,
        phi:      np.ndarray,
        mesh,
        material,                           # FusionMaterial
        Q_ext:    np.ndarray,
        li_material=None,                   # FusionMaterial with is_breeder=True
        li_mask:  np.ndarray | None = None,
    ) -> "FusionResults":
        """
        Convenience constructor: runs all post-processing from solver outputs.

        Parameters
        ----------
        phi      : (nx,ny,nz,G)  converged scalar flux
        mesh     : Mesh
        material : FusionMaterial  (for heating, DPA, reaction rate)
        Q_ext    : (nx,ny,nz,G)   external source (for S_DT normalisation)
        li_material : FusionMaterial with is_breeder=True, or None
            If None, uses material itself (only valid if is_breeder=True).
        li_mask  : optional boolean mask for Li-bearing region
        """
        from fusion.reactions import compute_reaction_rate
        from fusion.heating   import compute_heating, compute_heating_watts, integrate_power
        from fusion.damage    import compute_dpa, peak_dpa as _peak_dpa
        from fusion.tbr       import compute_tbr
        from fusion.source    import source_strength

        rr      = compute_reaction_rate(phi, material)
        heat_M  = compute_heating(phi, material)
        heat_W  = compute_heating_watts(phi, material)
        dpa     = compute_dpa(phi, material)
        pow_MW  = integrate_power(phi, material, mesh, unit="MW")
        pk_dpa, _ = _peak_dpa(phi, material)

        S_DT = source_strength(Q_ext, mesh)
        li_mat = li_material if li_material is not None else material
        tbr_val, breed = compute_tbr(phi, li_mat, mesh, S_DT, li_mask)

        # Energy-integrated leakage: S_DT - total_absorption  [n/s]
        # For vacuum BC the global balance is  S = absorption + leakage.
        # We estimate leakage residually; value is 0 for reflective BC.
        from fusion.reactions import integrate_reaction_rate
        abs_total = integrate_reaction_rate(phi, material, mesh)
        leakage   = max(0.0, S_DT - abs_total)

        return cls(
            phi=phi,
            reaction_rate=rr,
            tbr=tbr_val,
            breeding_map=breed,
            heating_MeV=heat_M,
            heating_W=heat_W,
            dpa_rate=dpa,
            total_power_MW=pow_MW,
            peak_dpa_rate=pk_dpa,
            leakage=leakage,
            mesh=mesh,
            material_name=material.name,
        )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def summary(self) -> str:
        """
        Human-readable summary of key engineering quantities.
        """
        lines = [
            "=" * 55,
            "  FUSION REACTOR PHYSICS SUMMARY",
            "=" * 55,
            f"  Material:           {self.material_name}",
            f"  TBR:                {self.tbr:.4f}",
            f"  Total power:        {self.total_power_MW:.4e} MW",
            f"  Peak heat [W/cm3]:  {self.heating_W.max():.4e}",
            f"  Mean heat [W/cm3]:  {self.heating_W.mean():.4e}",
            f"  Peak DPA rate:      {self.peak_dpa_rate:.4e} dpa/cm3/s",
            f"  Peak rxn rate:      {self.reaction_rate.max():.4e} rxn/cm3/s",
            f"  Peak breeding:      {self.breeding_map.max():.4e} T/cm3/s",
            f"  Energy leakage:     {self.leakage:.4e} n/s",
            "=" * 55,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    # ----------------------------------------------------------------
    # Spatial integration helper
    # ----------------------------------------------------------------

    def integrate(self, field: str, mesh=None) -> float:
        """
        Volume-integrate a named spatial field over the full domain.

        Parameters
        ----------
        field : str
            One of: "reaction_rate", "heating_MeV", "heating_W",
                    "dpa_rate", "breeding_map", or "phi_g{g}"
                    (e.g. "phi_g0" for group-0 scalar flux integral).
        mesh : Mesh, optional
            Provides dx, dy, dz.  Falls back to self.mesh if not given.

        Returns
        -------
        float : volume integral  [field_units * cm^3]

        Examples
        --------
        >>> total_abs  = fr.integrate("reaction_rate", mesh)  # reactions/s
        >>> total_heat = fr.integrate("heating_W", mesh)      # W
        >>> fast_flux  = fr.integrate("phi_g0", mesh)         # n/cm^2/s * cm^3
        """
        m = mesh if mesh is not None else self.mesh
        if m is None:
            raise ValueError(
                "integrate() requires a mesh.  Pass mesh= or set self.mesh."
            )
        vol_cell = m.dx * m.dy * m.dz

        named = {
            "reaction_rate": self.reaction_rate,
            "heating_MeV":  self.heating_MeV,
            "heating_W":    self.heating_W,
            "dpa_rate":     self.dpa_rate,
            "breeding_map": self.breeding_map,
        }
        if field in named:
            return float(named[field].sum()) * vol_cell

        # phi_g{g} — group-resolved flux integral
        if field.startswith("phi_g"):
            try:
                g = int(field[5:])
            except ValueError:
                raise ValueError(f"Cannot parse group index from field '{field}'.")
            if g < 0 or g >= self.phi.shape[-1]:
                raise ValueError(
                    f"Group index {g} out of range for phi with G={self.phi.shape[-1]}."
                )
            return float(self.phi[:, :, :, g].sum()) * vol_cell

        raise ValueError(
            f"Unknown field '{field}'. Choose from: "
            + ", ".join(list(named) + ["phi_g{g}"])
        )

    # ----------------------------------------------------------------
    # NumPy save / load (mandatory output format)
    # ----------------------------------------------------------------

    def save_npz(self, path: str) -> None:
        """
        Save all arrays and scalars to a compressed NumPy archive.

        Usage:
            results.save_npz("fusion_output.npz")
            r = FusionResults.load_npz("fusion_output.npz")

        Parameters
        ----------
        path : str
            Output file path (.npz appended automatically if absent).
        """
        np.savez_compressed(
            path,
            phi            = self.phi,
            reaction_rate  = self.reaction_rate,
            tbr            = np.array([self.tbr]),
            breeding_map   = self.breeding_map,
            heating_MeV    = self.heating_MeV,
            heating_W      = self.heating_W,
            dpa_rate       = self.dpa_rate,
            total_power_MW = np.array([self.total_power_MW]),
            peak_dpa_rate  = np.array([self.peak_dpa_rate]),
            leakage        = np.array([self.leakage]),
        )

    @classmethod
    def load_npz(cls, path: str) -> "FusionResults":
        """Load from a .npz archive saved by save_npz()."""
        d = np.load(path)
        return cls(
            phi            = d["phi"],
            reaction_rate  = d["reaction_rate"],
            tbr            = float(d["tbr"][0]),
            breeding_map   = d["breeding_map"],
            heating_MeV    = d["heating_MeV"],
            heating_W      = d["heating_W"],
            dpa_rate       = d["dpa_rate"],
            total_power_MW = float(d["total_power_MW"][0]),
            peak_dpa_rate  = float(d["peak_dpa_rate"][0]),
            # leakage added in Phase 8 patch; default 0.0 for older archives
            leakage        = float(d["leakage"][0]) if "leakage" in d else 0.0,
        )

    # ----------------------------------------------------------------
    # Optional VTK export (ParaView visualization)
    # ----------------------------------------------------------------

    def export_vtk(self, path: str) -> bool:
        """
        Export spatial fields to VTK structured grid (.vts) for ParaView.

        Requires pyevtk:  pip install pyevtk

        Fields exported:
            - scalar_flux_g{g}  : phi[:,:,:,g] per group
            - reaction_rate
            - heating_W
            - dpa_rate
            - breeding_map

        Parameters
        ----------
        path : str
            Output path without extension (pyevtk adds .vts automatically).

        Returns
        -------
        bool : True if export succeeded, False if pyevtk is not installed.
        """
        try:
            from pyevtk.hl import gridToVTK
        except ImportError:
            print(
                "  [VTK] pyevtk not installed. Skipping VTK export.\n"
                "  Install with:  pip install pyevtk"
            )
            return False

        if self.mesh is None:
            print("  [VTK] No mesh attached to FusionResults. Cannot export VTK.")
            return False

        mesh  = self.mesh
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        x = np.linspace(0.0, nx * mesh.dx, nx + 1)
        y = np.linspace(0.0, ny * mesh.dy, ny + 1)
        z = np.linspace(0.0, nz * mesh.dz, nz + 1)

        # pyevtk requires Fortran-contiguous float64 arrays
        def _f(arr):
            return np.asfortranarray(arr.astype(np.float64))

        point_data = {}
        cell_data  = {
            "reaction_rate":  _f(self.reaction_rate),
            "heating_W_cm3":  _f(self.heating_W),
            "dpa_rate":       _f(self.dpa_rate),
            "breeding_map":   _f(self.breeding_map),
        }
        for g in range(self.phi.shape[-1]):
            cell_data[f"phi_g{g}"] = _f(self.phi[:, :, :, g])

        gridToVTK(path, x, y, z, cellData=cell_data)
        print(f"  [VTK] Exported to {path}.vts")
        return True
