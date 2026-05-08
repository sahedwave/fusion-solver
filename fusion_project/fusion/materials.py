"""
fusion/materials.py — Fusion material cross-section library
============================================================

Provides multi-group macroscopic cross-section data for three key
fusion reactor materials:

    SS316    — 316L stainless steel (first wall / structural)
    Li4SiO4  — lithium orthosilicate breeder blanket (is_breeder=True)
    Beryllium — Be neutron multiplier / reflector

Energy group structure (3-group default, matching sn_core.make_3group_p1_material)
-----------------------------------------------------------------------------------
    g=0  fast     E > 0.1 MeV
    g=1  epi      1 eV < E < 0.1 MeV
    g=2  thermal  E < 1 eV

All macroscopic cross-sections in cm^-1 (density already folded in).

Cross-section philosophy
------------------------
Representative 3-group collapsed constants derived from FENDL-3.2
processed with NJOY2021 for an ITER-like neutron spectrum
(IAEA-TECDOC-1235, Ref. Petti et al. 2000).

For G != 3: _uniform_fill applies a 1/(1+g) spectral weight decay
from the fast-group value so the module works with any energy group
structure used by the solver.

All arrays must have length G.  FusionMaterial.__post_init__ enforces
this at construction time.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class FusionMaterial:
    """
    Material descriptor for fusion post-processing calculations.

    All cross-section arrays must have length G (number of energy groups).
    Units: cm^-1 (macroscopic cross-sections, density already included).

    Attributes
    ----------
    name : str
        Human-readable label.
    G : int
        Number of energy groups.
    sigma_t : np.ndarray (G,)
        Total macroscopic cross-section [cm^-1].
    sigma_a : np.ndarray (G,)
        Absorption macroscopic cross-section [cm^-1].
        For Li-bearing materials, this encodes the (n,alpha) breeding
        reaction cross-section used for TBR calculation.
    sigma_dpa : np.ndarray (G,)
        Displacement cross-section [cm^-1] — simplified NRT proxy.
        DPA(x,y,z) proportional to sum_g  sigma_dpa[g] * phi_g(x,y,z)
    energy_deposition : np.ndarray (G,)
        Mean energy deposited per absorption event [MeV] (kerma factors).
        Used for heating:  Q[x,y,z] = sum_g  E_dep[g] * sigma_a[g] * phi_g
    is_breeder : bool
        True if this material contributes to tritium breeding.
        Only is_breeder=True materials are integrated in TBR.
    density : float
        Material density [g/cm^3]. Informational only; cross-sections
        already include this density.
    """
    name:              str
    G:                 int
    sigma_t:           np.ndarray
    sigma_a:           np.ndarray
    sigma_dpa:         np.ndarray
    energy_deposition: np.ndarray
    is_breeder:        bool  = False
    density:           float = 1.0
    breeding_channels: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        for attr in ("sigma_t", "sigma_a", "sigma_dpa", "energy_deposition"):
            arr = getattr(self, attr)
            if len(arr) != self.G:
                raise ValueError(
                    f"{self.name}.{attr}: expected length {self.G}, got {len(arr)}"
                )
        # Enforce non-negative cross-sections
        for attr in ("sigma_t", "sigma_a", "sigma_dpa", "energy_deposition"):
            arr = getattr(self, attr)
            if np.any(arr < 0.0):
                raise ValueError(
                    f"{self.name}.{attr}: contains negative values — "
                    "cross-sections must be non-negative."
                )
        if self.breeding_channels is not None:
            channels = {}
            for key, values in dict(self.breeding_channels).items():
                arr = np.asarray(values, dtype=np.float64)
                if arr.shape != (self.G,):
                    raise ValueError(f"{self.name}.breeding_channels[{key!r}] expected shape {(self.G,)}, got {arr.shape}")
                if np.any(arr < 0.0):
                    raise ValueError(f"{self.name}.breeding_channels[{key!r}] contains negative values")
                channels[str(key)] = arr
            self.breeding_channels = channels


# ================================================================
# 3-GROUP MATERIAL FACTORIES
#
# Group structure:  g=0 fast,  g=1 epi,  g=2 thermal
#
# Sources:
#   - FENDL-3.2 processed with NJOY2021 for 316L-SS and Li4SiO4
#   - 3-group collapse over ITER blanket neutron spectrum
#   - DPA cross-sections from NRT threshold model (~40 eV)
#   - Heating from kerma factors (ENDF/B-VIII.0)
#
# Values are representative for scoping / benchmarking.
# ================================================================

def SS316(G: int = 3) -> FusionMaterial:
    """
    316L Stainless Steel — first wall / structural material.

    Composition (wt%): Fe 65, Cr 17, Ni 12, Mo 2.5, Mn 2, Si 1.
    Density: 7.99 g/cm^3.

    Key physics:
      - Dominant DPA producer at first wall (fast flux)
      - High kerma energy deposition in fast group
      - NOT a tritium breeder (is_breeder=False)

    Cross-section values [cm^-1]:
      sigma_t:           fast=0.282, epi=0.520, thermal=0.890
      sigma_a:           fast=0.008, epi=0.012, thermal=0.045
      sigma_dpa:         fast=0.045, epi=0.018, thermal=0.003
      energy_deposition: fast=6.50,  epi=2.10,  thermal=0.80  [MeV]
    """
    if G == 3:
        return FusionMaterial(
            name              = "316L Stainless Steel",
            G                 = 3,
            density           = 7.99,
            is_breeder        = False,
            sigma_t           = np.array([0.282, 0.520, 0.890]),
            sigma_a           = np.array([0.008, 0.012, 0.045]),
            sigma_dpa         = np.array([0.045, 0.018, 0.003]),
            energy_deposition = np.array([6.50,  2.10,  0.80]),
        )
    return _uniform_fill("316L Stainless Steel", G, 7.99, False,
                         sigma_t0=0.282, sigma_a0=0.008,
                         sigma_dpa0=0.045, edep0=6.50)


def Li4SiO4(G: int = 3, li6_enrichment: float = 0.076) -> FusionMaterial:
    """
    Li4SiO4 — lithium orthosilicate tritium breeder blanket.

    Density: 2.39 g/cm^3.
    Natural Li: 7.6% Li-6, 92.4% Li-7.

    Key physics:
      - Li-6(n,alpha)T: large thermal cross-section (~940 b at 0.025 eV)
        -> dominant TBR contribution at thermal energies (group 2)
      - Li-7(n,n'alpha)T: threshold ~2.5 MeV -> fast-group contribution
        (smaller sigma, ~0.3 b)
      - sigma_a stores the breeding reaction cross-section used for TBR
      - is_breeder = True

    Enrichment scaling:
      sigma_a_thermal  scales linearly with Li-6 fraction
      sigma_a_fast     scales with Li-7 fraction (Li-7 threshold reaction)

    Parameters
    ----------
    li6_enrichment : float
        Atomic fraction of Li-6 (0.076 = natural, up to ~0.90 for
        enriched designs such as ITER/DEMO blanket modules).
    """
    nat_li6  = 0.076
    scale_li6 = li6_enrichment / nat_li6
    scale_li7 = (1.0 - li6_enrichment) / (1.0 - nat_li6)

    if G == 3:
        # Natural-composition reference values [cm^-1]
        sigma_a_nat = np.array([0.004, 0.010, 0.180])
        # fast group driven by Li-7 threshold; epi/thermal by Li-6
        sigma_a_scaled = sigma_a_nat * np.array([scale_li7, scale_li6, scale_li6])

        return FusionMaterial(
            name              = f"Li4SiO4 (Li-6 {li6_enrichment*100:.1f}%)",
            G                 = 3,
            density           = 2.39,
            is_breeder        = True,
            sigma_t           = np.array([0.148, 0.212, 0.480]),
            sigma_a           = sigma_a_scaled,
            sigma_dpa         = np.array([0.002, 0.001, 0.0003]),
            # Li-6(n,alpha)T Q-value = 4.78 MeV; fast Q-value lower
            energy_deposition = np.array([4.80,  1.50,  4.78]),
            breeding_channels = {
                "li6_breeding": sigma_a_scaled * np.array([0.0, 1.0, 1.0]),
                "li7_breeding": sigma_a_scaled * np.array([1.0, 0.0, 0.0]),
            },
        )
    return _uniform_fill(f"Li4SiO4 ({li6_enrichment*100:.1f}%)", G, 2.39, True,
                         sigma_t0=0.148, sigma_a0=0.010,
                         sigma_dpa0=0.002, edep0=4.78)


def Beryllium(G: int = 3) -> FusionMaterial:
    """
    Beryllium — neutron multiplier / reflector.

    Density: 1.85 g/cm^3.

    Key physics:
      - Be(n,2n) at E > 1.84 MeV: neutron multiplication in fast group
      - Good moderator (A=9, small inelastic threshold)
      - Low absorption cross-section
      - NOT a tritium breeder (is_breeder=False)
    """
    if G == 3:
        return FusionMaterial(
            name              = "Beryllium",
            G                 = 3,
            density           = 1.85,
            is_breeder        = False,
            sigma_t           = np.array([0.421, 0.650, 0.780]),
            sigma_a           = np.array([0.002, 0.003, 0.006]),
            sigma_dpa         = np.array([0.030, 0.010, 0.001]),
            energy_deposition = np.array([1.20,  0.50,  0.08]),
        )
    return _uniform_fill("Beryllium", G, 1.85, False,
                         sigma_t0=0.421, sigma_a0=0.002,
                         sigma_dpa0=0.030, edep0=1.20)


def Helium(G: int = 3) -> FusionMaterial:
    """
    Helium coolant — void / low-density approximation.

    Density: 0.000164 g/cm^3  (He at 8 MPa, 300 °C, typical TBM coolant).

    Key physics:
      - Negligible absorption and scattering: acts as a near-void
      - Used to model helium-cooled blanket channels
      - Very small sigma_t: neutrons stream through without interaction
      - NOT a tritium breeder (is_breeder=False)
      - DPA negligible (no solid lattice to displace)

    Cross-section values [cm^-1] — all near zero:
      sigma_t:           fast=0.00027, epi=0.00031, thermal=0.00038
      sigma_a:           fast=0.0,     epi=0.0,     thermal=0.00001
      sigma_dpa:         fast=0.0,     epi=0.0,     thermal=0.0
      energy_deposition: fast=0.0,     epi=0.0,     thermal=0.0
    """
    if G == 3:
        return FusionMaterial(
            name              = "Helium (coolant)",
            G                 = 3,
            density           = 0.000164,
            is_breeder        = False,
            sigma_t           = np.array([0.00027, 0.00031, 0.00038]),
            sigma_a           = np.array([0.0,     0.0,     0.00001]),
            sigma_dpa         = np.array([0.0,     0.0,     0.0    ]),
            energy_deposition = np.array([0.0,     0.0,     0.0    ]),
        )
    decay = np.array([1.0 / (1.0 + g) for g in range(G)])
    return FusionMaterial(
        name              = "Helium (coolant)",
        G                 = G,
        density           = 0.000164,
        is_breeder        = False,
        sigma_t           = 0.00027 * decay,
        sigma_a           = np.zeros(G),
        sigma_dpa         = np.zeros(G),
        energy_deposition = np.zeros(G),
    )


def Tungsten(G: int = 3) -> FusionMaterial:
    """
    Tungsten (W) — first-wall / divertor armour material.

    Density: 19.3 g/cm^3.

    Key physics:
      - Very high Z (Z=74): large fast-group total cross-section
      - Significant resonance capture in epi group
      - High DPA rate: heavy nucleus, low displacement threshold (~40 eV)
      - Plasma-facing component: peak neutron wall load material
      - NOT a tritium breeder (is_breeder=False)

    Cross-section values [cm^-1] (FENDL-3.2 / ENDF/B-VIII.0, W-nat):
      sigma_t:           fast=0.647, epi=1.102, thermal=1.850
      sigma_a:           fast=0.018, epi=0.065, thermal=0.280
      sigma_dpa:         fast=0.095, epi=0.038, thermal=0.006
      energy_deposition: fast=5.80,  epi=2.50,  thermal=0.90  [MeV]
    """
    if G == 3:
        return FusionMaterial(
            name              = "Tungsten (W)",
            G                 = 3,
            density           = 19.3,
            is_breeder        = False,
            sigma_t           = np.array([0.647, 1.102, 1.850]),
            sigma_a           = np.array([0.018, 0.065, 0.280]),
            sigma_dpa         = np.array([0.095, 0.038, 0.006]),
            energy_deposition = np.array([5.80,  2.50,  0.90]),
        )
    return _uniform_fill("Tungsten (W)", G, 19.3, False,
                         sigma_t0=0.647, sigma_a0=0.018,
                         sigma_dpa0=0.095, edep0=5.80)


# ================================================================
# Internal helper
# ================================================================

def _uniform_fill(
    name: str, G: int, density: float, is_breeder: bool,
    sigma_t0: float, sigma_a0: float, sigma_dpa0: float, edep0: float,
) -> FusionMaterial:
    """
    Fallback for G != 3: fill G groups using a 1/(1+g) spectral decay
    from the fast-group (g=0) values.  This ensures the module is
    compatible with any energy group structure used by the solver,
    while preserving the correct ordering (fast > epi > thermal).
    """
    decay = np.array([1.0 / (1.0 + g) for g in range(G)])
    return FusionMaterial(
        name              = name,
        G                 = G,
        density           = density,
        is_breeder        = is_breeder,
        sigma_t           = sigma_t0   * decay,
        sigma_a           = sigma_a0   * decay,
        sigma_dpa         = sigma_dpa0 * decay,
        energy_deposition = edep0      * decay,
    )
