"""
fusion — Phase 8 post-processing layer
=======================================

Converts converged solver output (phi, J) into reactor engineering
quantities.  Nothing in this package touches the transport solver.
"""

from fusion.source    import make_dt_source, source_strength
from fusion.materials import FusionMaterial, SS316, Li4SiO4, Beryllium, Helium, Tungsten
from fusion.reactions import (compute_reaction_rate,
                               compute_group_reaction_rates,
                               integrate_reaction_rate)
from fusion.tbr       import compute_tbr, compute_tbr_components, tbr_sensitivity_enrichment
from fusion.heating   import (compute_heating, compute_heating_watts,
                               integrate_power, peak_heat_flux)
from fusion.damage    import compute_dpa, integrate_dpa, peak_dpa
from fusion.outputs   import FusionResults

__all__ = [
    "make_dt_source", "source_strength",
    "FusionMaterial", "SS316", "Li4SiO4", "Beryllium", "Helium", "Tungsten",
    "compute_reaction_rate", "compute_group_reaction_rates",
    "integrate_reaction_rate",
    "compute_tbr", "compute_tbr_components", "tbr_sensitivity_enrichment",
    "compute_heating", "compute_heating_watts",
    "integrate_power", "peak_heat_flux",
    "compute_dpa", "integrate_dpa", "peak_dpa",
    "FusionResults",
]
