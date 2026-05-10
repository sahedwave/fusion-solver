from __future__ import annotations

import numpy as np
import pytest

from sn_core import Mesh
from fusion.source import make_dt_source, make_dt_source
from fusion.materials import Li4SiO4_legacy_compat
from fusion.tbr import compute_tbr_components


def test_strict_dynamic_g_requires_energy_bounds_for_dt_source() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="requires energy_bounds or source_group_mapping"):
        make_dt_source(mesh, G=3)
    m=np.array([1.0,0.0,0.0])
    q = make_dt_source(mesh, G=3, source_group_mapping=m)
    assert q.shape == (2,2,2,3)


def test_strict_dynamic_g_rejects_legacy_tbr_split() -> None:
    phi = np.ones((2, 2, 2, 4), dtype=np.float64)
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="requires explicit breeding_channels"):
        compute_tbr_components(phi, li_material=Li4SiO4_legacy_compat(G=4, li6_enrichment=0.076), mesh=mesh, source_strength_val=1.0)
