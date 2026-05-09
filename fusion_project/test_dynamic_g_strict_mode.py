from __future__ import annotations

import numpy as np
import pytest

from sn_core import Mesh
from fusion.source import make_dt_source, make_dt_source_legacy_group0
from fusion.tbr import compute_tbr_components, compute_tbr_components_legacy_g3


def test_strict_dynamic_g_requires_energy_bounds_for_dt_source() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="requires energy_bounds or source_group_mapping"):
        make_dt_source(mesh, G=3)
    q = make_dt_source_legacy_group0(mesh, G=3)
    assert q.shape == (2,2,2,3)


def test_strict_dynamic_g_rejects_legacy_tbr_split() -> None:
    phi = np.ones((2, 2, 2, 3), dtype=np.float64)
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="forbids legacy_group_semantics"):
        compute_tbr_components(phi, 0.076, mesh, 1.0, legacy_group_semantics=True, strict_dynamic_g=True)
    with pytest.raises(ValueError, match="requires G==3"):
        compute_tbr_components_legacy_g3(np.ones((2,2,2,4)), 0.076, mesh, 1.0)
