from __future__ import annotations

import numpy as np
import pytest

from sn_core import Mesh
from fusion.source import make_dt_source
from fusion.tbr import compute_tbr_components


def test_strict_dynamic_g_requires_energy_bounds_for_dt_source() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="strict_dynamic_g=True"):
        make_dt_source(mesh, G=3, strict_dynamic_g=True)


def test_strict_dynamic_g_rejects_legacy_tbr_split() -> None:
    phi = np.ones((2, 2, 2, 3), dtype=np.float64)
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="forbids legacy_group_semantics"):
        compute_tbr_components(phi, 0.076, mesh, 1.0, legacy_group_semantics=True, strict_dynamic_g=True)
