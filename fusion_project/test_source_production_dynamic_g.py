from __future__ import annotations

import numpy as np
import pytest

from sn_core import Mesh
from fusion.source import make_dt_source


def test_make_dt_source_requires_explicit_metadata() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="requires energy_bounds or source_group_mapping"):
        make_dt_source(mesh, G=3)


def test_make_dt_source_accepts_explicit_mapping_for_required_g() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    for G in (1, 3, 10, 27, 70, 175):
        m = np.zeros(G, dtype=np.float64)
        m[0] = 1.0
        q = make_dt_source(mesh, G=G, source_group_mapping=m)
        assert q.shape == (2, 2, 2, G)
