from __future__ import annotations

import numpy as np
import pytest

from sn_core import Mesh
from fusion.source import make_dt_source_legacy_group0
from fusion.tbr import compute_tbr_components_legacy_g3


def test_legacy_source_helper_group0_mapping() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.deprecated_call(match="make_dt_source_legacy_group0 is deprecated"):
        q = make_dt_source_legacy_group0(mesh, G=3)
    assert q.shape == (2, 2, 2, 3)
    assert float(np.sum(q[..., 1:])) == 0.0


def test_legacy_tbr_helper_requires_g3() -> None:
    mesh = Mesh(2, 2, 2, 1.0, 1.0, 1.0)
    with pytest.deprecated_call(match="compute_tbr_components_legacy_g3 is deprecated"):
        with pytest.raises(ValueError, match="requires G==3"):
            compute_tbr_components_legacy_g3(np.ones((2, 2, 2, 4)), 0.5, mesh, 1.0)
