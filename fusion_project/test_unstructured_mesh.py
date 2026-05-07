from __future__ import annotations

import numpy as np

from sn_core import Mesh
from mesh_builder import MeshBuilder


def test_cartesian_conversion_volumes():
    mesh = MeshBuilder.from_cartesian(Mesh(4, 4, 4, 1.0, 1.0, 1.0))
    assert np.isclose(mesh.cell_volume.sum(), 64.0)
    assert np.allclose(mesh.cell_volume, 1.0)


def test_tet_box_volumes():
    mesh = MeshBuilder.tet_box(4, 4, 4, 1.0, 1.0, 1.0)
    assert np.isclose(mesh.cell_volume.sum(), 64.0)
    assert np.all(mesh.cell_volume > 0.0)


def test_face_normals_are_unit_vectors():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    assert np.allclose(np.linalg.norm(mesh.face_normal, axis=1), 1.0, rtol=0.0, atol=1.0e-12)


def test_face_normals_point_outward_from_left_cell():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    for f, (cL, cR) in enumerate(mesh.face_to_cells):
        if cR == -1:
            continue
        assert np.dot(mesh.face_centroid[f] - mesh.cell_centroid[cL], mesh.face_normal[f]) > 0.0


def test_connectivity_symmetry():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    cell_faces = [set(map(int, faces)) for faces in mesh.cell_to_faces]
    for c, faces in enumerate(cell_faces):
        for f in faces:
            cL, cR = map(int, mesh.face_to_cells[f])
            other = cR if c == cL else cL
            if other != -1:
                assert f in cell_faces[other]


def test_boundary_face_accounting():
    mesh = MeshBuilder.tet_box(2, 2, 2, 1.0, 1.0, 1.0)
    tagged = sum(len(v) for v in mesh.boundary_faces.values())
    actual = int(np.count_nonzero(mesh.face_to_cells[:, 1] == -1))
    assert tagged == actual
