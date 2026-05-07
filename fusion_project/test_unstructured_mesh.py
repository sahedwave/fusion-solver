from __future__ import annotations

import numpy as np
import pytest

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


def test_from_gmsh_physical_boundary_mapping(tmp_path):
    meshio = pytest.importorskip("meshio")

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=float)
    hexes = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    # xmin and xmax quad surfaces with distinct physical tags
    xmin = np.array([[0, 3, 7, 4]], dtype=int)
    xmax = np.array([[1, 2, 6, 5]], dtype=int)

    msh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", hexes), ("quad", xmin), ("quad", xmax)],
        cell_data={"gmsh:physical": [np.array([1]), np.array([11]), np.array([22])]},
    )
    path = tmp_path / "two_tags_box.msh"
    meshio.write(path, msh, file_format="gmsh22")

    mesh = MeshBuilder.from_gmsh(path, boundary_tags={"xmin": 11, "xmax": 22})

    xmin_faces = np.sort(mesh.boundary_faces["xmin"])
    xmax_faces = np.sort(mesh.boundary_faces["xmax"])
    assert xmin_faces.size > 0
    assert xmax_faces.size > 0
    assert not np.array_equal(xmin_faces, xmax_faces)
    assert "unassigned" in mesh.boundary_faces
    assert mesh.boundary_faces["unassigned"].size > 0
