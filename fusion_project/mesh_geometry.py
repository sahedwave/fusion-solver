"""Geometry utilities for unstructured finite-element meshes."""
from __future__ import annotations

from collections import deque
import numpy as np

TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def compute_tet_volume(nodes: np.ndarray) -> float:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.shape != (4, 3):
        raise ValueError(f"tet nodes must have shape (4, 3), got {nodes.shape}")
    return float(abs(np.linalg.det(np.vstack((nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0])))) / 6.0)


def compute_hex_volume(nodes: np.ndarray) -> float:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.shape != (8, 3):
        raise ValueError(f"hex nodes must have shape (8, 3), got {nodes.shape}")
    # Decompose a convex hex with the conventional Gmsh/VTK node order into six tets.
    tets = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    return float(sum(compute_tet_volume(nodes[list(t)]) for t in tets))


def compute_face_normal_and_area(face_nodes: np.ndarray, cell_centroid: np.ndarray) -> tuple[np.ndarray, float]:
    face_nodes = np.asarray(face_nodes, dtype=np.float64)
    if face_nodes.ndim != 2 or face_nodes.shape[1] != 3 or face_nodes.shape[0] not in (3, 4):
        raise ValueError(f"face nodes must have shape (3, 3) or (4, 3), got {face_nodes.shape}")
    centroid = face_nodes.mean(axis=0)
    if face_nodes.shape[0] == 3:
        area_vec = 0.5 * np.cross(face_nodes[1] - face_nodes[0], face_nodes[2] - face_nodes[0])
    else:
        area_vec = 0.5 * np.cross(face_nodes[1] - face_nodes[0], face_nodes[2] - face_nodes[0])
        area_vec += 0.5 * np.cross(face_nodes[2] - face_nodes[0], face_nodes[3] - face_nodes[0])
    area = float(np.linalg.norm(area_vec))
    if area <= 0.0:
        raise ValueError("face area must be positive")
    normal = area_vec / area
    if float(np.dot(normal, centroid - np.asarray(cell_centroid))) < 0.0:
        normal = -normal
    return normal.astype(np.float64), area


def _local_faces(nodes: np.ndarray) -> list[tuple[int, ...]]:
    n = len(nodes)
    if n == 4:
        return [tuple(int(nodes[i]) for i in face) for face in TET_FACES]
    if n == 8:
        return [tuple(int(nodes[i]) for i in face) for face in HEX_FACES]
    raise ValueError(f"unsupported cell with {n} nodes")


def build_face_connectivity(cell_nodes: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    face_map: dict[tuple[int, ...], int] = {}
    face_to_cells: list[list[int]] = []
    face_node_ids: list[np.ndarray] = []
    cell_to_faces: list[list[int]] = [[] for _ in cell_nodes]
    for c, nodes in enumerate(cell_nodes):
        for face in _local_faces(np.asarray(nodes, dtype=np.int64)):
            key = tuple(sorted(face))
            if key not in face_map:
                fid = len(face_to_cells)
                face_map[key] = fid
                face_to_cells.append([c, -1])
                face_node_ids.append(np.asarray(face, dtype=np.int64))
            else:
                fid = face_map[key]
                if face_to_cells[fid][1] != -1:
                    raise ValueError(f"face {key} belongs to more than two cells")
                face_to_cells[fid][1] = c
            cell_to_faces[c].append(fid)
    return np.asarray(face_to_cells, dtype=np.int64), face_node_ids, [np.asarray(f, dtype=np.int64) for f in cell_to_faces]


def upwind_cell(face_id: int, direction: np.ndarray, face_to_cells: np.ndarray, face_normal: np.ndarray) -> tuple[int, int]:
    cL, cR = map(int, face_to_cells[face_id])
    s = float(np.dot(direction, face_normal[face_id]))
    if cR == -1:
        return (cL, -1) if s > 0.0 else (-1, cL)
    return (cL, cR) if s > 0.0 else (cR, cL)


def _compute_sweep_order(mesh, direction: np.ndarray) -> np.ndarray:
    cart_shape = getattr(mesh, "cartesian_shape", None)
    if cart_shape is not None:
        nx, ny, nz = map(int, cart_shape)
        direction = np.asarray(direction, dtype=np.float64)
        i_range = range(nx) if direction[0] >= 0.0 else range(nx - 1, -1, -1)
        j_range = range(ny) if direction[1] >= 0.0 else range(ny - 1, -1, -1)
        k_range = range(nz) if direction[2] >= 0.0 else range(nz - 1, -1, -1)
        return np.asarray(
            [(i * ny + j) * nz + k for i in i_range for j in j_range for k in k_range],
            dtype=np.int64,
        )

    n_cells = mesh.N_cells
    indeg = np.zeros(n_cells, dtype=np.int64)
    adj: list[list[int]] = [[] for _ in range(n_cells)]
    for f in range(mesh.N_faces):
        cL, cR = map(int, mesh.face_to_cells[f])
        if cR < 0:
            continue
        s = float(np.dot(direction, mesh.face_normal[f]))
        if s > 1.0e-14:
            up, down = cL, cR
        elif s < -1.0e-14:
            up, down = cR, cL
        else:
            continue
        adj[up].append(down)
        indeg[down] += 1
    q = deque(int(i) for i in np.nonzero(indeg == 0)[0])
    order = []
    while q:
        c = q.popleft()
        order.append(c)
        for d in adj[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    if len(order) != n_cells:
        # Cyclic/curved meshes: fall back to projection ordering, deterministic and robust.
        centers = mesh.cell_centroid @ np.asarray(direction, dtype=np.float64)
        return np.argsort(centers, kind="mergesort").astype(np.int64)
    return np.asarray(order, dtype=np.int64)
