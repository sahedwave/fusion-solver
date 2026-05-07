"""Builders and container for unstructured transport meshes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from mesh_geometry import (
    compute_tet_volume, compute_hex_volume, compute_face_normal_and_area,
    build_face_connectivity,
)


@dataclass(frozen=True)
class UnstructuredMesh:
    nodes: np.ndarray
    cell_nodes: list[np.ndarray]
    cell_type: np.ndarray
    cell_volume: np.ndarray | None = None
    cell_centroid: np.ndarray | None = None
    face_area: np.ndarray | None = None
    face_normal: np.ndarray | None = None
    face_centroid: np.ndarray | None = None
    face_to_cells: np.ndarray | None = None
    cell_to_faces: list[np.ndarray] | None = None
    boundary_faces: dict[str, np.ndarray] = field(default_factory=dict)
    # Optional backward-compatibility metadata for Cartesian-equivalent meshes.
    cartesian_shape: tuple[int, int, int] | None = None
    cartesian_spacing: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        cell_nodes = [np.asarray(c, dtype=np.int64) for c in self.cell_nodes]
        cell_type = np.asarray(self.cell_type, dtype=np.int64)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "cell_nodes", cell_nodes)
        object.__setattr__(self, "cell_type", cell_type)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(f"nodes must have shape (N, 3), got {nodes.shape}")
        if len(cell_nodes) == 0:
            raise ValueError("mesh must contain at least one cell")
        if cell_type.shape != (len(cell_nodes),):
            raise ValueError("cell_type must have one entry per cell")

        volumes = self.cell_volume
        centroids = self.cell_centroid
        if volumes is None or centroids is None:
            vols = []
            cents = []
            for cn in cell_nodes:
                pts = nodes[cn]
                cents.append(pts.mean(axis=0))
                if len(cn) == 4:
                    vols.append(compute_tet_volume(pts))
                elif len(cn) == 8:
                    vols.append(compute_hex_volume(pts))
                else:
                    raise ValueError(f"unsupported cell with {len(cn)} nodes")
            volumes = np.asarray(vols, dtype=np.float64)
            centroids = np.asarray(cents, dtype=np.float64)
        else:
            volumes = np.asarray(volumes, dtype=np.float64)
            centroids = np.asarray(centroids, dtype=np.float64)

        ftc = self.face_to_cells
        ctf = self.cell_to_faces
        face_node_ids = None
        if ftc is None or ctf is None:
            ftc, face_node_ids, ctf = build_face_connectivity(cell_nodes)
        ftc = np.asarray(ftc, dtype=np.int64)
        ctf = [np.asarray(f, dtype=np.int64) for f in ctf]

        f_area = self.face_area
        f_normal = self.face_normal
        f_centroid = self.face_centroid
        if f_area is None or f_normal is None or f_centroid is None:
            if face_node_ids is None:
                raise ValueError("face geometry requires face-node connectivity")
            areas = []
            normals = []
            cents = []
            for f, fnodes in enumerate(face_node_ids):
                cL = int(ftc[f, 0])
                fpts = nodes[fnodes]
                normal, area = compute_face_normal_and_area(fpts, centroids[cL])
                areas.append(area)
                normals.append(normal)
                cents.append(fpts.mean(axis=0))
            f_area = np.asarray(areas, dtype=np.float64)
            f_normal = np.asarray(normals, dtype=np.float64)
            f_centroid = np.asarray(cents, dtype=np.float64)
        else:
            f_area = np.asarray(f_area, dtype=np.float64)
            f_normal = np.asarray(f_normal, dtype=np.float64)
            f_centroid = np.asarray(f_centroid, dtype=np.float64)

        bfaces = {str(k): np.asarray(v, dtype=np.int64) for k, v in dict(self.boundary_faces).items()}
        if not bfaces:
            bfaces = {"boundary": np.nonzero(ftc[:, 1] == -1)[0].astype(np.int64)}

        object.__setattr__(self, "cell_volume", volumes)
        object.__setattr__(self, "cell_centroid", centroids)
        object.__setattr__(self, "face_to_cells", ftc)
        object.__setattr__(self, "cell_to_faces", ctf)
        object.__setattr__(self, "face_area", f_area)
        object.__setattr__(self, "face_normal", f_normal)
        object.__setattr__(self, "face_centroid", f_centroid)
        object.__setattr__(self, "boundary_faces", bfaces)

        if np.any(volumes <= 0.0) or not np.all(np.isfinite(volumes)):
            raise ValueError("all cell volumes must be positive and finite")
        if np.any(f_area <= 0.0) or not np.all(np.isfinite(f_area)):
            raise ValueError("all face areas must be positive and finite")
        if not np.allclose(np.linalg.norm(f_normal, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("all face normals must be unit vectors")
        if any(len(f) == 0 for f in ctf):
            raise ValueError("no cell may have zero faces")
        boundary_union = set(int(f) for arr in bfaces.values() for f in arr)
        actual_boundary = set(int(f) for f in np.nonzero(ftc[:, 1] == -1)[0])
        if boundary_union != actual_boundary:
            raise ValueError("boundary face sets must exactly match faces with right cell -1")

    @property
    def N_cells(self) -> int:
        return len(self.cell_nodes)

    @property
    def N_faces(self) -> int:
        return int(self.face_to_cells.shape[0])


class MeshBuilder:
    @staticmethod
    def from_cartesian(mesh) -> UnstructuredMesh:
        nx, ny, nz = int(mesh.nx), int(mesh.ny), int(mesh.nz)
        dx, dy, dz = float(mesh.dx), float(mesh.dy), float(mesh.dz)
        def node_id(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k
        nodes = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    nodes.append((i * dx, j * dy, k * dz))
        cells = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cells.append(np.array([
                        node_id(i, j, k), node_id(i + 1, j, k), node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                        node_id(i, j, k + 1), node_id(i + 1, j, k + 1), node_id(i + 1, j + 1, k + 1), node_id(i, j + 1, k + 1),
                    ], dtype=np.int64))
        mesh_u = UnstructuredMesh(
            nodes=np.asarray(nodes, dtype=np.float64),
            cell_nodes=cells,
            cell_type=np.full(len(cells), 8, dtype=np.int64),
            cartesian_shape=(nx, ny, nz),
            cartesian_spacing=(dx, dy, dz),
        )
        # Retag boundary faces by Cartesian plane.
        tags = {name: [] for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
        tol = 1.0e-12
        for f in np.nonzero(mesh_u.face_to_cells[:, 1] == -1)[0]:
            x, y, z = mesh_u.face_centroid[f]
            if abs(x) < tol: tags["xmin"].append(f)
            elif abs(x - nx * dx) < tol: tags["xmax"].append(f)
            elif abs(y) < tol: tags["ymin"].append(f)
            elif abs(y - ny * dy) < tol: tags["ymax"].append(f)
            elif abs(z) < tol: tags["zmin"].append(f)
            elif abs(z - nz * dz) < tol: tags["zmax"].append(f)
        return UnstructuredMesh(
            nodes=mesh_u.nodes, cell_nodes=mesh_u.cell_nodes, cell_type=mesh_u.cell_type,
            cell_volume=mesh_u.cell_volume, cell_centroid=mesh_u.cell_centroid,
            face_area=mesh_u.face_area, face_normal=mesh_u.face_normal,
            face_centroid=mesh_u.face_centroid, face_to_cells=mesh_u.face_to_cells,
            cell_to_faces=mesh_u.cell_to_faces,
            boundary_faces={k: np.asarray(v, dtype=np.int64) for k, v in tags.items()},
            cartesian_shape=(nx, ny, nz), cartesian_spacing=(dx, dy, dz),
        )

    @staticmethod
    def tet_box(nx, ny, nz, dx, dy, dz) -> UnstructuredMesh:
        base = MeshBuilder.from_cartesian(type("CartesianMesh", (), {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dy, "dz": dz})())
        cells = []
        for hex_nodes in base.cell_nodes:
            n = list(map(int, hex_nodes))
            for tet in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
                cells.append(np.asarray([n[i] for i in tet], dtype=np.int64))
        return UnstructuredMesh(nodes=base.nodes, cell_nodes=cells, cell_type=np.full(len(cells), 4, dtype=np.int64))

    @staticmethod
    def from_gmsh(path: str | Path, boundary_tags: dict[str, int]) -> UnstructuredMesh:
        import meshio
        msh = meshio.read(path)
        nodes = np.asarray(msh.points[:, :3], dtype=np.float64)
        cells = []
        types = []
        for block in msh.cells:
            if block.type == "tetra":
                cells.extend(np.asarray(row, dtype=np.int64) for row in block.data)
                types.extend([4] * len(block.data))
            elif block.type == "hexahedron":
                cells.extend(np.asarray(row, dtype=np.int64) for row in block.data)
                types.extend([8] * len(block.data))
        if not cells:
            raise ValueError("Gmsh file contains no tetrahedral or hexahedral volume cells")
        mesh = UnstructuredMesh(nodes=nodes, cell_nodes=cells, cell_type=np.asarray(types, dtype=np.int64))
        # Physical surface tags vary by meshio version; keep all boundary faces and expose requested names if present.
        boundary = np.nonzero(mesh.face_to_cells[:, 1] == -1)[0].astype(np.int64)
        tags = {name: boundary.copy() for name in boundary_tags} or {"boundary": boundary}
        return UnstructuredMesh(
            nodes=mesh.nodes, cell_nodes=mesh.cell_nodes, cell_type=mesh.cell_type,
            cell_volume=mesh.cell_volume, cell_centroid=mesh.cell_centroid,
            face_area=mesh.face_area, face_normal=mesh.face_normal,
            face_centroid=mesh.face_centroid, face_to_cells=mesh.face_to_cells,
            cell_to_faces=mesh.cell_to_faces, boundary_faces=tags,
        )
