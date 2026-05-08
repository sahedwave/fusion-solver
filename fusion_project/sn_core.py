from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Mesh:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        for name in ("dx", "dy", "dz"):
            value = getattr(self, name)
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite, got {value}")


@dataclass(frozen=True)
class BoundaryConditions:
    xmin: bool = False
    xmax: bool = False
    ymin: bool = False
    ymax: bool = False
    zmin: bool = False
    zmax: bool = False
    boundary_types: dict[str, str] = field(default_factory=dict)

    def is_reflective(self, face: str) -> bool:
        if face not in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}:
            raise ValueError(f"unknown boundary face {face!r}")
        return bool(getattr(self, face))

    def boundary_type(self, tag: str) -> str:
        """Return the unstructured boundary condition for a named face tag.

        Unstructured meshes use their ``boundary_faces`` keys as boundary tags.
        Tags named ``"vacuum"`` or ``"reflective"`` are interpreted directly;
        Cartesian face tags (``xmin``/``xmax``/...) keep using the six legacy
        boolean flags; all other physical tags default to vacuum unless an
        explicit ``boundary_types`` entry maps the tag to ``"vacuum"`` or
        ``"reflective"``.
        """
        tag = str(tag)
        if tag in self.boundary_types:
            kind = str(self.boundary_types[tag]).lower()
            if kind not in {"vacuum", "reflective"}:
                raise ValueError(f"unknown boundary type {self.boundary_types[tag]!r} for tag {tag!r}")
            return kind
        kind = tag.lower()
        if kind in {"vacuum", "reflective"}:
            return kind
        if tag in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}:
            return "reflective" if self.is_reflective(tag) else "vacuum"
        return "vacuum"


def _as_scattering_csc(values: object, G: int, name: str) -> sparse.csc_matrix:
    if sparse.issparse(values):
        matrix = values.astype(np.float64).tocsc()
    else:
        matrix = sparse.csc_matrix(np.asarray(values, dtype=np.float64))
    if matrix.shape != (G, G):
        raise ValueError(f"{name} must have shape {(G, G)}, got {matrix.shape}")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


@dataclass(frozen=True)
class P1Material:
    sigma_t: np.ndarray
    sigma_s0: np.ndarray
    sigma_s1: np.ndarray

    def __post_init__(self) -> None:
        sigma_t = np.asarray(self.sigma_t, dtype=np.float64)
        if sigma_t.ndim != 1:
            raise ValueError(f"sigma_t must have shape (G,), got {sigma_t.shape}")
        G = sigma_t.shape[0]
        sigma_s0_csc = _as_scattering_csc(self.sigma_s0, G, "sigma_s0")
        sigma_s1_csc = _as_scattering_csc(self.sigma_s1, G, "sigma_s1")
        sigma_s0 = sigma_s0_csc.toarray()
        sigma_s1 = sigma_s1_csc.toarray()

        if sigma_s0.shape != (G, G):
            raise ValueError(f"sigma_s0 must have shape {(G, G)}, got {sigma_s0.shape}")
        if sigma_s1.shape != (G, G):
            raise ValueError(f"sigma_s1 must have shape {(G, G)}, got {sigma_s1.shape}")
        for name, arr in (("sigma_t", sigma_t), ("sigma_s0", sigma_s0), ("sigma_s1", sigma_s1)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(sigma_t <= 0.0):
            raise ValueError("sigma_t entries must be positive")
        if np.any(sigma_s0_csc.data < 0.0):
            raise ValueError("sigma_s0 entries must be non-negative")

        object.__setattr__(self, "sigma_t", sigma_t)
        object.__setattr__(self, "sigma_s0", sigma_s0)
        object.__setattr__(self, "sigma_s1", sigma_s1)
        object.__setattr__(self, "sigma_s0_sparse", sigma_s0_csc)
        object.__setattr__(self, "sigma_s1_sparse", sigma_s1_csc)

        sigma_a = sigma_t - np.asarray(sigma_s0_csc.sum(axis=1)).ravel()
        if np.any(sigma_a <= 0.0):
            raise ValueError("P0 absorption sigma_t - sum_g' sigma_s0[g,g'] must be positive")

    @property
    def G(self) -> int:
        return int(self.sigma_t.shape[0])

    @property
    def sigma_a(self) -> np.ndarray:
        return self.sigma_t - np.asarray(self.sigma_s0_sparse.sum(axis=1)).ravel()

    @property
    def D(self) -> np.ndarray:
        sigma_tr = self.sigma_t - np.diag(self.sigma_s1)
        if np.any(sigma_tr <= 0.0):
            raise ValueError("transport cross section sigma_t - diag(sigma_s1) must be positive")
        return 1.0 / (3.0 * sigma_tr)


def make_single_group_material(sigma_t: float = 1.0, c: float = 0.5) -> P1Material:
    if sigma_t <= 0.0 or not math.isfinite(sigma_t):
        raise ValueError(f"sigma_t must be positive and finite, got {sigma_t}")
    if c < 0.0 or c >= 1.0 or not math.isfinite(c):
        raise ValueError(f"scattering ratio c must be in [0, 1), got {c}")
    return P1Material(
        sigma_t=np.array([sigma_t], dtype=np.float64),
        sigma_s0=np.array([[c * sigma_t]], dtype=np.float64),
        sigma_s1=np.zeros((1, 1), dtype=np.float64),
    )


def make_3group_p1_material() -> P1Material:
    sigma_t = np.array([1.15, 0.85, 0.55], dtype=np.float64)
    sigma_s0 = np.array(
        [
            [0.45, 0.25, 0.08],
            [0.02, 0.38, 0.18],
            [0.00, 0.03, 0.30],
        ],
        dtype=np.float64,
    )
    sigma_s1 = np.array(
        [
            [0.12, 0.04, 0.01],
            [0.00, 0.08, 0.03],
            [0.00, 0.00, 0.05],
        ],
        dtype=np.float64,
    )
    return P1Material(sigma_t=sigma_t, sigma_s0=sigma_s0, sigma_s1=sigma_s1)


def make_point_source(
    mesh: Mesh,
    G: int,
    group: int = 0,
    strength: float = 1.0,
) -> np.ndarray:
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if group < 0 or group >= G:
        raise ValueError(f"group must satisfy 0 <= group < G, got {group}")
    if strength <= 0.0 or not math.isfinite(strength):
        raise ValueError(f"strength must be positive and finite, got {strength}")

    if hasattr(mesh, "N_cells"):
        Q = np.zeros((mesh.N_cells, G), dtype=np.float64)
        c = int(np.argmin(np.linalg.norm(mesh.cell_centroid - mesh.cell_centroid.mean(axis=0), axis=1)))
        Q[c, group] = strength / float(mesh.cell_volume[c])
        return Q

    Q = np.zeros((mesh.nx, mesh.ny, mesh.nz, G), dtype=np.float64)
    vol = mesh.dx * mesh.dy * mesh.dz
    Q[mesh.nx // 2, mesh.ny // 2, mesh.nz // 2, group] = strength / vol
    return Q


def make_uniform_source(mesh: Mesh, G: int, strength: float = 1.0) -> np.ndarray:
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if strength <= 0.0 or not math.isfinite(strength):
        raise ValueError(f"strength must be positive and finite, got {strength}")
    if hasattr(mesh, "N_cells"):
        return np.full((mesh.N_cells, G), strength, dtype=np.float64)
    return np.full((mesh.nx, mesh.ny, mesh.nz, G), strength, dtype=np.float64)


def _spatial_source_shape_structured(
    mesh: Mesh,
    geometry: str,
    plasma_fraction: float = 0.25,
    gaussian_sigma_cm: float | None = None,
) -> np.ndarray:
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    shape = np.zeros((nx, ny, nz), dtype=np.float64)
    if geometry == "point":
        shape[nx // 2, ny // 2, nz // 2] = 1.0
    elif geometry == "volumetric":
        half = max(1, int(round(nx * plasma_fraction / 2)))
        ci, cj, ck = nx // 2, ny // 2, nz // 2
        shape[
            max(0, ci - half): min(nx, ci + half),
            max(0, cj - half): min(ny, cj + half),
            max(0, ck - half): min(nz, ck + half),
        ] = 1.0
    elif geometry == "gaussian":
        if gaussian_sigma_cm is None:
            gaussian_sigma_cm = 0.15 * nx * mesh.dx
        if gaussian_sigma_cm <= 0.0 or not math.isfinite(gaussian_sigma_cm):
            raise ValueError(f"gaussian_sigma_cm must be positive and finite, got {gaussian_sigma_cm}")
        cx = (nx / 2.0) * mesh.dx
        cy = (ny / 2.0) * mesh.dy
        cz = (nz / 2.0) * mesh.dz
        x = (np.arange(nx) + 0.5) * mesh.dx - cx
        y = (np.arange(ny) + 0.5) * mesh.dy - cy
        z = (np.arange(nz) + 0.5) * mesh.dz - cz
        r2 = x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2
        shape = np.exp(-r2 / (2.0 * gaussian_sigma_cm ** 2))
    else:
        raise ValueError(f"unknown source geometry {geometry!r}")
    if shape.sum() <= 0.0:
        raise ValueError(f"source geometry {geometry!r} produced zero support")
    return shape


def _spatial_source_shape_unstructured(
    mesh,
    geometry: str,
    plasma_fraction: float = 0.25,
    gaussian_sigma_cm: float | None = None,
) -> np.ndarray:
    centroids = np.asarray(mesh.cell_centroid, dtype=np.float64)
    volumes = np.asarray(mesh.cell_volume, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"mesh.cell_centroid must have shape (N_cells, 3), got {centroids.shape}")
    if volumes.shape != (centroids.shape[0],):
        raise ValueError(f"mesh.cell_volume must have shape ({centroids.shape[0]},), got {volumes.shape}")
    if np.any(volumes <= 0.0) or not np.all(np.isfinite(volumes)):
        raise ValueError("mesh.cell_volume entries must be positive and finite")

    N = centroids.shape[0]
    shape = np.zeros(N, dtype=np.float64)
    center = centroids.mean(axis=0)
    r = np.linalg.norm(centroids - center[None, :], axis=1)
    if geometry == "point":
        shape[int(np.argmin(r))] = 1.0
    elif geometry == "volumetric":
        frac = float(plasma_fraction)
        if frac <= 0.0 or not math.isfinite(frac):
            raise ValueError(f"plasma_fraction must be positive and finite, got {plasma_fraction}")
        threshold = np.quantile(r, min(max(frac, 1.0 / max(N, 1)), 1.0))
        shape[r <= threshold] = 1.0
    elif geometry == "gaussian":
        if gaussian_sigma_cm is None:
            extent = np.max(np.linalg.norm(centroids - center[None, :], axis=1))
            gaussian_sigma_cm = max(0.15 * extent, 1.0e-12)
        if gaussian_sigma_cm <= 0.0 or not math.isfinite(gaussian_sigma_cm):
            raise ValueError(f"gaussian_sigma_cm must be positive and finite, got {gaussian_sigma_cm}")
        shape = np.exp(-(r ** 2) / (2.0 * gaussian_sigma_cm ** 2))
    else:
        raise ValueError(f"unknown source geometry {geometry!r}")
    if shape.sum() <= 0.0:
        raise ValueError(f"source geometry {geometry!r} produced zero support")
    return shape


def make_spectrum_source(
    mesh,
    spectrum: np.ndarray,
    strength: float = 1.0,
    geometry: str = "point",
    plasma_fraction: float = 0.25,
    gaussian_sigma_cm: float | None = None,
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise ValueError(f"spectrum must have shape (G,), got {spectrum.shape}")
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("spectrum must be finite and nonnegative")
    total = float(spectrum.sum())
    if total <= 0.0:
        raise ValueError("spectrum must have positive sum")
    if strength <= 0.0 or not math.isfinite(strength):
        raise ValueError(f"strength must be positive and finite, got {strength}")

    spectrum = spectrum / total
    if hasattr(mesh, "N_cells"):
        shape = _spatial_source_shape_unstructured(mesh, geometry, plasma_fraction, gaussian_sigma_cm)
        volumes = np.asarray(mesh.cell_volume, dtype=np.float64)
        spatial_density = shape / max(float(np.sum(shape * volumes)), 1.0e-300)
        return strength * spatial_density[:, np.newaxis] * spectrum[np.newaxis, :]

    shape = _spatial_source_shape_structured(mesh, geometry, plasma_fraction, gaussian_sigma_cm)
    vol = mesh.dx * mesh.dy * mesh.dz
    spatial_density = shape / max(float(shape.sum() * vol), 1.0e-300)
    return strength * spatial_density[:, :, :, np.newaxis] * spectrum[np.newaxis, np.newaxis, np.newaxis, :]


def dt_source_spectrum(energy_bounds: np.ndarray, neutron_energy_ev: float = 14.1e6) -> np.ndarray:
    energy_bounds = np.asarray(energy_bounds, dtype=np.float64)
    if energy_bounds.ndim != 1 or energy_bounds.size < 2:
        raise ValueError(f"energy_bounds must have shape (G+1,), got {energy_bounds.shape}")
    if not np.all(np.isfinite(energy_bounds)):
        raise ValueError("energy_bounds contains non-finite values")
    G = energy_bounds.size - 1
    spectrum = np.zeros(G, dtype=np.float64)
    for g in range(G):
        lo = min(energy_bounds[g], energy_bounds[g + 1])
        hi = max(energy_bounds[g], energy_bounds[g + 1])
        if lo <= neutron_energy_ev <= hi:
            spectrum[g] = 1.0
            return spectrum
    raise ValueError(f"{neutron_energy_ev} eV is outside the energy group bounds")


def build_quadrature(sn: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 3D level-symmetric S4 or S8 ordinates and weights."""
    if sn == 4:
        a = math.sqrt((5.0 - math.sqrt(10.0)) / 15.0)
        b = math.sqrt((5.0 + 2.0 * math.sqrt(10.0)) / 15.0)
        octant = [
            (p, 1.0 / 3.0)
            for p in sorted(set(itertools.permutations((a, a, b))))
        ]
    elif sn == 8:
        levels = (
            math.sqrt(1.0 / 21.0),
            math.sqrt(1.0 / 3.0),
            math.sqrt(13.0 / 21.0),
            math.sqrt(19.0 / 21.0),
        )
        w114 = 0.12098765432098765
        w123 = 0.09074074074074075
        w222 = 0.09259259259259259
        octant = []
        for p in sorted(set(itertools.permutations((levels[0], levels[0], levels[3])))):
            octant.append((p, w114))
        for p in sorted(set(itertools.permutations((levels[0], levels[1], levels[2])))):
            octant.append((p, w123))
        octant.append(((levels[1], levels[1], levels[1]), w222))
    else:
        raise ValueError("build_quadrature supports only S4 and S8")

    directions = []
    weights = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        for direction, weight in octant:
            directions.append(tuple(s * d for s, d in zip(signs, direction)))
            weights.append(weight)

    directions = np.asarray(directions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights *= (4.0 * math.pi) / weights.sum()

    expected = sn * (sn + 2)
    if directions.shape != (expected, 3):
        raise RuntimeError(f"S{sn} quadrature has {directions.shape[0]} directions, expected {expected}")
    if weights.shape != (expected,):
        raise RuntimeError(f"S{sn} quadrature has invalid weight shape {weights.shape}")
    if not np.allclose(np.sum(directions * directions, axis=1), 1.0, rtol=0.0, atol=1.0e-14):
        raise RuntimeError(f"S{sn} quadrature contains non-normalized directions")
    if not np.isclose(weights.sum(), 4.0 * math.pi, rtol=0.0, atol=1.0e-14):
        raise RuntimeError(f"S{sn} quadrature weights do not sum to 4*pi")

    direction_set = {tuple(row) for row in directions}
    for row in directions:
        if tuple(-row) not in direction_set:
            raise RuntimeError(f"S{sn} quadrature is missing an opposite direction")

    return directions, weights


def _validate_moment_inputs(
    psi_ang: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi_ang = np.asarray(psi_ang)
    directions = np.asarray(directions)
    weights = np.asarray(weights)

    if psi_ang.ndim not in (3, 5):
        raise ValueError(f"psi_ang must have shape (..., n_dir, G), got {psi_ang.shape}")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape (n_dir, 3), got {directions.shape}")
    if weights.ndim != 1:
        raise ValueError(f"weights must have shape (n_dir,), got {weights.shape}")

    n_dir = psi_ang.shape[-2]
    if directions.shape[0] != n_dir:
        raise ValueError(
            f"directions has {directions.shape[0]} rows but psi_ang has n_dir={n_dir}"
        )
    if weights.shape[0] != n_dir:
        raise ValueError(f"weights has length {weights.shape[0]} but psi_ang has n_dir={n_dir}")

    if not np.all(np.isfinite(psi_ang)):
        raise ValueError("psi_ang contains non-finite values")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions contains non-finite values")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values")

    return psi_ang, directions, weights


def integrate_J(
    psi_ang: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Integrate angular flux into current J with shape (..., G, 3)."""
    psi_ang, directions, weights = _validate_moment_inputs(psi_ang, directions, weights)
    dtype = np.result_type(psi_ang.dtype, directions.dtype, weights.dtype)
    weighted_dirs = weights.astype(dtype, copy=False)[:, np.newaxis] * directions.astype(dtype, copy=False)
    J = np.einsum("...mg,mc->...gc", psi_ang.astype(dtype, copy=False), weighted_dirs)
    if not np.all(np.isfinite(J)):
        raise FloatingPointError("integrate_J produced non-finite values")
    return J


def integrate_moments(
    psi_ang: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate angular flux into scalar flux phi and current J."""
    psi_ang, directions, weights = _validate_moment_inputs(psi_ang, directions, weights)
    dtype = np.result_type(psi_ang.dtype, directions.dtype, weights.dtype)
    psi = psi_ang.astype(dtype, copy=False)
    w = weights.astype(dtype, copy=False)
    phi = np.tensordot(psi, w, axes=([-2], [0]))
    J = integrate_J(psi, directions.astype(dtype, copy=False), w)
    if not np.all(np.isfinite(phi)):
        raise FloatingPointError("integrate_moments produced non-finite scalar flux")
    return phi, J


def build_reflection_map(directions: np.ndarray) -> dict[str, np.ndarray]:
    """Build reflected-direction index maps for Cartesian boundary faces."""
    directions = np.asarray(directions)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape (n_dir, 3), got {directions.shape}")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions contains non-finite values")

    faces = {
        "xmin": np.array([-1.0, 1.0, 1.0]),
        "xmax": np.array([-1.0, 1.0, 1.0]),
        "ymin": np.array([1.0, -1.0, 1.0]),
        "ymax": np.array([1.0, -1.0, 1.0]),
        "zmin": np.array([1.0, 1.0, -1.0]),
        "zmax": np.array([1.0, 1.0, -1.0]),
    }

    refl_map: dict[str, np.ndarray] = {}
    for face, signs in faces.items():
        mapping = np.empty(directions.shape[0], dtype=np.int64)
        for m, direction in enumerate(directions):
            target = direction * signs
            matches = np.nonzero(np.all(np.isclose(directions, target, rtol=0.0, atol=1.0e-12), axis=1))[0]
            if len(matches) != 1:
                raise ValueError(
                    f"could not find unique reflected direction for face {face}, index {m}: {target}"
                )
            mapping[m] = int(matches[0])
        refl_map[face] = mapping

    return refl_map


# Re-export unstructured mesh support for callers that import from sn_core.
from mesh_builder import UnstructuredMesh, MeshBuilder
