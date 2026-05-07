from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from sn_core import P1Material


@dataclass(frozen=True)
class MaterialXS:
    name: str
    sigma_t: np.ndarray
    sigma_s0: np.ndarray
    sigma_s1: np.ndarray
    reactions: dict[str, np.ndarray] = field(default_factory=dict)
    heating: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sigma_t = np.asarray(self.sigma_t, dtype=np.float64)
        sigma_s0 = np.asarray(self.sigma_s0, dtype=np.float64)
        sigma_s1 = np.asarray(self.sigma_s1, dtype=np.float64)
        if sigma_t.ndim != 1:
            raise ValueError(f"{self.name}.sigma_t must have shape (G,), got {sigma_t.shape}")
        G = sigma_t.shape[0]
        if sigma_s0.shape != (G, G):
            raise ValueError(f"{self.name}.sigma_s0 must have shape {(G, G)}, got {sigma_s0.shape}")
        if sigma_s1.shape != (G, G):
            raise ValueError(f"{self.name}.sigma_s1 must have shape {(G, G)}, got {sigma_s1.shape}")
        for label, arr in (("sigma_t", sigma_t), ("sigma_s0", sigma_s0), ("sigma_s1", sigma_s1)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.name}.{label} contains non-finite values")
            if np.any(arr < 0.0):
                raise ValueError(f"{self.name}.{label} contains negative values")
        if np.any(sigma_t <= 0.0):
            raise ValueError(f"{self.name}.sigma_t entries must be positive")
        sigma_a = sigma_t - sigma_s0.sum(axis=1)
        if np.any(sigma_a < -1.0e-12):
            raise ValueError(f"{self.name} has negative absorption sigma_t - sum(sigma_s0)")

        reactions = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in dict(self.reactions).items()
        }
        for key, arr in reactions.items():
            if arr.shape != (G,):
                raise ValueError(f"{self.name}.reactions[{key!r}] must have shape {(G,)}, got {arr.shape}")
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
                raise ValueError(f"{self.name}.reactions[{key!r}] must be finite and nonnegative")

        heating = None if self.heating is None else np.asarray(self.heating, dtype=np.float64)
        if heating is not None:
            if heating.shape != (G,):
                raise ValueError(f"{self.name}.heating must have shape {(G,)}, got {heating.shape}")
            if not np.all(np.isfinite(heating)) or np.any(heating < 0.0):
                raise ValueError(f"{self.name}.heating must be finite and nonnegative")

        object.__setattr__(self, "sigma_t", sigma_t)
        object.__setattr__(self, "sigma_s0", sigma_s0)
        object.__setattr__(self, "sigma_s1", sigma_s1)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "heating", heating)

    @property
    def G(self) -> int:
        return int(self.sigma_t.shape[0])

    @property
    def sigma_a(self) -> np.ndarray:
        return self.sigma_t - self.sigma_s0.sum(axis=1)

    def to_p1_material(self) -> P1Material:
        return P1Material(self.sigma_t, self.sigma_s0, self.sigma_s1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sigma_t": self.sigma_t.tolist(),
            "sigma_s0": self.sigma_s0.tolist(),
            "sigma_s1": self.sigma_s1.tolist(),
            "reactions": {key: value.tolist() for key, value in self.reactions.items()},
            "heating": None if self.heating is None else self.heating.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MaterialXS":
        return cls(
            name=data["name"],
            sigma_t=np.asarray(data["sigma_t"], dtype=np.float64),
            sigma_s0=np.asarray(data["sigma_s0"], dtype=np.float64),
            sigma_s1=np.asarray(data["sigma_s1"], dtype=np.float64),
            reactions={
                key: np.asarray(value, dtype=np.float64)
                for key, value in data.get("reactions", {}).items()
            },
            heating=None if data.get("heating") is None else np.asarray(data["heating"], dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MultigroupLibrary:
    energy_bounds: np.ndarray
    materials: dict[str, MaterialXS]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energy_bounds = np.asarray(self.energy_bounds, dtype=np.float64)
        if energy_bounds.ndim != 1 or energy_bounds.size < 2:
            raise ValueError(f"energy_bounds must have shape (G+1,), got {energy_bounds.shape}")
        if not np.all(np.isfinite(energy_bounds)):
            raise ValueError("energy_bounds contains non-finite values")
        diffs = np.diff(energy_bounds)
        if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
            raise ValueError("energy_bounds must be strictly monotonic")
        G = energy_bounds.size - 1
        materials = dict(self.materials)
        if not materials:
            raise ValueError("MultigroupLibrary requires at least one material")
        for key, mat in materials.items():
            if mat.G != G:
                raise ValueError(f"material {key!r} has G={mat.G}, expected {G}")
        object.__setattr__(self, "energy_bounds", energy_bounds)
        object.__setattr__(self, "materials", materials)

    @property
    def G(self) -> int:
        return int(self.energy_bounds.size - 1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "energy_bounds": self.energy_bounds.tolist(),
            "materials": {key: mat.to_json_dict() for key, mat in self.materials.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MultigroupLibrary":
        return cls(
            energy_bounds=np.asarray(data["energy_bounds"], dtype=np.float64),
            materials={
                key: MaterialXS.from_json_dict(value)
                for key, value in data["materials"].items()
            },
            metadata=dict(data.get("metadata", {})),
        )


def save_multigroup_library(library: MultigroupLibrary, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(json.dumps(library.to_json_dict(), indent=2), encoding="utf-8")
        return
    if path.suffix == ".npz":
        arrays: dict[str, Any] = {
            "energy_bounds": library.energy_bounds,
            "metadata_json": json.dumps(library.metadata),
            "material_keys_json": json.dumps(list(library.materials)),
        }
        for key, mat in library.materials.items():
            prefix = f"material/{key}/"
            arrays[prefix + "name"] = np.asarray(mat.name)
            arrays[prefix + "sigma_t"] = mat.sigma_t
            arrays[prefix + "sigma_s0"] = mat.sigma_s0
            arrays[prefix + "sigma_s1"] = mat.sigma_s1
            arrays[prefix + "heating"] = np.asarray([] if mat.heating is None else mat.heating)
            arrays[prefix + "metadata_json"] = json.dumps(mat.metadata)
            arrays[prefix + "reaction_keys_json"] = json.dumps(list(mat.reactions))
            for reaction_key, values in mat.reactions.items():
                arrays[prefix + f"reaction/{reaction_key}"] = values
        np.savez(path, **arrays)
        return
    raise ValueError(f"unsupported library format {path.suffix!r}; use .json or .npz")


def load_multigroup_library(path: str | Path) -> MultigroupLibrary:
    path = Path(path)
    if path.suffix == ".json":
        return MultigroupLibrary.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            material_keys = json.loads(str(data["material_keys_json"]))
            materials = {}
            for key in material_keys:
                prefix = f"material/{key}/"
                reaction_keys = json.loads(str(data[prefix + "reaction_keys_json"]))
                reactions = {
                    reaction_key: data[prefix + f"reaction/{reaction_key}"]
                    for reaction_key in reaction_keys
                }
                heating_arr = data[prefix + "heating"]
                materials[key] = MaterialXS(
                    name=str(data[prefix + "name"]),
                    sigma_t=data[prefix + "sigma_t"],
                    sigma_s0=data[prefix + "sigma_s0"],
                    sigma_s1=data[prefix + "sigma_s1"],
                    reactions=reactions,
                    heating=None if heating_arr.size == 0 else heating_arr,
                    metadata=json.loads(str(data[prefix + "metadata_json"])),
                )
            return MultigroupLibrary(
                energy_bounds=data["energy_bounds"],
                materials=materials,
                metadata=json.loads(str(data["metadata_json"])),
            )
    raise ValueError(f"unsupported library format {path.suffix!r}; use .json or .npz")


def make_synthetic_library(G: int, name: str | None = None) -> MultigroupLibrary:
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    energy_bounds = np.geomspace(2.0e7, 1.0e-5, G + 1)
    g = np.arange(G, dtype=np.float64)
    sigma_t = 0.65 + 0.35 * (g / max(G - 1, 1))
    sigma_s0 = np.zeros((G, G), dtype=np.float64)
    for src in range(G):
        sigma_s0[src, src] = 0.35 * sigma_t[src]
        if src + 1 < G:
            sigma_s0[src, src + 1] = 0.18 * sigma_t[src]
        if src + 2 < G:
            sigma_s0[src, src + 2] = 0.07 * sigma_t[src]
        if src > 0:
            sigma_s0[src, src - 1] = 0.015 * sigma_t[src]
    row_scale = np.minimum(0.82 * sigma_t / np.maximum(sigma_s0.sum(axis=1), 1.0e-300), 1.0)
    sigma_s0 *= row_scale[:, np.newaxis]
    sigma_s1 = 0.12 * np.diag(sigma_s0.diagonal())
    reactions = {
        "absorption": sigma_t - sigma_s0.sum(axis=1),
        "damage": 0.02 * np.exp(-g / max(G, 1)),
    }
    heating = 0.5 + 6.0 * np.exp(-g / max(G / 3.0, 1.0))
    material_name = name or f"synthetic_{G}g"
    return MultigroupLibrary(
        energy_bounds=energy_bounds,
        materials={
            material_name: MaterialXS(
                name=material_name,
                sigma_t=sigma_t,
                sigma_s0=sigma_s0,
                sigma_s1=sigma_s1,
                reactions=reactions,
                heating=heating,
                metadata={"synthetic": True},
            )
        },
        metadata={"description": f"Synthetic {G}-group downscatter-dominant library"},
    )


def estimate_memory_bytes(
    nx: int,
    ny: int,
    nz: int,
    n_dir: int,
    G: int,
    dtype_bytes: int = 8,
) -> dict[str, int]:
    cells = nx * ny * nz
    return {
        "angular_flux": cells * n_dir * G * dtype_bytes,
        "scattering_source_dense": cells * n_dir * G * dtype_bytes,
        "scattering_source_blocked": cells * dtype_bytes,
        "scalar_flux": cells * G * dtype_bytes,
        "current": cells * G * 3 * dtype_bytes,
    }


def format_memory_report(nx: int, ny: int, nz: int, n_dirs: tuple[int, ...] = (24, 80), groups: tuple[int, ...] = (10, 27, 70, 175)) -> str:
    lines = [f"Memory estimate for mesh {nx}x{ny}x{nz} (float64):"]
    for n_dir in n_dirs:
        for G in groups:
            est = estimate_memory_bytes(nx, ny, nz, n_dir, G)
            dense_total = est["angular_flux"] + est["scattering_source_dense"] + est["scalar_flux"] + est["current"]
            blocked_total = est["angular_flux"] + est["scattering_source_blocked"] + est["scalar_flux"] + est["current"]
            lines.append(
                f"  n_dir={n_dir:3d}, G={G:3d}: "
                f"psi={est['angular_flux']/2**20:8.1f} MiB, "
                f"Qscat_dense={est['scattering_source_dense']/2**20:8.1f} MiB, "
                f"Qscat_block={est['scattering_source_blocked']/2**20:6.1f} MiB, "
                f"moments={(est['scalar_flux'] + est['current'])/2**20:8.1f} MiB, "
                f"dense_total={dense_total/2**20:8.1f} MiB, "
                f"blocked_total={blocked_total/2**20:8.1f} MiB"
            )
    return "\n".join(lines)
