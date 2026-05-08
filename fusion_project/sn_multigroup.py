from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy import sparse

from sn_core import P1Material


def _as_scattering_csc(values: object, G: int, label: str) -> sparse.csc_matrix:
    if sparse.issparse(values):
        matrix = values.astype(np.float64).tocsc()
    else:
        matrix = sparse.csc_matrix(np.asarray(values, dtype=np.float64))
    if matrix.shape != (G, G):
        raise ValueError(f"{label} must have shape {(G, G)}, got {matrix.shape}")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def _sparse_triplet_dict(matrix: sparse.spmatrix) -> dict[str, Any]:
    coo = matrix.tocoo()
    return {
        "format": "coo_triplet_v1",
        "shape": list(coo.shape),
        "row": coo.row.astype(np.int64).tolist(),
        "col": coo.col.astype(np.int64).tolist(),
        "data": coo.data.astype(np.float64).tolist(),
    }


def _sparse_from_triplet_dict(data: dict[str, Any], G: int, label: str) -> sparse.csc_matrix:
    if data.get("format") != "coo_triplet_v1":
        raise ValueError(f"{label} sparse format must be 'coo_triplet_v1'")
    shape = tuple(int(v) for v in data["shape"])
    if shape != (G, G):
        raise ValueError(f"{label} sparse shape must be {(G, G)}, got {shape}")
    row = np.asarray(data["row"], dtype=np.int64)
    col = np.asarray(data["col"], dtype=np.int64)
    values = np.asarray(data["data"], dtype=np.float64)
    if row.shape != col.shape or row.shape != values.shape:
        raise ValueError(f"{label} sparse row/col/data lengths differ")
    return _as_scattering_csc(sparse.coo_matrix((values, (row, col)), shape=shape), G, label)


@dataclass(frozen=True)
class MaterialXS:
    name: str
    sigma_t: np.ndarray
    sigma_s0: np.ndarray
    sigma_s1: np.ndarray
    reactions: dict[str, np.ndarray] = field(default_factory=dict)
    heating: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chi: np.ndarray | None = None
    nu_sigma_f: np.ndarray | None = None

    def __post_init__(self) -> None:
        sigma_t = np.asarray(self.sigma_t, dtype=np.float64)
        if sigma_t.ndim != 1:
            raise ValueError(f"{self.name}.sigma_t must have shape (G,), got {sigma_t.shape}")
        G = sigma_t.shape[0]
        sigma_s0_csc = _as_scattering_csc(self.sigma_s0, G, f"{self.name}.sigma_s0")
        sigma_s1_csc = _as_scattering_csc(self.sigma_s1, G, f"{self.name}.sigma_s1")
        sigma_s0 = sigma_s0_csc.toarray()
        sigma_s1 = sigma_s1_csc.toarray()
        if np.any(sigma_s0_csc.data < 0.0):
            raise ValueError(f"{self.name}.sigma_s0 contains negative values")
        if np.any(sigma_s1_csc.data < 0.0):
            raise ValueError(f"{self.name}.sigma_s1 contains negative values")
        if not np.all(np.isfinite(sigma_t)):
            raise ValueError(f"{self.name}.sigma_t contains non-finite values")
        if np.any(sigma_t <= 0.0):
            raise ValueError(f"{self.name}.sigma_t entries must be positive")
        sigma_a = sigma_t - np.asarray(sigma_s0_csc.sum(axis=1)).ravel()
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

        chi = None if self.chi is None else np.asarray(self.chi, dtype=np.float64)
        if chi is not None:
            if chi.shape != (G,):
                raise ValueError(f"{self.name}.chi must have shape {(G,)}, got {chi.shape}")
            if not np.all(np.isfinite(chi)) or np.any(chi < 0.0):
                raise ValueError(f"{self.name}.chi must be finite and nonnegative")
            chi_sum = float(chi.sum())
            if not np.isclose(chi_sum, 1.0):
                raise ValueError(f"{self.name}.chi must be normalized with sum close to 1.0, got {chi_sum}")

        nu_sigma_f = None if self.nu_sigma_f is None else np.asarray(self.nu_sigma_f, dtype=np.float64)
        if nu_sigma_f is not None:
            if nu_sigma_f.shape != (G,):
                raise ValueError(f"{self.name}.nu_sigma_f must have shape {(G,)}, got {nu_sigma_f.shape}")
            if not np.all(np.isfinite(nu_sigma_f)) or np.any(nu_sigma_f < 0.0):
                raise ValueError(f"{self.name}.nu_sigma_f must be finite and nonnegative")

        object.__setattr__(self, "sigma_t", sigma_t)
        object.__setattr__(self, "sigma_s0", sigma_s0)
        object.__setattr__(self, "sigma_s1", sigma_s1)
        object.__setattr__(self, "sigma_s0_sparse", sigma_s0_csc)
        object.__setattr__(self, "sigma_s1_sparse", sigma_s1_csc)
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "heating", heating)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "nu_sigma_f", nu_sigma_f)

    @property
    def G(self) -> int:
        return int(self.sigma_t.shape[0])

    @property
    def sigma_a(self) -> np.ndarray:
        return self.sigma_t - np.asarray(self.sigma_s0_sparse.sum(axis=1)).ravel()

    def to_p1_material(self) -> P1Material:
        return P1Material(self.sigma_t, self.sigma_s0_sparse, self.sigma_s1_sparse)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sigma_t": self.sigma_t.tolist(),
            "sigma_s0": self.sigma_s0.tolist(),
            "sigma_s1": self.sigma_s1.tolist(),
            "sigma_s0_sparse": _sparse_triplet_dict(self.sigma_s0_sparse),
            "sigma_s1_sparse": _sparse_triplet_dict(self.sigma_s1_sparse),
            "reactions": {key: value.tolist() for key, value in self.reactions.items()},
            "heating": None if self.heating is None else self.heating.tolist(),
            "chi": None if self.chi is None else self.chi.tolist(),
            "nu_sigma_f": None if self.nu_sigma_f is None else self.nu_sigma_f.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MaterialXS":
        return cls(
            name=data["name"],
            sigma_t=np.asarray(data["sigma_t"], dtype=np.float64),
            sigma_s0=(
                _sparse_from_triplet_dict(data["sigma_s0_sparse"], np.asarray(data["sigma_t"], dtype=np.float64).shape[0], "sigma_s0")
                if "sigma_s0_sparse" in data else np.asarray(data["sigma_s0"], dtype=np.float64)
            ),
            sigma_s1=(
                _sparse_from_triplet_dict(data["sigma_s1_sparse"], np.asarray(data["sigma_t"], dtype=np.float64).shape[0], "sigma_s1")
                if "sigma_s1_sparse" in data else np.asarray(data["sigma_s1"], dtype=np.float64)
            ),
            reactions={
                key: np.asarray(value, dtype=np.float64)
                for key, value in data.get("reactions", {}).items()
            },
            heating=None if data.get("heating") is None else np.asarray(data["heating"], dtype=np.float64),
            chi=None if data.get("chi") is None else np.asarray(data["chi"], dtype=np.float64),
            nu_sigma_f=None if data.get("nu_sigma_f") is None else np.asarray(data["nu_sigma_f"], dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MultigroupLibrary:
    energy_bounds: np.ndarray
    materials: dict[str, MaterialXS]
    group_names: tuple[str, ...] | None = None
    lethargy_widths: np.ndarray | None = None
    source_group_mapping: dict[str, Any] = field(default_factory=dict)
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
        group_names = None if self.group_names is None else tuple(str(name) for name in self.group_names)
        if group_names is not None and len(group_names) != G:
            raise ValueError(f"group_names must have length {G}, got {len(group_names)}")
        lethargy_widths = None if self.lethargy_widths is None else np.asarray(self.lethargy_widths, dtype=np.float64)
        if lethargy_widths is None:
            lethargy_widths = np.abs(np.log(energy_bounds[:-1] / energy_bounds[1:]))
        if lethargy_widths.shape != (G,):
            raise ValueError(f"lethargy_widths must have shape {(G,)}, got {lethargy_widths.shape}")
        if not np.all(np.isfinite(lethargy_widths)):
            raise ValueError("lethargy_widths contains non-finite values")
        source_group_mapping = dict(self.source_group_mapping)
        for source_name, value in source_group_mapping.items():
            if isinstance(value, dict):
                if "group" not in value:
                    raise ValueError(f"source_group_mapping[{source_name!r}] dict entries must contain a 'group' key")
                group_index = int(value["group"])
            else:
                group_index = int(value)
            if not (0 <= group_index < G):
                raise ValueError(f"source_group_mapping[{source_name!r}] group index {group_index} out of range for G={G}")
        object.__setattr__(self, "energy_bounds", energy_bounds)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "group_names", group_names)
        object.__setattr__(self, "lethargy_widths", lethargy_widths)
        object.__setattr__(self, "source_group_mapping", source_group_mapping)

    @property
    def G(self) -> int:
        return int(self.energy_bounds.size - 1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "energy_bounds": self.energy_bounds.tolist(),
            "materials": {key: mat.to_json_dict() for key, mat in self.materials.items()},
            "group_names": None if self.group_names is None else list(self.group_names),
            "lethargy_widths": self.lethargy_widths.tolist(),
            "source_group_mapping": self.source_group_mapping,
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
            group_names=None if data.get("group_names") is None else tuple(data["group_names"]),
            lethargy_widths=(
                None if data.get("lethargy_widths") is None
                else np.asarray(data["lethargy_widths"], dtype=np.float64)
            ),
            source_group_mapping=dict(data.get("source_group_mapping", {})),
            metadata=dict(data.get("metadata", {})),
        )


def source_spectrum_for_named_source(
    library: MultigroupLibrary,
    source_name: str,
    default_energy_ev: float = 14.1e6,
) -> np.ndarray:
    """Return a normalized one-hot group spectrum for a named source.

    Resolution order:
    1. Use ``library.source_group_mapping`` entry when present.
    2. For D-T aliases, fall back to ``sn_core.dt_source_spectrum`` energy scan.
    """
    source_key = str(source_name)
    mapping = library.source_group_mapping.get(source_key)
    if mapping is not None:
        group_index = int(mapping["group"]) if isinstance(mapping, dict) else int(mapping)
        spectrum = np.zeros(library.G, dtype=np.float64)
        spectrum[group_index] = 1.0
        return spectrum
    if source_key.upper() in {"DT_14MEV", "DT", "D-T"}:
        from sn_core import dt_source_spectrum

        return dt_source_spectrum(library.energy_bounds, neutron_energy_ev=default_energy_ev)
    raise ValueError(f"unknown source {source_name!r} and no source_group_mapping entry present")


def import_processed_fusion_xs_json(path: str | Path) -> MultigroupLibrary:
    """Load a constrained processed-fusion-XS JSON schema into MultigroupLibrary.

    This is a schema-integration importer with strict validation. It does not
    imply external benchmark or experimental validation of the source data.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "energy_bounds" not in payload or "materials" not in payload:
        raise ValueError("processed XS JSON requires 'energy_bounds' and 'materials'")
    if not isinstance(payload["materials"], dict) or not payload["materials"]:
        raise ValueError("processed XS JSON 'materials' must be a non-empty object")

    metadata = dict(payload.get("metadata", {}))
    provenance = metadata.get("provenance")
    if provenance is None or str(provenance).strip() == "":
        warnings.warn(
            "processed XS JSON missing metadata.provenance; imported as unqualified schema payload",
            UserWarning,
            stacklevel=2,
        )
        metadata["processed_external_format"] = False
    else:
        metadata["processed_external_format"] = True

    materials: dict[str, MaterialXS] = {}
    for key, mat in payload["materials"].items():
        if "name" not in mat:
            raise ValueError(f"processed XS material {key!r} missing required field 'name'")
        for required in ("sigma_t", "sigma_s0", "sigma_s1"):
            if required not in mat:
                raise ValueError(f"processed XS material {key!r} missing required field {required!r}")
        materials[str(key)] = MaterialXS(
            name=str(mat["name"]),
            sigma_t=np.asarray(mat["sigma_t"], dtype=np.float64),
            sigma_s0=np.asarray(mat["sigma_s0"], dtype=np.float64),
            sigma_s1=np.asarray(mat["sigma_s1"], dtype=np.float64),
            reactions={rk: np.asarray(rv, dtype=np.float64) for rk, rv in dict(mat.get("reactions", {})).items()},
            heating=None if mat.get("heating") is None else np.asarray(mat["heating"], dtype=np.float64),
            chi=None if mat.get("chi") is None else np.asarray(mat["chi"], dtype=np.float64),
            nu_sigma_f=None if mat.get("nu_sigma_f") is None else np.asarray(mat["nu_sigma_f"], dtype=np.float64),
            metadata=dict(mat.get("metadata", {})),
        )

    return MultigroupLibrary(
        energy_bounds=np.asarray(payload["energy_bounds"], dtype=np.float64),
        materials=materials,
        metadata=metadata,
    )


def _require_h5py() -> Any:
    if importlib.util.find_spec("h5py") is None:
        raise ImportError(
            "HDF5 multigroup library support requires optional dependency h5py. "
            "Install it with `pip install h5py` to read or write .h5/.hdf5 libraries."
        )
    return importlib.import_module("h5py")


def _hdf5_string(value: str) -> str:
    return value


def _hdf5_scalar_string(dataset: Any) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _hdf5_optional_array(group: Any, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    values = np.asarray(group[name])
    return None if values.size == 0 else values


def _write_hdf5_sparse(group: Any, name: str, matrix: sparse.spmatrix) -> None:
    triplet = _sparse_triplet_dict(matrix)
    sparse_group = group.create_group(name)
    sparse_group.attrs["format"] = triplet["format"]
    sparse_group.create_dataset("shape", data=np.asarray(triplet["shape"], dtype=np.int64))
    sparse_group.create_dataset("row", data=np.asarray(triplet["row"], dtype=np.int64))
    sparse_group.create_dataset("col", data=np.asarray(triplet["col"], dtype=np.int64))
    sparse_group.create_dataset("data", data=np.asarray(triplet["data"], dtype=np.float64))


def _read_hdf5_sparse(group: Any, name: str, G: int) -> sparse.csc_matrix:
    sparse_group = group[name]
    triplet = {
        "format": str(sparse_group.attrs["format"]),
        "shape": np.asarray(sparse_group["shape"], dtype=np.int64).tolist(),
        "row": np.asarray(sparse_group["row"], dtype=np.int64).tolist(),
        "col": np.asarray(sparse_group["col"], dtype=np.int64).tolist(),
        "data": np.asarray(sparse_group["data"], dtype=np.float64).tolist(),
    }
    return _sparse_from_triplet_dict(triplet, G, name)


def _save_hdf5_multigroup_library(library: MultigroupLibrary, path: Path) -> None:
    h5py = _require_h5py()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("energy_bounds", data=library.energy_bounds)
        if library.group_names is not None:
            h5.create_dataset("group_names_json", data=_hdf5_string(json.dumps(list(library.group_names))), dtype=string_dtype)
        h5.create_dataset("lethargy_widths", data=library.lethargy_widths)
        h5.create_dataset("source_group_mapping_json", data=_hdf5_string(json.dumps(library.source_group_mapping)), dtype=string_dtype)
        h5.create_dataset("metadata_json", data=_hdf5_string(json.dumps(library.metadata)), dtype=string_dtype)
        h5.create_dataset("material_keys_json", data=_hdf5_string(json.dumps(list(library.materials))), dtype=string_dtype)
        materials_group = h5.create_group("materials")
        for key, mat in library.materials.items():
            mat_group = materials_group.create_group(key)
            mat_group.create_dataset("name", data=_hdf5_string(mat.name), dtype=string_dtype)
            mat_group.create_dataset("sigma_t", data=mat.sigma_t)
            mat_group.create_dataset("sigma_s0", data=mat.sigma_s0)
            mat_group.create_dataset("sigma_s1", data=mat.sigma_s1)
            sparse_group = mat_group.create_group("scattering_sparse")
            _write_hdf5_sparse(sparse_group, "sigma_s0", mat.sigma_s0_sparse)
            _write_hdf5_sparse(sparse_group, "sigma_s1", mat.sigma_s1_sparse)
            if mat.heating is not None:
                mat_group.create_dataset("heating", data=mat.heating)
            if mat.chi is not None:
                mat_group.create_dataset("chi", data=mat.chi)
            if mat.nu_sigma_f is not None:
                mat_group.create_dataset("nu_sigma_f", data=mat.nu_sigma_f)
            mat_group.create_dataset("metadata_json", data=_hdf5_string(json.dumps(mat.metadata)), dtype=string_dtype)
            mat_group.create_dataset("reaction_keys_json", data=_hdf5_string(json.dumps(list(mat.reactions))), dtype=string_dtype)
            reactions_group = mat_group.create_group("reactions")
            for reaction_key, values in mat.reactions.items():
                reactions_group.create_dataset(reaction_key, data=values)


def _load_hdf5_multigroup_library(path: Path) -> MultigroupLibrary:
    h5py = _require_h5py()
    with h5py.File(path, "r") as h5:
        material_keys = json.loads(_hdf5_scalar_string(h5["material_keys_json"]))
        materials = {}
        materials_group = h5["materials"]
        for key in material_keys:
            mat_group = materials_group[key]
            if "reaction_keys_json" in mat_group:
                reaction_keys = json.loads(_hdf5_scalar_string(mat_group["reaction_keys_json"]))
            else:
                reaction_keys = sorted(mat_group.get("reactions", {}).keys())
            reactions_group = mat_group.get("reactions", {})
            reactions = {
                reaction_key: np.asarray(reactions_group[reaction_key])
                for reaction_key in reaction_keys
            }
            materials[key] = MaterialXS(
                name=_hdf5_scalar_string(mat_group["name"]),
                sigma_t=np.asarray(mat_group["sigma_t"]),
                sigma_s0=(
                    _read_hdf5_sparse(mat_group["scattering_sparse"], "sigma_s0", np.asarray(mat_group["sigma_t"]).shape[0])
                    if "scattering_sparse" in mat_group else np.asarray(mat_group["sigma_s0"])
                ),
                sigma_s1=(
                    _read_hdf5_sparse(mat_group["scattering_sparse"], "sigma_s1", np.asarray(mat_group["sigma_t"]).shape[0])
                    if "scattering_sparse" in mat_group else np.asarray(mat_group["sigma_s1"])
                ),
                reactions=reactions,
                heating=_hdf5_optional_array(mat_group, "heating"),
                chi=_hdf5_optional_array(mat_group, "chi"),
                nu_sigma_f=_hdf5_optional_array(mat_group, "nu_sigma_f"),
                metadata=json.loads(_hdf5_scalar_string(mat_group["metadata_json"])),
            )
        return MultigroupLibrary(
            energy_bounds=np.asarray(h5["energy_bounds"]),
            materials=materials,
            group_names=(
                None if "group_names_json" not in h5
                else tuple(json.loads(_hdf5_scalar_string(h5["group_names_json"])))
            ),
            lethargy_widths=(np.asarray(h5["lethargy_widths"]) if "lethargy_widths" in h5 else None),
            source_group_mapping=(
                {} if "source_group_mapping_json" not in h5
                else json.loads(_hdf5_scalar_string(h5["source_group_mapping_json"]))
            ),
            metadata=json.loads(_hdf5_scalar_string(h5["metadata_json"])),
        )


def save_multigroup_library(library: MultigroupLibrary, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(library.to_json_dict(), indent=2), encoding="utf-8")
        return
    if suffix == ".npz":
        arrays: dict[str, Any] = {
            "energy_bounds": library.energy_bounds,
            "group_names_json": json.dumps(None if library.group_names is None else list(library.group_names)),
            "lethargy_widths": library.lethargy_widths,
            "source_group_mapping_json": json.dumps(library.source_group_mapping),
            "metadata_json": json.dumps(library.metadata),
            "material_keys_json": json.dumps(list(library.materials)),
        }
        for key, mat in library.materials.items():
            prefix = f"material/{key}/"
            arrays[prefix + "name"] = np.asarray(mat.name)
            arrays[prefix + "sigma_t"] = mat.sigma_t
            arrays[prefix + "sigma_s0"] = mat.sigma_s0
            arrays[prefix + "sigma_s1"] = mat.sigma_s1
            for scatter_name, scatter_matrix in (("sigma_s0", mat.sigma_s0_sparse), ("sigma_s1", mat.sigma_s1_sparse)):
                triplet = _sparse_triplet_dict(scatter_matrix)
                arrays[prefix + f"{scatter_name}_sparse/shape"] = np.asarray(triplet["shape"], dtype=np.int64)
                arrays[prefix + f"{scatter_name}_sparse/row"] = np.asarray(triplet["row"], dtype=np.int64)
                arrays[prefix + f"{scatter_name}_sparse/col"] = np.asarray(triplet["col"], dtype=np.int64)
                arrays[prefix + f"{scatter_name}_sparse/data"] = np.asarray(triplet["data"], dtype=np.float64)
                arrays[prefix + f"{scatter_name}_sparse/format"] = np.asarray(triplet["format"])
            arrays[prefix + "heating"] = np.asarray([] if mat.heating is None else mat.heating)
            arrays[prefix + "chi"] = np.asarray([] if mat.chi is None else mat.chi)
            arrays[prefix + "nu_sigma_f"] = np.asarray([] if mat.nu_sigma_f is None else mat.nu_sigma_f)
            arrays[prefix + "metadata_json"] = json.dumps(mat.metadata)
            arrays[prefix + "reaction_keys_json"] = json.dumps(list(mat.reactions))
            for reaction_key, values in mat.reactions.items():
                arrays[prefix + f"reaction/{reaction_key}"] = values
        np.savez(path, **arrays)
        return
    if suffix in (".h5", ".hdf5"):
        _save_hdf5_multigroup_library(library, path)
        return
    raise ValueError(f"unsupported library format {path.suffix!r}; use .json, .npz, .h5, or .hdf5")


def load_multigroup_library(path: str | Path) -> MultigroupLibrary:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return MultigroupLibrary.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".npz":
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
                chi_arr = data[prefix + "chi"] if prefix + "chi" in data else None
                nu_sigma_f_arr = data[prefix + "nu_sigma_f"] if prefix + "nu_sigma_f" in data else None

                def _npz_scatter(scatter_name: str) -> object:
                    base = prefix + f"{scatter_name}_sparse/"
                    if base + "data" not in data:
                        return data[prefix + scatter_name]
                    triplet = {
                        "format": str(data[base + "format"]),
                        "shape": data[base + "shape"].astype(np.int64).tolist(),
                        "row": data[base + "row"].astype(np.int64).tolist(),
                        "col": data[base + "col"].astype(np.int64).tolist(),
                        "data": data[base + "data"].astype(np.float64).tolist(),
                    }
                    return _sparse_from_triplet_dict(triplet, data[prefix + "sigma_t"].shape[0], scatter_name)

                materials[key] = MaterialXS(
                    name=str(data[prefix + "name"]),
                    sigma_t=data[prefix + "sigma_t"],
                    sigma_s0=_npz_scatter("sigma_s0"),
                    sigma_s1=_npz_scatter("sigma_s1"),
                    reactions=reactions,
                    heating=None if heating_arr.size == 0 else heating_arr,
                    chi=None if chi_arr is None or chi_arr.size == 0 else chi_arr,
                    nu_sigma_f=None if nu_sigma_f_arr is None or nu_sigma_f_arr.size == 0 else nu_sigma_f_arr,
                    metadata=json.loads(str(data[prefix + "metadata_json"])),
                )
            return MultigroupLibrary(
                energy_bounds=data["energy_bounds"],
                materials=materials,
                group_names=(
                    None
                    if json.loads(str(data["group_names_json"])) is None
                    else tuple(json.loads(str(data["group_names_json"])))
                ) if "group_names_json" in data else None,
                lethargy_widths=(data["lethargy_widths"] if "lethargy_widths" in data else None),
                source_group_mapping=(
                    json.loads(str(data["source_group_mapping_json"]))
                    if "source_group_mapping_json" in data else {}
                ),
                metadata=json.loads(str(data["metadata_json"])),
            )
    if suffix in (".h5", ".hdf5"):
        return _load_hdf5_multigroup_library(path)
    raise ValueError(f"unsupported library format {path.suffix!r}; use .json, .npz, .h5, or .hdf5")


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


def make_sparse_synthetic_library(G: int, name: str | None = None) -> MultigroupLibrary:
    """Build a deliberately sparse upscatter/downscatter synthetic library.

    Scattering entries are stored through the sparse MaterialXS path while
    retaining dense array compatibility on the public attributes.  Each source
    group scatters to itself, the two neighboring downscatter groups, and one
    periodic upscatter group so tests exercise nontrivial column sparsity.
    """
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    energy_bounds = np.geomspace(2.0e7, 1.0e-5, G + 1)
    idx = np.arange(G, dtype=np.float64)
    sigma_t = 0.8 + 0.2 * idx / max(G - 1, 1)
    rows: list[int] = []
    cols: list[int] = []
    data0: list[float] = []
    data1: list[float] = []
    for src in range(G):
        entries = [(src, 0.28), ((src + 1) % G, 0.10), ((src + 2) % G, 0.045), ((src - 3) % G, 0.012)]
        for out, frac in entries:
            rows.append(src)
            cols.append(out)
            value = frac * sigma_t[src]
            data0.append(value)
            data1.append(0.08 * value if out == src else 0.015 * value)
    sigma_s0 = sparse.coo_matrix((data0, (rows, cols)), shape=(G, G)).tocsc()
    sigma_s1 = sparse.coo_matrix((data1, (rows, cols)), shape=(G, G)).tocsc()
    reactions = {
        "absorption": sigma_t - np.asarray(sigma_s0.sum(axis=1)).ravel(),
        "damage": 0.01 + 0.005 * np.exp(-idx / max(G, 1)),
    }
    material_name = name or f"sparse_synthetic_{G}g"
    return MultigroupLibrary(
        energy_bounds=energy_bounds,
        materials={
            material_name: MaterialXS(
                name=material_name,
                sigma_t=sigma_t,
                sigma_s0=sigma_s0,
                sigma_s1=sigma_s1,
                reactions=reactions,
                heating=0.25 + np.exp(-idx / max(G / 2.0, 1.0)),
                metadata={"synthetic": True, "sparse_scattering": True},
            )
        },
        metadata={"description": f"Sparse synthetic {G}-group library with upscatter/downscatter"},
    )


def estimate_memory_bytes(
    nx: int,
    ny: int,
    nz: int,
    n_dir: int,
    G: int,
    dtype_bytes: int = 8,
    scattering_nnz: int | None = None,
) -> dict[str, int]:
    cells = nx * ny * nz
    nnz = G * G if scattering_nnz is None else int(scattering_nnz)
    index_bytes = 4
    sparse_matrix_bytes = nnz * (dtype_bytes + index_bytes) + (G + 1) * index_bytes
    return {
        "angular_flux": cells * n_dir * G * dtype_bytes,
        "scattering_source_dense": cells * n_dir * G * dtype_bytes,
        "scattering_source_blocked": cells * dtype_bytes,
        "scattering_matrix_dense_pair": 2 * G * G * dtype_bytes,
        "scattering_matrix_sparse_pair": 2 * sparse_matrix_bytes,
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
