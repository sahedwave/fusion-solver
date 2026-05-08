from __future__ import annotations

from pathlib import Path
import json
import tempfile

import numpy as np
import pytest

import sn_multigroup

from sn_core import (
    BoundaryConditions,
    Mesh,
    build_quadrature,
    build_reflection_map,
    dt_source_spectrum,
    make_spectrum_source,
)
from mesh_builder import MeshBuilder
from sn_multigroup import (
    MaterialXS,
    MultigroupLibrary,
    estimate_memory_bytes,
    import_processed_fusion_xs_json,
    load_multigroup_library,
    make_synthetic_library,
    save_multigroup_library,
    source_spectrum_for_named_source,
)
from sn_operators import _scattering_source, _scattering_source_direction_group, _step_cell, _step_cell_python
from sn_solver import SolverConfig, solve_gmres_dsa


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))


def test_schema_validation() -> None:
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values()))
    _check("library G", lib.G == 10)
    _check("material conversion", mat.to_p1_material().G == 10)
    try:
        MaterialXS("bad", np.ones(3), np.ones((3, 2)), np.zeros((3, 3)))
    except ValueError:
        _check("shape mismatch rejected", True)
    else:
        raise AssertionError("shape mismatch was not rejected")


def test_json_npz_roundtrip() -> None:
    lib = make_synthetic_library(10)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for suffix in ("json", "npz"):
            path = tmp_path / f"library.{suffix}"
            save_multigroup_library(lib, path)
            loaded = load_multigroup_library(path)
            key = next(iter(lib.materials))
            _check(f"{suffix} roundtrip G", loaded.G == lib.G)
            _check(
                f"{suffix} roundtrip sigma_t",
                bool(np.allclose(loaded.materials[key].sigma_t, lib.materials[key].sigma_t)),
            )


def _make_fission_material() -> MaterialXS:
    sigma_t = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    sigma_s0 = np.diag([0.2, 0.3, 0.4]).astype(np.float64)
    sigma_s1 = np.zeros((3, 3), dtype=np.float64)
    return MaterialXS(
        name="fissionable",
        sigma_t=sigma_t,
        sigma_s0=sigma_s0,
        sigma_s1=sigma_s1,
        # Keep these payloads in the fission fixture so JSON/NPZ/HDF5
        # roundtrips cover all first-class and legacy material fields.
        reactions={"fission": np.array([0.01, 0.02, 0.03], dtype=np.float64)},
        heating=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        metadata={"kind": "test_fissionable"},
        chi=np.array([0.6, 0.3, 0.1], dtype=np.float64),
        nu_sigma_f=np.array([0.02, 0.05, 0.01], dtype=np.float64),
    )


def test_fission_schema_validation() -> None:
    mat = _make_fission_material()
    _check("fission chi accepted", bool(np.allclose(mat.chi, [0.6, 0.3, 0.1])))
    _check("fission nu_sigma_f accepted", bool(np.allclose(mat.nu_sigma_f, [0.02, 0.05, 0.01])))

    for label, kwargs in (
        ("chi shape mismatch rejected", {"chi": np.array([1.0, 0.0])}),
        ("nu_sigma_f shape mismatch rejected", {"nu_sigma_f": np.array([0.0, 0.0])}),
        ("negative chi rejected", {"chi": np.array([0.6, -0.1, 0.5])}),
        ("negative nu_sigma_f rejected", {"nu_sigma_f": np.array([0.0, -0.1, 0.2])}),
        ("unnormalized chi rejected", {"chi": np.array([0.6, 0.3, 0.2])}),
    ):
        try:
            base = _make_fission_material()
            MaterialXS(
                name="bad_fission",
                sigma_t=base.sigma_t,
                sigma_s0=base.sigma_s0,
                sigma_s1=base.sigma_s1,
                chi=kwargs.get("chi", base.chi),
                nu_sigma_f=kwargs.get("nu_sigma_f", base.nu_sigma_f),
            )
        except ValueError:
            _check(label, True)
        else:
            raise AssertionError(f"{label} was not rejected")


def test_fission_json_npz_roundtrip() -> None:
    mat = _make_fission_material()
    lib = MultigroupLibrary(
        energy_bounds=np.array([20.0e6, 1.0e6, 1.0e3, 1.0e-5], dtype=np.float64),
        materials={mat.name: mat},
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for suffix in ("json", "npz"):
            path = tmp_path / f"fission_library.{suffix}"
            save_multigroup_library(lib, path)
            loaded = load_multigroup_library(path)
            loaded_mat = loaded.materials[mat.name]
            _check(f"{suffix} fission chi roundtrip", bool(np.allclose(loaded_mat.chi, mat.chi)))
            _check(f"{suffix} fission nu_sigma_f roundtrip", bool(np.allclose(loaded_mat.nu_sigma_f, mat.nu_sigma_f)))


def test_old_schema_without_fission_fields_loads() -> None:
    old_json = {
        "energy_bounds": [3.0, 2.0, 1.0],
        "materials": {
            "legacy": {
                "name": "legacy",
                "sigma_t": [1.0, 1.1],
                "sigma_s0": [[0.2, 0.0], [0.0, 0.3]],
                "sigma_s1": [[0.0, 0.0], [0.0, 0.0]],
                "reactions": {},
                "heating": None,
                "metadata": {"schema": "legacy"},
            }
        },
        "metadata": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "legacy.json"
        json_path.write_text(json.dumps(old_json), encoding="utf-8")
        loaded_json = load_multigroup_library(json_path)
        _check("legacy json chi omitted", loaded_json.materials["legacy"].chi is None)
        _check("legacy json nu_sigma_f omitted", loaded_json.materials["legacy"].nu_sigma_f is None)

        npz_path = tmp_path / "legacy.npz"
        np.savez(
            npz_path,
            energy_bounds=np.array([3.0, 2.0, 1.0], dtype=np.float64),
            metadata_json="{}",
            material_keys_json='["legacy"]',
            **{
                "material/legacy/name": np.asarray("legacy"),
                "material/legacy/sigma_t": np.array([1.0, 1.1], dtype=np.float64),
                "material/legacy/sigma_s0": np.array([[0.2, 0.0], [0.0, 0.3]], dtype=np.float64),
                "material/legacy/sigma_s1": np.zeros((2, 2), dtype=np.float64),
                "material/legacy/heating": np.asarray([]),
                "material/legacy/metadata_json": '{"schema": "legacy"}',
                "material/legacy/reaction_keys_json": "[]",
            },
        )
        loaded_npz = load_multigroup_library(npz_path)
        _check("legacy npz chi omitted", loaded_npz.materials["legacy"].chi is None)
        _check("legacy npz nu_sigma_f omitted", loaded_npz.materials["legacy"].nu_sigma_f is None)


def _assert_material_equal(name: str, got: MaterialXS, expected: MaterialXS) -> None:
    _check(f"{name} name", got.name == expected.name)
    _check(f"{name} sigma_t", bool(np.allclose(got.sigma_t, expected.sigma_t)))
    _check(f"{name} sigma_s0", bool(np.allclose(got.sigma_s0, expected.sigma_s0)))
    _check(f"{name} sigma_s1", bool(np.allclose(got.sigma_s1, expected.sigma_s1)))
    _check(f"{name} metadata", got.metadata == expected.metadata)
    _check(f"{name} reaction keys", set(got.reactions) == set(expected.reactions))
    for reaction_key, expected_values in expected.reactions.items():
        _check(f"{name} reaction {reaction_key}", bool(np.allclose(got.reactions[reaction_key], expected_values)))
    _check(
        f"{name} heating",
        got.heating is expected.heating if expected.heating is None else bool(np.allclose(got.heating, expected.heating)),
    )
    _check(
        f"{name} chi",
        got.chi is expected.chi if expected.chi is None else bool(np.allclose(got.chi, expected.chi)),
    )
    _check(
        f"{name} nu_sigma_f",
        got.nu_sigma_f is expected.nu_sigma_f
        if expected.nu_sigma_f is None
        else bool(np.allclose(got.nu_sigma_f, expected.nu_sigma_f)),
    )


def _assert_library_equal(got: MultigroupLibrary, expected: MultigroupLibrary) -> None:
    _check("library energy_bounds", bool(np.allclose(got.energy_bounds, expected.energy_bounds)))
    _check("library group_names", got.group_names == expected.group_names)
    _check("library lethargy_widths", bool(np.allclose(got.lethargy_widths, expected.lethargy_widths)))
    _check("library source_group_mapping", got.source_group_mapping == expected.source_group_mapping)
    _check("library metadata", got.metadata == expected.metadata)
    _check("library material keys", list(got.materials) == list(expected.materials))
    for key, expected_mat in expected.materials.items():
        _assert_material_equal(f"material {key}", got.materials[key], expected_mat)


def test_hdf5_roundtrip_synthetic_library() -> None:
    pytest.importorskip("h5py")
    lib = make_synthetic_library(10)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.h5"
        save_multigroup_library(lib, path)
        _assert_library_equal(load_multigroup_library(path), lib)


def test_hdf5_roundtrip_fission_fields_and_hdf5_suffix() -> None:
    pytest.importorskip("h5py")
    mat = _make_fission_material()
    lib = MultigroupLibrary(
        energy_bounds=np.array([20.0e6, 1.0e6, 1.0e3, 1.0e-5], dtype=np.float64),
        materials={mat.name: mat},
        metadata={"format": "hdf5_test"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fission.hdf5"
        save_multigroup_library(lib, path)
        _assert_library_equal(load_multigroup_library(path), lib)


def test_optional_group_metadata_validation_and_computed_lethargy() -> None:
    lib = make_synthetic_library(4)
    bounds = lib.energy_bounds
    expected = np.abs(np.log(bounds[:-1] / bounds[1:]))
    _check("computed lethargy from bounds", bool(np.allclose(lib.lethargy_widths, expected)))

    with pytest.raises(ValueError, match="group_names must have length"):
        MultigroupLibrary(energy_bounds=bounds, materials=lib.materials, group_names=("g0",))
    with pytest.raises(ValueError, match="lethargy_widths must have shape"):
        MultigroupLibrary(energy_bounds=bounds, materials=lib.materials, lethargy_widths=np.ones((3, 1)))
    with pytest.raises(ValueError, match="out of range"):
        MultigroupLibrary(energy_bounds=bounds, materials=lib.materials, source_group_mapping={"dt": 7})


def test_group_metadata_roundtrip_json_npz_hdf5() -> None:
    pytest.importorskip("h5py")
    base = make_synthetic_library(5)
    mat_name = next(iter(base.materials))
    group_names = tuple(f"group_{g}" for g in range(base.G))
    lethargy = np.linspace(0.1, 0.5, base.G)
    source_map = {"dt": 0, "alpha_n": {"group": 3, "tag": "fusion"}}
    lib = MultigroupLibrary(
        energy_bounds=base.energy_bounds,
        materials={mat_name: base.materials[mat_name]},
        group_names=group_names,
        lethargy_widths=lethargy,
        source_group_mapping=source_map,
        metadata={"schema": "with_group_metadata"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("json", "npz", "h5"):
            path = Path(tmp) / f"metadata_roundtrip.{suffix}"
            save_multigroup_library(lib, path)
            loaded = load_multigroup_library(path)
            _assert_library_equal(loaded, lib)


def test_hdf5_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = sn_multigroup.importlib.util.find_spec

    def missing_h5py(name: str) -> object:
        return None if name == "h5py" else find_spec(name)

    monkeypatch.setattr(sn_multigroup.importlib.util, "find_spec", missing_h5py)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "library.h5"
        with pytest.raises(ImportError, match="pip install h5py"):
            save_multigroup_library(make_synthetic_library(2), path)
        with pytest.raises(ImportError, match="pip install h5py"):
            load_multigroup_library(path)


def test_real_schema_example() -> None:
    lib = load_multigroup_library(
        Path(__file__).resolve().parent / "data" / "multigroup" / "example_real_schema.json"
    )
    _check("real schema example G", lib.G == 2)
    _check("real schema material present", "mock_steel" in lib.materials)
    _check("real schema converts", lib.materials["mock_steel"].to_p1_material().G == 2)


def test_processed_importer_success_and_provenance_flag() -> None:
    path = Path(__file__).resolve().parent / "data" / "multigroup" / "example_real_schema.json"
    lib = import_processed_fusion_xs_json(path)
    _check("processed import G", lib.G == 2)
    _check("processed import material present", "mock_steel" in lib.materials)
    _check("processed format flag true with provenance", lib.metadata.get("processed_external_format") is True)


def test_processed_importer_shape_mismatch_rejected(tmp_path: Path) -> None:
    bad = {
        "energy_bounds": [3.0, 2.0, 1.0],
        "metadata": {"provenance": "unit-test"},
        "materials": {
            "m": {
                "name": "m",
                "sigma_t": [1.0, 1.1],
                "sigma_s0": [[0.2, 0.0], [0.0, 0.3]],
                "sigma_s1": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            }
        },
    }
    p = tmp_path / "bad_processed.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="must have shape"):
        import_processed_fusion_xs_json(p)


def test_processed_importer_missing_provenance_warns(tmp_path: Path) -> None:
    payload = {
        "energy_bounds": [3.0, 2.0, 1.0],
        "metadata": {"description": "no provenance"},
        "materials": {
            "m": {
                "name": "m",
                "sigma_t": [1.0, 1.1],
                "sigma_s0": [[0.2, 0.0], [0.0, 0.3]],
                "sigma_s1": [[0.0, 0.0], [0.0, 0.0]],
            }
        },
    }
    p = tmp_path / "no_provenance.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.warns(UserWarning, match="missing metadata.provenance"):
        lib = import_processed_fusion_xs_json(p)
    _check("processed format flag false without provenance", lib.metadata.get("processed_external_format") is False)


def test_sources() -> None:
    mesh = Mesh(4, 4, 4, 1.0, 1.0, 1.0)
    spectrum = np.arange(1, 11, dtype=np.float64)
    Q = make_spectrum_source(mesh, spectrum, strength=7.5, geometry="gaussian")
    _check("spectrum source shape", Q.shape == (4, 4, 4, 10), str(Q.shape))
    _check("spectrum source normalization", abs(float(Q.sum()) - 7.5) < 1.0e-12)

    bounds_desc = np.geomspace(2.0e7, 1.0e-5, 11)
    dt = dt_source_spectrum(bounds_desc)
    _check("D-T spectrum shape", dt.shape == (10,))
    _check("D-T spectrum normalized", float(dt.sum()) == 1.0)
    idx = int(np.argmax(dt))
    lo = min(bounds_desc[idx], bounds_desc[idx + 1])
    hi = max(bounds_desc[idx], bounds_desc[idx + 1])
    _check("D-T group contains 14.1 MeV", lo <= 14.1e6 <= hi)


def test_dt_source_spectrum_descending_and_ascending_bounds() -> None:
    bounds_desc = np.array([20.0e6, 14.05e6, 1.0e6], dtype=np.float64)
    desc = dt_source_spectrum(bounds_desc)
    _check("descending dt source picks first bin", bool(np.allclose(desc, [1.0, 0.0])))

    bounds_asc = bounds_desc[::-1]
    asc = dt_source_spectrum(bounds_asc)
    _check("ascending dt source picks last bin", bool(np.allclose(asc, [0.0, 1.0])))


def test_source_spectrum_for_named_source_mapping_and_fallback() -> None:
    base = make_synthetic_library(4)
    mapped = MultigroupLibrary(
        energy_bounds=base.energy_bounds,
        materials=base.materials,
        source_group_mapping={"DT_14MeV": {"group": 2, "label": "user_override"}},
    )
    mapped_spec = source_spectrum_for_named_source(mapped, "DT_14MeV")
    _check("explicit dt mapping honored", bool(np.allclose(mapped_spec, [0.0, 0.0, 1.0, 0.0])))

    fallback_spec = source_spectrum_for_named_source(base, "DT_14MeV")
    expected = dt_source_spectrum(base.energy_bounds)
    _check("dt fallback matches energy scan", bool(np.allclose(fallback_spec, expected)))


def test_dt_source_out_of_range_raises_cleanly() -> None:
    bounds = np.array([1.0e6, 5.0e5, 1.0e5], dtype=np.float64)
    with pytest.raises(ValueError, match="outside the energy group bounds"):
        dt_source_spectrum(bounds, neutron_energy_ev=14.1e6)


def test_named_source_spectrum_conserves_strength_through_make_spectrum_source() -> None:
    lib = make_synthetic_library(6)
    spectrum = source_spectrum_for_named_source(lib, "DT_14MeV")
    mesh = Mesh(4, 3, 2, 0.5, 0.75, 1.25)
    strength = 3.7
    Q = make_spectrum_source(mesh, spectrum, strength=strength, geometry="gaussian")
    vol = mesh.dx * mesh.dy * mesh.dz
    _check("named source spectrum one-hot normalized", float(spectrum.sum()) == 1.0)
    _check("named source strength conserved", abs(float(np.sum(Q) * vol) - strength) < 1.0e-12)


def test_spectrum_source_conservation_structured_unstructured() -> None:
    spectrum = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    strength = 4.2

    # Structured reference.
    mesh_s = Mesh(4, 3, 2, 0.5, 0.75, 1.25)
    Qs = make_spectrum_source(mesh_s, spectrum, strength=strength, geometry="gaussian")
    vol_s = mesh_s.dx * mesh_s.dy * mesh_s.dz
    _check("structured spectrum shape", Qs.shape == (mesh_s.nx, mesh_s.ny, mesh_s.nz, spectrum.size))
    _check("structured source conservation", abs(float(np.sum(Qs) * vol_s) - strength) < 1.0e-12)

    # Cartesian-converted unstructured.
    mesh_c = MeshBuilder.from_cartesian(mesh_s)
    Qc = make_spectrum_source(mesh_c, spectrum, strength=strength, geometry="volumetric", plasma_fraction=0.35)
    _check("from_cartesian spectrum shape", Qc.shape == (mesh_c.N_cells, spectrum.size))
    _check("from_cartesian source conservation", abs(float(np.sum(Qc * mesh_c.cell_volume[:, None])) - strength) < 1.0e-12)

    # General tet mesh.
    mesh_t = MeshBuilder.tet_box(4, 3, 2, 1.0, 1.0, 1.0)
    Qt = make_spectrum_source(mesh_t, spectrum, strength=strength, geometry="point")
    _check("tet_box spectrum shape", Qt.shape == (mesh_t.N_cells, spectrum.size))
    _check("tet_box source conservation", abs(float(np.sum(Qt * mesh_t.cell_volume[:, None])) - strength) < 1.0e-12)


def _solver_smoke(G: int) -> None:
    lib = make_synthetic_library(G)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(3, 3, 3, 1.0, 1.0, 1.0)
    dirs, wts = build_quadrature(4)
    spectrum = dt_source_spectrum(lib.energy_bounds)
    Q = make_spectrum_source(mesh, spectrum, strength=1.0, geometry="point")
    result = solve_gmres_dsa(
        mesh,
        mat,
        Q,
        dirs,
        wts,
        BoundaryConditions(),
        build_reflection_map(dirs),
        SolverConfig(tol=1.0e-6, max_outer=8, gmres_restart=20, inner_tol=1.0e-8),
    )
    _check(f"{G}-group smoke finite", bool(np.all(np.isfinite(result.phi))))
    _check(f"{G}-group smoke nonnegative", float(result.phi.min()) >= 0.0)
    _check(f"{G}-group smoke nonzero", float(result.phi.sum()) > 0.0)


def test_solver_smoke() -> None:
    _solver_smoke(10)
    _solver_smoke(27)


def test_positivity_diagnostics() -> None:
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values())).to_p1_material()
    mesh = Mesh(3, 3, 3, 1.0, 1.0, 1.0)
    dirs, wts = build_quadrature(4)
    Q = make_spectrum_source(mesh, dt_source_spectrum(lib.energy_bounds), strength=1.0)
    result = solve_gmres_dsa(
        mesh,
        mat,
        Q,
        dirs,
        wts,
        BoundaryConditions(),
        build_reflection_map(dirs),
        SolverConfig(tol=1.0e-6, max_outer=8, gmres_restart=20, inner_tol=1.0e-8),
    )
    diag = result.positivity_diagnostics
    _check("positivity diagnostics keys", {"negative_flux_before_floor", "negative_cell_count", "clipped_flux_integral", "relative_balance_change", "rebalance_applied"} <= set(diag))
    _check("positivity diagnostics nonnegative clip", diag["clipped_flux_integral"] >= 0.0)


def test_memory_estimate() -> None:
    est = estimate_memory_bytes(10, 10, 10, 80, 175)
    _check("memory estimate keys", {"angular_flux", "scattering_source_dense", "scattering_source_blocked", "scalar_flux", "current"} <= set(est))
    _check("memory estimate positive", all(value > 0 for value in est.values()))
    _check("blocked scattering memory smaller", est["scattering_source_blocked"] < est["scattering_source_dense"])


def test_blocked_scattering_equivalence() -> None:
    rng = np.random.default_rng(42)
    lib = make_synthetic_library(10)
    mat = next(iter(lib.materials.values())).to_p1_material()
    directions, _ = build_quadrature(4)
    phi = rng.random((2, 3, 2, mat.G))
    J = rng.random((2, 3, 2, mat.G, 3))
    dense = _scattering_source(phi, J, mat, directions)
    max_err = 0.0
    for m, direction in enumerate(directions):
        for g in range(mat.G):
            blocked = _scattering_source_direction_group(phi, J, mat, direction, g)
            max_err = max(max_err, float(np.max(np.abs(blocked - dense[:, :, :, m, g]))))
    _check("blocked scattering equals dense", max_err < 1.0e-12, f"max_err={max_err:.2e}")


def test_step_cell_acceleration_equivalence() -> None:
    args = (0.2, 0.1, 0.05, 0.7, 1.1, 0.3, 0.4, 0.5, 2.0, 2.5, 3.0)
    ref = _step_cell_python(*args)
    got = _step_cell(*args)
    _check("step cell accelerated equivalence", bool(np.allclose(got, ref, rtol=0.0, atol=1.0e-14)))


def main() -> None:
    test_schema_validation()
    test_json_npz_roundtrip()
    test_fission_schema_validation()
    test_fission_json_npz_roundtrip()
    test_old_schema_without_fission_fields_loads()
    test_real_schema_example()
    test_sources()
    test_solver_smoke()
    test_positivity_diagnostics()
    test_memory_estimate()
    test_blocked_scattering_equivalence()
    test_step_cell_acceleration_equivalence()
    print("Multigroup library validation complete.")


if __name__ == "__main__":
    main()
