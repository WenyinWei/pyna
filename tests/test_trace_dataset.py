from types import SimpleNamespace

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo import Orbit, Trajectory
from pyna.trace_dataset import (
    DATASET_COMPATIBILITY,
    DATASET_SCHEMA_NAME,
    DATASET_SCHEMA_VERSION,
    SEGMENT_SCHEMA_NAME,
    TraceRecord,
    TraceDataset,
)


def _skip_without_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.trace_orbit_along_phi is None:
        pytest.skip("cyna field-line tracing is unavailable")


def _rotation_field(omega=0.25):
    R0 = 1.0
    Z0 = 0.0
    R = np.linspace(0.7, 1.3, 33)
    Z = np.linspace(-0.3, 0.3, 33)
    Phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    RR, ZZ, _PP = np.meshgrid(R, Z, Phi, indexing="ij")
    fR = -omega * (ZZ - Z0)
    fZ = omega * (RR - R0)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=fR / RR,
        BZ=fZ / RR,
        BPhi=np.ones_like(RR),
    )


def _outward_field():
    R = np.linspace(0.7, 1.3, 33)
    Z = np.linspace(-0.3, 0.3, 17)
    Phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    RR, ZZ, _PP = np.meshgrid(R, Z, Phi, indexing="ij")
    fR = np.full_like(RR, 0.05)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=fR / RR,
        BZ=np.zeros_like(RR),
        BPhi=np.ones_like(RR),
    )


def _toroidal_wall():
    wall_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_R = np.tile(1.0 + 0.2 * np.cos(theta), (wall_phi.size, 1))
    wall_Z = np.tile(0.2 * np.sin(theta), (wall_phi.size, 1))
    return wall_phi, wall_R, wall_Z


class LinearFlowND:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.trajectory_calls = 0

    def rhs(self, t, x):
        return 0.1 + 0.02 * np.arange(self.dim, dtype=float) + 0.03 * np.asarray(x, dtype=float)

    def trajectory(self, x0, t_span, *, dt=None, t_eval=None, **kwargs):
        del dt, kwargs
        self.trajectory_calls += 1
        if t_eval is not None:
            times = np.asarray(t_eval, dtype=float)
        else:
            times = np.linspace(float(t_span[0]), float(t_span[1]), 3)
        x = np.asarray(x0, dtype=float)
        states = [x.copy()]
        for index in range(times.size - 1):
            h = float(times[index + 1] - times[index])
            x = x + h * self.rhs(float(times[index]), x)
            states.append(x.copy())
        return Trajectory(
            states=np.vstack(states),
            times=times,
            coordinate_names=tuple(["x", "y", "z", "w"][: self.dim]),
            metadata={"trajectory_calls": self.trajectory_calls},
        )


class ShiftMapND:
    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)
        self.orbit_calls = 0

    def orbit(self, x0, n_iter):
        self.orbit_calls += 1
        x = np.asarray(x0, dtype=float)
        states = [x.copy()]
        for _ in range(int(n_iter)):
            x = x + self.delta
            states.append(x.copy())
        return np.vstack(states)


class StepOnlyMapND:
    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)
        self.step_calls = 0

    def step(self, x0):
        self.step_calls += 1
        return np.asarray(x0, dtype=float) + self.delta


def test_trace_dataset_round_trips_topo_trajectory_and_schema(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    trajectory = Trajectory(
        states=np.asarray([[0.0, 1.0], [0.2, 1.1], [0.4, 1.2]]),
        times=np.asarray([0.0, 0.5, 1.0]),
        time_name="tau",
        coordinate_names=("x", "y"),
        metadata={"case": "trajectory"},
    )

    trajectory_id = dataset.append_topo_trajectory(
        trajectory,
        source_signature={"system": "linear"},
        settings={"dt": 0.5},
    )

    record = dataset.get(trajectory_id)
    assert record.kind == "trajectory"
    assert record.n_segments == 1
    assert record.n_samples == 3
    assert record.ambient_dim == 2
    assert record.independent_name == "tau"
    assert record.coordinate_names == ("x", "y")
    np.testing.assert_allclose(record.last_state, [0.4, 1.2])

    loaded = dataset.load_topo_trajectory(trajectory_id)
    assert isinstance(loaded, Trajectory)
    assert loaded.time_name == "tau"
    assert loaded.coordinate_names == ("x", "y")
    assert loaded.metadata["case"] == "trajectory"
    assert loaded.independent_name == "tau"
    np.testing.assert_allclose(loaded.independent, trajectory.times)
    np.testing.assert_allclose(loaded.initial, trajectory.states[0])
    np.testing.assert_allclose(loaded.final, trajectory.states[-1])
    assert np.isclose(loaded.start_independent, 0.0)
    assert np.isclose(loaded.end_independent, 1.0)
    np.testing.assert_allclose(loaded.states, trajectory.states)
    np.testing.assert_allclose(loaded.times, trajectory.times)

    catalog = dataset._conn.execute("SELECT * FROM dataset WHERE id = 1").fetchone()
    assert catalog["schema_name"] == DATASET_SCHEMA_NAME
    assert int(catalog["schema_version"]) == DATASET_SCHEMA_VERSION
    assert catalog["compatibility"] == DATASET_COMPATIBILITY

    segment_row = dataset._conn.execute("SELECT * FROM segment WHERE trajectory_id = ?", (trajectory_id,)).fetchone()
    with np.load(tmp_path / "trace_dataset" / segment_row["chunk_path"], allow_pickle=False) as raw:
        assert str(np.asarray(raw["schema_name"]).item()) == SEGMENT_SCHEMA_NAME
        assert int(np.asarray(raw["schema_version"]).item()) == DATASET_SCHEMA_VERSION
        assert str(np.asarray(raw["compatibility"]).item()) == DATASET_COMPATIBILITY
        assert bool(np.asarray(raw["complete"]).item()) is True
        np.testing.assert_allclose(raw["states"], trajectory.states)
        np.testing.assert_allclose(raw["independent"], trajectory.times)

    dataset.close()


def test_trace_dataset_appends_and_loads_generic_topo_trace_with_metadata_identity(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    orbit = Orbit(
        states=np.asarray([[0.0, 0.0], [1.0, -0.5], [2.0, -1.0]]),
        steps=None,
        coordinate_names=("x", "y"),
        metadata={
            "source_signature": {"map": "metadata-shift"},
            "trace_settings": {"segment_steps": 2},
            "seed": [0.0, 0.0],
            "case": "generic-topo-trace",
        },
    )

    trajectory_id = dataset.append_topo_trace(orbit)
    record = dataset.get(trajectory_id)
    assert isinstance(record, TraceRecord)
    assert record.trace_id == record.trajectory_id
    assert record.kind == "orbit"
    assert record.source_signature == {"map": "metadata-shift"}
    assert record.settings == {"segment_steps": 2}
    assert record.seed == [0.0, 0.0]

    loaded = dataset.load_topo_trace(trajectory_id)
    assert isinstance(loaded, Orbit)
    assert loaded.independent_name == "step"
    np.testing.assert_array_equal(loaded.independent, np.arange(3))
    np.testing.assert_allclose(loaded.initial, [0.0, 0.0])
    np.testing.assert_allclose(loaded.final, [2.0, -1.0])
    assert loaded.start_independent == 0
    assert loaded.end_independent == 2
    dataset.close()


def test_trace_dataset_rejects_corrupted_segment_chunk(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    trajectory = Trajectory(
        states=np.asarray([[0.0, 0.0], [1.0, 0.5]]),
        times=np.asarray([0.0, 1.0]),
    )
    trajectory_id = dataset.append_topo_trace(trajectory)
    segment_row = dataset._conn.execute(
        "SELECT * FROM segment WHERE trajectory_id = ?",
        (trajectory_id,),
    ).fetchone()
    chunk_path = tmp_path / "trace_dataset" / segment_row["chunk_path"]
    with np.load(chunk_path, allow_pickle=False) as raw:
        arrays = {key: np.array(raw[key]) for key in raw.files}
    arrays["states"] = np.asarray(arrays["states"], dtype=float)
    arrays["states"][1, 0] += 1.0
    np.savez_compressed(chunk_path, **arrays)

    report = dataset.validate_storage()
    assert not report["ok"]
    assert report["issues"][0]["kind"] == "segment_chunk_invalid"

    with pytest.raises(ValueError, match="states summary mismatch"):
        dataset.load_segments(trajectory_id)
    dataset.close()


def test_trace_dataset_validate_storage_reports_clean_and_orphan_chunks(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    trajectory = Trajectory(
        states=np.asarray([[0.0, 0.0], [1.0, 0.5]]),
        times=np.asarray([0.0, 1.0]),
    )
    dataset.append_topo_trace(trajectory)

    clean = dataset.validate_storage()
    assert clean["ok"]
    assert clean["n_issues"] == 0
    assert clean["n_segments"] == 1
    assert clean["n_chunks"] == 1

    orphan_dir = tmp_path / "trace_dataset" / "chunks" / "orphan"
    orphan_dir.mkdir(parents=True)
    np.savez_compressed(orphan_dir / "unused.npz", data=np.asarray([1.0]))

    report = dataset.validate_storage()
    assert not report["ok"]
    assert report["n_orphan_chunks"] == 1
    assert any(issue["kind"] == "orphan_chunk" for issue in report["issues"])
    dataset.close()


def test_trace_dataset_validate_storage_reports_trajectory_aggregate_mismatch(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    trajectory_id = dataset.append_topo_trace(
        Trajectory(
            states=np.asarray([[0.0, 0.0], [1.0, 0.5]]),
            times=np.asarray([0.0, 1.0]),
        )
    )
    dataset._conn.execute(
        "UPDATE trajectory SET n_samples = ? WHERE trajectory_id = ?",
        (99, trajectory_id),
    )
    dataset._conn.commit()

    report = dataset.validate_storage()
    assert not report["ok"]
    assert any(
        issue["kind"] == "trajectory_aggregate_mismatch"
        and issue["field"] == "n_samples"
        for issue in report["issues"]
    )
    dataset.close()


def test_trace_dataset_round_trips_orbit_with_and_without_steps(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    orbit = Orbit(
        states=np.asarray([[1.0, 0.0], [1.5, -0.5], [2.0, -1.0]]),
        steps=np.asarray([0, 2, 4]),
        coordinate_names=("u", "v"),
        metadata={"case": "orbit"},
    )
    stepped_id = dataset.append_topo_orbit(orbit, settings={"stride": 2})
    loaded = dataset.load_topo_trajectory(stepped_id)
    assert isinstance(loaded, Orbit)
    np.testing.assert_allclose(loaded.states, orbit.states)
    np.testing.assert_array_equal(loaded.steps, [0, 2, 4])
    assert loaded.coordinate_names == ("u", "v")

    nosteps = Orbit(states=np.asarray([[0.0], [1.0], [2.0]]), steps=None)
    nosteps_id = dataset.append_topo_orbit(nosteps)
    loaded_nosteps = dataset.load_topo_trajectory(nosteps_id)
    np.testing.assert_array_equal(loaded_nosteps.steps, [0, 1, 2])
    np.testing.assert_allclose(loaded_nosteps.states, nosteps.states)
    dataset.close()


def test_trace_dataset_selectively_resumes_generic_4d_trajectories(tmp_path):
    flow = LinearFlowND(dim=4)
    signature = {"system": "linear4d"}
    dataset = TraceDataset(tmp_path / "trace_dataset")
    seeds = [
        np.asarray([0.0, 0.0, 0.2, -0.1]),
        np.asarray([0.1, 0.02, 0.2, -0.1]),
        np.asarray([0.8, 0.0, 0.2, -0.1]),
    ]
    trajectory_ids = []
    for seed in seeds:
        initial = flow.trajectory(seed, (0.0, 0.2))
        trajectory_ids.append(dataset.append_topo_trajectory(initial, source_signature=signature))

    polygon = np.asarray([[-0.1, -0.05], [0.2, -0.05], [0.2, 0.08], [-0.1, 0.08]])
    selected = dataset.select(kind="trajectory", seed_polygon=polygon, seed_coords=("x", "y"))
    assert set(selected.ids) == set(trajectory_ids[:2])

    resumed = selected.resume_trajectories(
        flow,
        target_time=0.6,
        segment_time_span=0.2,
        source_signature=signature,
    )

    assert {record.trajectory_id for record in resumed} == set(trajectory_ids[:2])
    assert flow.trajectory_calls == 7
    for trajectory_id in trajectory_ids[:2]:
        record = dataset.get(trajectory_id)
        assert record.n_segments == 3
        assert record.n_samples == 7
        assert record.ambient_dim == 4
        assert np.isclose(record.end_independent, 0.6)
        loaded = dataset.load_topo_trajectory(trajectory_id)
        assert loaded.ambient_dim == 4
        np.testing.assert_allclose(loaded.times, np.linspace(0.0, 0.6, 7), atol=1.0e-14)

    unselected = dataset.get(trajectory_ids[2])
    assert unselected.n_segments == 1
    assert unselected.n_samples == 3
    assert np.isclose(unselected.end_independent, 0.2)
    summary = dataset.summary(kind="trajectory")
    assert summary["n_trajectories"] == 3
    assert summary["n_segments"] == 7
    assert summary["chunk_files"] == 7
    dataset.close()


def test_trace_dataset_selectively_iterates_generic_5d_orbits(tmp_path):
    from pyna.topo.workflow import orbit_from_map

    map_obj = ShiftMapND(delta=np.asarray([1.0, -0.5, 0.25, 0.1, -0.2]))
    signature = {"map": "shift5d"}
    dataset = TraceDataset(tmp_path / "trace_dataset")
    seeds = [
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.asarray([0.1, 0.02, 0.0, 0.0, 0.0]),
        np.asarray([0.8, 0.0, 0.0, 0.0, 0.0]),
    ]
    trajectory_ids = []
    for seed in seeds:
        initial = orbit_from_map(map_obj, seed, 2)
        trajectory_ids.append(dataset.append_topo_orbit(initial, source_signature=signature))

    polygon = np.asarray([[-0.1, -0.05], [0.2, -0.05], [0.2, 0.08], [-0.1, 0.08]])
    selected = dataset.select(kind="orbit", seed_polygon=polygon, seed_coords=("x0", "x1"))
    assert set(selected.ids) == set(trajectory_ids[:2])

    resumed = selected.iterate_orbits(
        map_obj,
        target_step=5,
        segment_steps=2,
        source_signature=signature,
    )

    assert {record.trajectory_id for record in resumed} == set(trajectory_ids[:2])
    for trajectory_id in trajectory_ids[:2]:
        record = dataset.get(trajectory_id)
        assert record.n_segments == 3
        assert record.n_samples == 6
        assert record.ambient_dim == 5
        assert np.isclose(record.end_independent, 5.0)
        loaded = dataset.load_topo_trajectory(trajectory_id)
        np.testing.assert_array_equal(loaded.steps, np.arange(6))
        np.testing.assert_allclose(loaded.states[-1], loaded.states[0] + 5.0 * map_obj.delta)

    unselected = dataset.get(trajectory_ids[2])
    assert unselected.n_segments == 1
    assert unselected.n_samples == 3
    assert np.isclose(unselected.end_independent, 2.0)
    summary = dataset.summary(kind="orbit")
    assert summary["n_trajectories"] == 3
    assert summary["n_segments"] == 7
    assert summary["chunk_files"] == 7
    dataset.close()


def test_trace_dataset_resumes_generic_orbits_from_step_only_map(tmp_path):
    from pyna.topo.workflow import orbit_from_map

    map_obj = StepOnlyMapND(delta=np.asarray([0.5, -0.25, 1.0]))
    signature = {"map": "step-only-3d"}
    dataset = TraceDataset(tmp_path / "trace_dataset")
    seeds = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ]
    trajectory_ids = []
    for seed in seeds:
        initial = orbit_from_map(map_obj, seed, 1)
        trajectory_ids.append(dataset.append_topo_orbit(initial, source_signature=signature))

    polygon = np.asarray([[-0.1, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.1, 0.1]])
    selected = dataset.select(kind="orbit", seed_polygon=polygon, seed_coords=("x0", "x1"))
    assert selected.ids == [trajectory_ids[0]]

    selected.iterate_orbits(
        map_obj,
        target_step=4,
        segment_steps=1,
        source_signature=signature,
    )

    resumed = dataset.get(trajectory_ids[0])
    assert resumed.n_segments == 4
    assert resumed.n_samples == 5
    assert np.isclose(resumed.end_independent, 4.0)
    loaded = dataset.load_topo_trajectory(trajectory_ids[0])
    np.testing.assert_allclose(loaded.states[-1], seeds[0] + 4.0 * map_obj.delta)

    unselected = dataset.get(trajectory_ids[1])
    assert unselected.n_segments == 1
    assert unselected.n_samples == 2
    assert np.isclose(unselected.end_independent, 1.0)
    assert map_obj.step_calls == 5
    dataset.close()


def test_trace_dataset_fieldline_polygon_selection_and_round_trip(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    field_signature = {"case": "synthetic_field"}
    inside = SimpleNamespace(
        phi=np.asarray([0.0, 0.25, 0.5]),
        R=np.asarray([1.0, 1.01, 1.02]),
        Z=np.asarray([0.0, 0.02, 0.04]),
        alive=np.asarray([1, 1, 1], dtype=np.int8),
        metadata={"R0": 1.0, "Z0": 0.0, "phi_start": 0.0, "phi_end": 0.5, "DPhi": 0.01, "dphi_out": 0.25},
    )
    outside = SimpleNamespace(
        phi=np.asarray([0.0, 0.25]),
        R=np.asarray([1.4, 1.41]),
        Z=np.asarray([0.0, 0.01]),
        alive=np.asarray([1, 1], dtype=np.int8),
        metadata={"R0": 1.4, "Z0": 0.0, "phi_start": 0.0, "phi_end": 0.25, "DPhi": 0.01, "dphi_out": 0.25},
    )

    inside_id = dataset.append_fieldline_trajectory(inside, field_signature=field_signature)
    dataset.append_fieldline_trajectory(outside, field_signature=field_signature)
    seed_row = dataset._conn.execute(
        "SELECT seed0_name, seed0, seed1_name, seed1, seed2_name, seed2 FROM trajectory WHERE trajectory_id = ?",
        (inside_id,),
    ).fetchone()
    assert seed_row["seed0_name"] == "R"
    assert seed_row["seed1_name"] == "Z"
    assert seed_row["seed2_name"] == "phi"
    assert np.isclose(seed_row["seed0"], 1.0)

    polygon = np.asarray([[0.9, -0.1], [1.1, -0.1], [1.1, 0.1], [0.9, 0.1]])
    selection = dataset.select(kind="fieldline", seed_polygon=polygon, seed_coords=("R", "Z"))
    assert selection.ids == [inside_id]

    loaded = dataset.load_fieldline_trajectory(inside_id)
    assert loaded.status == "extendable"
    np.testing.assert_allclose(loaded.phi, inside.phi)
    np.testing.assert_allclose(loaded.R, inside.R)
    np.testing.assert_array_equal(loaded.alive, inside.alive)
    dataset.close()


def test_trace_dataset_collection_selects_polygon_across_cache_roots(tmp_path):
    field_signature = {"case": "synthetic_field"}
    ids = []
    for index, seed_R in enumerate((1.0, 1.05, 1.4)):
        dataset = TraceDataset(tmp_path / f"trace_dataset_{index}")
        trajectory = SimpleNamespace(
            phi=np.asarray([0.0, 0.2]),
            R=np.asarray([seed_R, seed_R + 0.01]),
            Z=np.asarray([0.0, 0.01]),
            alive=np.asarray([1, 1], dtype=np.int8),
            metadata={"R0": seed_R, "Z0": 0.0, "phi_start": 0.0, "phi_end": 0.2, "DPhi": 0.01, "dphi_out": 0.2},
        )
        ids.append(dataset.append_fieldline_trajectory(trajectory, field_signature=field_signature))
        dataset.close()

    collection = TraceDataset.open_many(tmp_path / f"trace_dataset_{index}" for index in range(3))
    polygon = np.asarray([[0.9, -0.1], [1.1, -0.1], [1.1, 0.1], [0.9, 0.1]])

    selection = collection.select(kind="fieldline", seed_polygon=polygon, seed_coords=("R", "Z"))

    assert selection.ids == ids[:2]
    for dataset in collection.datasets:
        dataset.close()


def test_trace_dataset_fieldline_resume_rejects_wrong_direction(tmp_path):
    dataset = TraceDataset(tmp_path / "trace_dataset")
    trajectory = SimpleNamespace(
        phi=np.asarray([0.0, 0.5]),
        R=np.asarray([1.0, 1.1]),
        Z=np.asarray([0.0, 0.1]),
        alive=np.asarray([1, 1], dtype=np.int8),
        metadata={"R0": 1.0, "Z0": 0.0, "phi_start": 0.0, "phi_end": 0.5, "DPhi": 0.01, "dphi_out": 0.5},
    )
    trajectory_id = dataset.append_fieldline_trajectory(
        trajectory,
        field_signature={"type": "object"},
        direction="forward",
    )

    with pytest.raises(ValueError, match="behind forward"):
        dataset.resume_fieldline_trajectories(
            object(),
            target_phi=0.25,
            trajectory_ids=[trajectory_id],
            validate_field=False,
        )
    dataset.close()


def test_trace_dataset_resumes_selected_fieldline_from_cursor(tmp_path):
    _skip_without_cyna()

    from pyna.toroidal.flt import trace_fieldline_trajectory

    field = _rotation_field()
    dataset = TraceDataset(tmp_path / "trace_dataset")
    first = trace_fieldline_trajectory(
        field,
        1.08,
        0.02,
        0.0,
        0.4,
        0.01,
        dphi_out=0.1,
        storage="memory",
    )
    trajectory_id = dataset.append_fieldline_trajectory(first, status="extendable")

    resumed = dataset.resume_fieldline_trajectories(
        field,
        target_phi=0.8,
        trajectory_ids=[trajectory_id],
        chunk_phi_span=0.2,
        storage="memory",
    )

    assert [record.trajectory_id for record in resumed] == [trajectory_id]
    record = dataset.get(trajectory_id)
    assert record.n_segments == 2
    assert record.status == "extendable"
    assert np.isclose(record.end_independent, 0.8)
    assert record.n_samples == 9

    loaded = dataset.load_fieldline_trajectory(trajectory_id)
    assert loaded.phi.shape == (9,)
    np.testing.assert_allclose(loaded.phi, np.linspace(0.0, 0.8, 9), atol=1.0e-14)
    dataset.close()


def test_trace_dataset_selectively_resumes_real_fieldlines_by_seed_polygon(tmp_path):
    _skip_without_cyna()

    from pyna.toroidal.flt import trace_fieldline_trajectory

    field = _rotation_field()
    dataset = TraceDataset(tmp_path / "trace_dataset")
    seeds = [(1.04, 0.02), (1.08, 0.02), (1.22, 0.02)]
    trajectory_ids = []
    for R0, Z0 in seeds:
        initial = trace_fieldline_trajectory(
            field,
            R0,
            Z0,
            0.0,
            0.2,
            0.01,
            dphi_out=0.1,
            storage="memory",
        )
        trajectory_ids.append(dataset.append_fieldline_trajectory(initial, status="extendable"))

    polygon = np.asarray([[1.0, -0.02], [1.12, -0.02], [1.12, 0.06], [1.0, 0.06]])
    selected = dataset.select(kind="fieldline", seed_polygon=polygon, seed_coords=("R", "Z"))
    assert set(selected.ids) == set(trajectory_ids[:2])
    selected_summary = selected.summary()
    assert selected_summary["n_trajectories"] == 2
    assert selected_summary["by_kind"] == {"fieldline": 2}
    assert selected_summary["by_status"] == {"extendable": 2}

    resumed = selected.resume_fieldlines(
        field,
        target_phi=0.6,
        segment_phi_span=0.2,
        chunk_phi_span=0.2,
        storage="memory",
    )

    assert {record.trajectory_id for record in resumed} == set(trajectory_ids[:2])
    for trajectory_id in trajectory_ids[:2]:
        record = dataset.get(trajectory_id)
        assert record.n_segments == 3
        assert record.n_samples == 7
        assert np.isclose(record.end_independent, 0.6)
        loaded = dataset.load_fieldline_trajectory(trajectory_id)
        np.testing.assert_allclose(loaded.phi, np.linspace(0.0, 0.6, 7), atol=1.0e-14)

    unselected = dataset.get(trajectory_ids[2])
    assert unselected.n_segments == 1
    assert unselected.n_samples == 3
    assert np.isclose(unselected.end_independent, 0.2)

    segment_rows = dataset._conn.execute("SELECT * FROM segment ORDER BY trajectory_id, segment_index").fetchall()
    assert len(segment_rows) == 7
    assert all(int(row["n_samples"]) <= 3 for row in segment_rows)
    assert len(list((tmp_path / "trace_dataset" / "chunks").glob("*/*.npz"))) == 7
    resumed_rows = [
        row for row in segment_rows
        if row["trajectory_id"] in set(trajectory_ids[:2]) and int(row["segment_index"]) > 0
    ]
    assert all(np.isclose(float(row["end_independent"]) - float(row["start_independent"]), 0.2) for row in resumed_rows)
    summary = dataset.summary(kind="fieldline")
    assert summary["n_trajectories"] == 3
    assert summary["by_status"] == {"extendable": 3}
    assert summary["n_segments"] == 7
    assert summary["chunk_files"] == 7
    explain = dataset.explain_fieldline_request(
        trajectory_ids[0],
        field,
        target_phi=0.4,
    )
    assert not explain["matches"]
    assert explain["mismatches"][0]["field"] == "target_phi"
    dataset.close()


def test_trace_dataset_selectively_extends_real_wall_hits_by_seed_polygon(tmp_path):
    _skip_without_cyna()

    field = _outward_field()
    wall_phi, wall_R, wall_Z = _toroidal_wall()
    dataset = TraceDataset(tmp_path / "trace_dataset")
    seed_R = np.asarray([1.0, 1.08, 1.22])
    seed_Z = np.asarray([0.0, 0.0, 0.0])

    initial = dataset.trace_wall_hits_field(
        field,
        seed_R=seed_R,
        seed_Z=seed_Z,
        phi_start=0.0,
        max_turns=0,
        DPhi=0.01,
        wall_phi=wall_phi,
        wall_R_all=wall_R,
        wall_Z_all=wall_Z,
        directions=("plus", "minus"),
        batch_size=2,
    )

    assert len(initial) == 6
    assert {record.status for record in initial} == {"extendable"}
    assert dataset.events() == []

    polygon = np.asarray([[0.95, -0.04], [1.1, -0.04], [1.1, 0.04], [0.95, 0.04]])
    selected = dataset.select(kind="wall_hit", seed_polygon=polygon, seed_coords=("R", "Z"))
    assert len(selected) == 4

    resumed = selected.trace_wall_hits(
        field,
        max_turns=1,
        wall_phi=wall_phi,
        wall_R_all=wall_R,
        wall_Z_all=wall_Z,
        batch_size=2,
    )

    assert len(resumed) == 4
    assert {record.status for record in resumed} == {"terminal"}
    terminal_events = dataset.events(event_type="wall_hit")
    assert len(terminal_events) == 4
    assert all(event["terminal"] for event in terminal_events)
    assert all(event["payload"]["term"] == 1 for event in terminal_events)
    selected_summary = selected.summary(include_events=True)
    assert selected_summary["n_trajectories"] == 4
    assert selected_summary["events_by_type"] == {"wall_hit": 4}

    selected_ids = set(selected.ids)
    for record in dataset.select(kind="wall_hit"):
        if record.trajectory_id in selected_ids:
            assert record.n_segments == 2
            assert record.status == "terminal"
            assert np.isclose(record.end_independent, 1.0)
        else:
            assert record.n_segments == 1
            assert record.status == "extendable"
            assert np.isclose(record.end_independent, 0.0)

    assert len(list((tmp_path / "trace_dataset" / "chunks").glob("*/*.npz"))) == 10
    summary = dataset.summary(kind="wall_hit")
    assert summary["n_trajectories"] == 6
    assert summary["by_status"] == {"extendable": 2, "terminal": 4}
    assert summary["events_by_type"] == {"wall_hit": 4}
    assert summary["terminal_events"] == 4
    dataset.close()


def test_trace_dataset_wall_hit_resume_rejects_mismatched_wall(tmp_path):
    _skip_without_cyna()

    field = _outward_field()
    wall_phi, wall_R, wall_Z = _toroidal_wall()
    dataset = TraceDataset(tmp_path / "trace_dataset")
    initial = dataset.trace_wall_hits_field(
        field,
        seed_R=np.asarray([1.0]),
        seed_Z=np.asarray([0.0]),
        phi_start=0.0,
        max_turns=0,
        DPhi=0.01,
        wall_phi=wall_phi,
        wall_R_all=wall_R,
        wall_Z_all=wall_Z,
        directions=("plus",),
    )

    explain = dataset.explain_wall_hit_request(
        initial[0].trajectory_id,
        field,
        max_turns=1,
        wall_phi=wall_phi,
        wall_R_all=wall_R + 0.01,
        wall_Z_all=wall_Z,
    )
    assert not explain["matches"]
    assert explain["mismatches"][0]["field"] == "wall_signature"

    with pytest.raises(ValueError, match="wall signature"):
        dataset.trace_wall_hits_field(
            field,
            max_turns=1,
            wall_phi=wall_phi,
            wall_R_all=wall_R + 0.01,
            wall_Z_all=wall_Z,
            trajectory_ids=[initial[0].trajectory_id],
        )
    dataset.close()
