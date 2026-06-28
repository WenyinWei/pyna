import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import (
    BoundaryIslandOrbit,
    assemble_boundary_island_chains,
    boundary_island_edge_state_payload,
    boundary_recurrence_seed_candidates_field,
    boundary_seed_grid,
    boundary_wall_fractions,
    deduplicate_boundary_island_orbits,
    vector_field_cylind_from_field,
    find_boundary_island_fixed_points_field,
    refine_fixed_points_monodromy_span_field,
    trace_map_batch_span_field,
    trace_boundary_island_shapes_field,
    trace_boundary_island_chain_sections_span_field,
    trace_boundary_island_chain_dense_span_field,
    trace_poincare_sections_from_same_orbits_field,
    trace_fixed_point_orbit_sections_span_field,
    trace_fixed_point_orbit_dense_span_field,
    trace_fixed_point_orbits_span_field,
    trace_fixed_point_manifolds_field,
    trace_fixed_point_manifolds_multi_section_field,
)


def _skip_without_fixed_point_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.find_fixed_points_batch is None:
        pytest.skip("cyna fixed-point search is unavailable")


def _skip_without_cyna_field_handle():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind handle is unavailable")


def _hyperbolic_field(rate: float = 0.18, *, nfp: int = 1):
    axis_R = 1.0
    axis_Z = 0.0
    R = np.linspace(0.82, 1.18, 33)
    Z = np.linspace(-0.18, 0.18, 33)
    Phi = np.array([0.0])
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    dR_dphi = rate * (RR - axis_R)
    dZ_dphi = -rate * (ZZ - axis_Z)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=dR_dphi,
        BZ=dZ_dphi,
        BPhi=np.ones_like(RR),
        nfp=nfp,
    )


def _rotation_field(*, nfp: int = 1):
    axis_R = 1.0
    axis_Z = 0.0
    omega = 0.25
    R = np.linspace(0.82, 1.18, 33)
    Z = np.linspace(-0.18, 0.18, 33)
    Phi = np.array([0.0])
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    dR_dphi = -omega * (ZZ - axis_Z)
    dZ_dphi = omega * (RR - axis_R)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=dR_dphi,
        BZ=dZ_dphi,
        BPhi=np.ones_like(RR),
        nfp=nfp,
    )


def _orbit_fp(phi, R, Z, kind, map_power):
    if kind == "X":
        dpm = np.array([[2.0, 0.0], [0.0, 0.5]])
    else:
        dpm = np.array([[0.0, -1.0], [1.0, 0.0]])
    fp = FixedPoint(phi=float(phi), R=float(R), Z=float(Z), kind=kind, DPm=dpm)
    fp.map_power = int(map_power)
    fp.residual = 1.0e-9
    return fp


def _orbit_from_coords(coords, *, kind, source_index=0, map_span=np.pi):
    points = tuple(
        _orbit_fp(i * map_span, r, z, kind, len(coords))
        for i, (r, z) in enumerate(coords)
    )
    return BoundaryIslandOrbit(
        points=points,
        orbit_size=len(points),
        kind=kind,
        map_span=float(map_span),
        source_index=int(source_index),
        closure_residual=1.0e-9,
        map_count=len(points),
        alive=True,
    )


def test_boundary_seed_grid_tracks_wall_fraction():
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_R = 1.0 + 0.2 * np.cos(theta)
    wall_Z = 0.1 * np.sin(theta)

    seed_R, seed_Z = boundary_seed_grid(
        1.0,
        0.0,
        wall_R=wall_R,
        wall_Z=wall_Z,
        wall_fraction_min=0.5,
        wall_fraction_max=0.9,
        n_r=3,
        n_theta=24,
    )

    assert seed_R.shape == (72,)
    assert seed_Z.shape == (72,)
    assert np.all(seed_R >= wall_R.min() - 1.0e-12)
    assert np.all(seed_R <= wall_R.max() + 1.0e-12)
    assert np.all(seed_Z >= wall_Z.min() - 1.0e-12)
    assert np.all(seed_Z <= wall_Z.max() + 1.0e-12)


def test_boundary_wall_fractions_measure_axis_to_wall_radius():
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    wall_R = np.cos(theta)
    wall_Z = np.sin(theta)

    fractions = boundary_wall_fractions(
        0.0,
        0.0,
        [0.0, 0.5, -0.75, 0.0],
        [0.0, 0.0, 0.0, 0.9],
        wall_R,
        wall_Z,
    )

    np.testing.assert_allclose(fractions, [0.0, 0.5, 0.75, 0.9], atol=2.0e-4)


def test_boundary_chain_assembly_deduplicates_orbits_and_pairs_xo():
    o_coords = [(1.00, 0.00), (0.20, 0.30), (0.20, -0.30)]
    x_coords = [(1.10, 0.00), (0.25, 0.36), (0.25, -0.36)]
    shifted_o = [o_coords[1], o_coords[2], o_coords[0]]

    chains = assemble_boundary_island_chains(
        [
            _orbit_from_coords(o_coords, kind="O", source_index=0),
            _orbit_from_coords(shifted_o, kind="O", source_index=1),
            _orbit_from_coords(x_coords, kind="X", source_index=2),
        ],
        m=6,
        n=2,
        orbit_dedup_tol=1.0e-8,
    )

    assert len(chains) == 1
    chain = chains[0]
    assert chain.winding == (6, 2)
    assert chain.reduced_winding == (3, 1)
    assert len(chain.o_orbits) == 1
    assert len(chain.x_orbits) == 1
    assert len(chain.fixed_points) == 6
    assert chain.connected_component_count == 2
    assert chain.points_per_connected_component == 3
    assert chain.metadata["connected_component_count"] == 2
    assert chain.metadata["points_per_connected_component"] == 3
    assert {fp.metadata["chain_id"] for fp in chain.fixed_points} == {0}
    assert {fp.metadata["reduced_winding"] for fp in chain.fixed_points} == {(3, 1)}
    assert sorted(fp.metadata["point_index"] for fp in chain.o_orbits[0].points) == [0, 1, 2]


def test_deduplicate_boundary_island_orbits_assigns_cross_section_identity():
    coords = [(1.00, 0.00), (0.20, 0.30), (0.20, -0.30)]
    shifted = [coords[1], coords[2], coords[0]]

    orbits = deduplicate_boundary_island_orbits(
        [
            _orbit_from_coords(coords, kind="X", source_index=0),
            _orbit_from_coords(shifted, kind="X", source_index=1),
        ],
        orbit_dedup_tol=1.0e-8,
        start_orbit_id=7,
        chain_id=3,
        winding=(3, 1),
        reduced_winding=(3, 1),
    )

    assert len(orbits) == 1
    orbit = orbits[0]
    assert orbit.orbit_id == 7
    assert orbit.chain_id == 3
    assert orbit.metadata["same_orbit_key"] == "chain=3:orbit=7:kind=X"
    assert {fp.metadata["same_orbit_key"] for fp in orbit.points} == {"chain=3:orbit=7:kind=X"}
    assert sorted(fp.metadata["point_index"] for fp in orbit.points) == [0, 1, 2]


def test_trace_fixed_point_orbits_span_field_uses_batch_outputs(monkeypatch):
    calls = []

    def fake_trace(field, R0, Z0, phi_start, map_span, N_steps, DPhi, **kwargs):
        calls.append((float(phi_start), np.asarray(R0).copy(), np.asarray(Z0).copy()))
        assert N_steps == 3
        flat_R = np.asarray([2.0, 3.0, 1.0] * len(R0), dtype=float)
        flat_Z = np.asarray([0.0, 0.0, 0.0] * len(R0), dtype=float)
        return np.full(len(R0), 3, dtype=int), flat_R, flat_Z

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_map_batch_span_field",
        fake_trace,
    )
    fp0 = _orbit_fp(0.0, 1.0, 0.0, "O", 3)
    fp1 = _orbit_fp(0.25, 1.0, 0.0, "O", 3)

    orbits = trace_fixed_point_orbits_span_field(
        object(),
        [fp0, fp1],
        map_span=np.pi,
        DPhi=0.1,
    )

    assert len(orbits) == 2
    assert [call[0] for call in calls] == [0.0, 0.25]
    np.testing.assert_allclose(orbits[0].R, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(orbits[0].phi, [0.0, np.pi, 2.0 * np.pi])
    assert orbits[0].closure_residual == pytest.approx(0.0)
    assert orbits[0].alive is True
    assert orbits[0].points[1].metadata["map_span"] == pytest.approx(np.pi)

    calls.clear()
    orbits_unique = trace_fixed_point_orbits_span_field(
        object(),
        [fp0, _orbit_fp(0.0, 1.0, 0.0, "O", 3)],
        map_span=np.pi,
        DPhi=0.1,
        deduplicate=True,
        start_orbit_id=5,
        chain_id=2,
        winding=(3, 1),
        reduced_winding=(3, 1),
    )
    assert len(orbits_unique) == 1
    assert len(calls) == 1
    assert orbits_unique[0].orbit_id == 5
    assert orbits_unique[0].metadata["same_orbit_key"] == "chain=2:orbit=5:kind=O"


def test_trace_poincare_sections_from_same_orbits_uses_multi_trace(monkeypatch):
    calls = []

    def fake_multi(field, R0, Z0, phi_sections, N_turns, DPhi, wall_R, wall_Z, **kwargs):
        calls.append((np.asarray(R0).copy(), np.asarray(phi_sections).copy(), int(N_turns), kwargs))
        n_seed = len(R0)
        n_sec = len(phi_sections)
        counts = np.full((n_seed, n_sec), int(N_turns), dtype=int)
        flat_R = []
        flat_Z = []
        for i in range(n_seed):
            for j in range(n_sec):
                for k in range(int(N_turns)):
                    flat_R.append(100.0 * i + 10.0 * j + k)
                    flat_Z.append(-100.0 * i - 10.0 * j - k)
        return counts, np.asarray(flat_R), np.asarray(flat_Z)

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_poincare_multi_batch_field",
        fake_multi,
    )

    traces = trace_poincare_sections_from_same_orbits_field(
        object(),
        [1.0, 2.0],
        [0.0, 0.1],
        [0.0, 0.5, 1.0],
        N_turns=2,
        DPhi=0.1,
        wall_R=[0.0, 3.0, 3.0, 0.0],
        wall_Z=[-1.0, -1.0, 1.0, 1.0],
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][1], [0.0, 0.5, 1.0])
    assert traces.metadata["trace_source"] == "same_orbit_multi_section"
    np.testing.assert_array_equal(traces.counts, np.full((2, 3), 2))
    R_sec, Z_sec, seed_index = traces.section_points(1)
    np.testing.assert_allclose(R_sec, [10.0, 11.0, 110.0, 111.0])
    np.testing.assert_allclose(Z_sec, [-10.0, -11.0, -110.0, -111.0])
    np.testing.assert_array_equal(seed_index, [0, 0, 1, 1])


def test_trace_fixed_point_orbit_sections_uses_one_orbit(monkeypatch):
    orbit_calls = []

    def fail_map_trace(*args, **kwargs):
        raise AssertionError("section cuts should come from one dense orbit, not per-section map traces")

    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        orbit_calls.append((float(phi_start), float(phi_end), kwargs))
        phi = np.asarray([
            0.0,
            0.25 * np.pi,
            np.pi,
            1.25 * np.pi,
            2.0 * np.pi,
            2.25 * np.pi,
            3.0 * np.pi,
        ])
        R = np.asarray([1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 1.00001])
        Z = np.zeros_like(R)
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_map_batch_span_field",
        fail_map_trace,
    )
    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    base = _orbit_from_coords(
        [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        kind="X",
        map_span=np.pi,
    )
    base = BoundaryIslandOrbit(
        points=base.points,
        orbit_size=base.orbit_size,
        kind=base.kind,
        map_span=base.map_span,
        source_index=base.source_index,
        closure_residual=base.closure_residual,
        map_count=base.map_count,
        alive=True,
        metadata={"source_phi": 0.0},
    )

    sections = trace_fixed_point_orbit_sections_span_field(
        object(),
        base,
        [0.0, 0.25 * np.pi],
        DPhi=0.1,
        dphi_out=0.05,
        section_dedup_tol=1.0e-3,
    )

    assert list(sections) == [0.0, 0.25 * np.pi]
    assert len(orbit_calls) == 1
    assert orbit_calls[0][0] == pytest.approx(0.0)
    assert orbit_calls[0][1] == pytest.approx(3.0 * np.pi)
    assert orbit_calls[0][2]["dphi_out"] == pytest.approx(0.05)
    np.testing.assert_allclose(sections[0.0].R, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sections[0.25 * np.pi].R, [1.1, 2.1, 3.1])
    assert [fp.phi for fp in sections[0.25 * np.pi].points] == [0.25 * np.pi] * 3
    assert [fp.metadata["orbit_point_index"] for fp in sections[0.25 * np.pi].points] == [0, 1, 2]
    assert [fp.metadata["map_order_index"] for fp in sections[0.25 * np.pi].points] == [0, 1, 2]
    assert [fp.metadata["poincare_map_power"] for fp in sections[0.25 * np.pi].points] == [3, 3, 3]
    assert sections[0.0].metadata["raw_crossing_count"] == 4
    assert sections[0.0].metadata["dedup_crossing_count"] == 3
    assert sections[0.0].metadata["expected_crossing_count"] == 3
    assert sections[0.0].metadata["complete_crossing_count"] is True


def test_section_orbit_keeps_close_distinct_orbit_points(monkeypatch):
    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        phi = np.asarray([
            0.0,
            0.5 * np.pi,
            np.pi,
            1.5 * np.pi,
            2.0 * np.pi,
            2.5 * np.pi,
            3.0 * np.pi,
        ])
        R = np.asarray([2.0, 1.0000, 3.0, 1.0002, 4.0, 1.0004, 2.00001])
        Z = np.asarray([0.0, 0.0, 0.1, 0.0002, 0.2, 0.0004, 0.0])
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    base = _orbit_from_coords(
        [(2.0, 0.0), (3.0, 0.1), (4.0, 0.2)],
        kind="O",
        map_span=np.pi,
    )
    base = BoundaryIslandOrbit(
        points=base.points,
        orbit_size=base.orbit_size,
        kind=base.kind,
        map_span=base.map_span,
        source_index=base.source_index,
        closure_residual=base.closure_residual,
        map_count=base.map_count,
        alive=True,
        metadata={"source_phi": 0.0},
    )

    sections = trace_fixed_point_orbit_sections_span_field(
        object(),
        base,
        [0.0, 0.5 * np.pi],
        DPhi=0.1,
        dphi_out=0.05,
        section_dedup_tol=1.0e-3,
    )

    np.testing.assert_allclose(sections[0.0].R, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(sections[0.5 * np.pi].R, [1.0000, 1.0002, 1.0004])
    assert len(sections[0.0].points) == 3
    assert len(sections[0.5 * np.pi].points) == 3
    assert sections[0.5 * np.pi].metadata["raw_crossing_count"] == 3
    assert sections[0.5 * np.pi].metadata["dedup_crossing_count"] == 3


def test_section_orbit_dedups_closure_endpoint_using_orbit_residual(monkeypatch):
    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        phi = np.asarray([0.0, np.pi, 2.0 * np.pi, 3.0 * np.pi])
        R = np.asarray([1.0, 2.0, 3.0, 1.015])
        Z = np.zeros_like(R)
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    base = _orbit_from_coords(
        [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        kind="X",
        map_span=np.pi,
    )
    base = BoundaryIslandOrbit(
        points=base.points,
        orbit_size=base.orbit_size,
        kind=base.kind,
        map_span=base.map_span,
        source_index=base.source_index,
        closure_residual=0.02,
        map_count=base.map_count,
        alive=True,
        metadata={"source_phi": 0.0},
    )

    sections = trace_fixed_point_orbit_sections_span_field(
        object(),
        base,
        [0.0],
        DPhi=0.1,
        dphi_out=0.05,
        section_dedup_tol=1.0e-3,
    )

    np.testing.assert_allclose(sections[0.0].R, [1.0, 2.0, 3.0])
    assert sections[0.0].metadata["raw_crossing_count"] == 4
    assert sections[0.0].metadata["dedup_crossing_count"] == 3
    assert sections[0.0].metadata["section_dedup_tol"] == pytest.approx(1.0e-3)
    assert sections[0.0].metadata["effective_section_dedup_tol"] == pytest.approx(0.021)


def test_boundary_chain_section_counts_are_consistent(monkeypatch):
    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        phi = np.asarray([
            0.0,
            0.25 * np.pi,
            np.pi,
            1.25 * np.pi,
            2.0 * np.pi,
            2.25 * np.pi,
            3.0 * np.pi,
        ])
        R = np.asarray([R0, R0 + 0.1, R0 + 1.0, R0 + 1.1, R0 + 2.0, R0 + 2.1, R0])
        Z = np.asarray([Z0, Z0 + 0.1, Z0, Z0 + 0.1, Z0, Z0 + 0.1, Z0])
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    chains = assemble_boundary_island_chains(
        [
            _orbit_from_coords([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], kind="O"),
            _orbit_from_coords([(1.2, 0.0), (2.2, 0.0), (3.2, 0.0)], kind="X"),
        ],
        m=3,
        n=1,
    )

    sections = trace_boundary_island_chain_sections_span_field(
        object(),
        chains[0],
        [0.0, 0.25 * np.pi],
        DPhi=0.1,
        dphi_out=0.05,
    )

    for section_chain in sections.values():
        assert [len(orbit.points) for orbit in section_chain.orbits] == [3, 3]
        assert section_chain.metadata["require_complete_sections"] is True
        assert section_chain.metadata["section_orbit_count_by_orbit"]


def test_trace_fixed_point_orbit_dense_span_outputs_continuous_geometry(monkeypatch, tmp_path):
    calls = []

    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        calls.append((float(R0), float(Z0), float(phi_start), float(phi_end), kwargs))
        phi = np.linspace(float(phi_start), float(phi_end), 5)
        R = np.asarray([R0, R0 + 0.1, R0 + 0.2, R0 + 0.1, R0], dtype=float)
        Z = np.asarray([Z0, Z0 + 0.2, Z0, Z0 - 0.2, Z0], dtype=float)
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    base = _orbit_from_coords(
        [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        kind="O",
        map_span=0.5,
    )

    dense = trace_fixed_point_orbit_dense_span_field(
        object(),
        base,
        DPhi=0.1,
        dphi_out=0.05,
    )

    assert calls[0][2] == pytest.approx(0.0)
    assert calls[0][3] == pytest.approx(1.5)
    assert calls[0][4]["dphi_out"] == pytest.approx(0.05)
    assert dense.n_samples == 5
    assert dense.complete is True
    assert dense.closure_residual == pytest.approx(0.0)
    assert dense.xyz.shape == (5, 3)
    assert len(dense.section_points) == 3
    arrays = dense.as_arrays(include_xyz=True)
    assert set(["phi", "R", "Z", "alive", "x", "y", "z"]).issubset(arrays)

    path = tmp_path / "dense_orbit.npz"
    dense.save_npz(path)
    saved = np.load(path)
    np.testing.assert_allclose(saved["R"], dense.R)
    assert int(saved["orbit_size"]) == 3


def test_trace_boundary_island_chain_dense_span_traces_each_orbit(monkeypatch):
    def fake_orbit(field, R0, Z0, phi_start, phi_end, DPhi, **kwargs):
        phi = np.linspace(float(phi_start), float(phi_end), 3)
        R = np.asarray([R0, R0 + 0.1, R0], dtype=float)
        Z = np.asarray([Z0, Z0 + 0.1, Z0], dtype=float)
        DP = np.repeat(np.eye(2)[None, :, :], len(phi), axis=0)
        alive = np.ones(len(phi), dtype=bool)
        return R, Z, phi, DP, alive

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.trace_orbit_along_phi_field",
        fake_orbit,
    )
    chains = assemble_boundary_island_chains(
        [
            _orbit_from_coords([(1.0, 0.0), (2.0, 0.0)], kind="O"),
            _orbit_from_coords([(1.1, 0.0), (2.1, 0.0)], kind="X"),
        ],
        m=2,
        n=1,
    )

    dense_chain = trace_boundary_island_chain_dense_span_field(
        object(),
        chains[0],
        DPhi=0.1,
    )

    assert len(dense_chain.dense_orbits) == 2
    assert len(dense_chain.o_orbits) == 1
    assert len(dense_chain.x_orbits) == 1
    assert dense_chain.as_arrays()["chain_id"] == 0


def test_cyna_field_handle_traces_span_map_like_object_wrapper():
    _skip_without_cyna_field_handle()
    field = _rotation_field()
    R0 = np.array([1.04, 0.98])
    Z0 = np.array([0.01, -0.03])
    map_span = 0.7

    counts_obj, R_obj, Z_obj = trace_map_batch_span_field(
        field,
        R0,
        Z0,
        0.0,
        map_span,
        5,
        0.02,
        n_threads=1,
    )
    handle = vector_field_cylind_from_field(field)
    counts_handle, R_handle, Z_handle = handle.trace_map_batch_span(
        R0,
        Z0,
        0.0,
        map_span,
        5,
        0.02,
        n_threads=1,
    )

    np.testing.assert_array_equal(counts_obj, counts_handle)
    np.testing.assert_allclose(R_obj, R_handle)
    np.testing.assert_allclose(Z_obj, Z_handle)

    wall = np.asarray(
        [
            [0.75, -0.25],
            [1.25, -0.25],
            [1.25, 0.25],
            [0.75, 0.25],
        ],
        dtype=float,
    )
    wall_R_view = wall[:, 0]
    wall_Z_view = wall[:, 1]
    assert not wall_R_view.flags.c_contiguous
    counts_obj, R_obj, Z_obj = trace_map_batch_span_field(
        field,
        R0,
        Z0,
        0.0,
        map_span,
        5,
        0.02,
        wall_R=wall_R_view,
        wall_Z=wall_Z_view,
        n_threads=1,
    )
    counts_handle, R_handle, Z_handle = handle.trace_map_batch_span(
        R0,
        Z0,
        0.0,
        map_span,
        5,
        0.02,
        wall_R_view.copy(),
        wall_Z_view.copy(),
        n_threads=1,
    )

    np.testing.assert_array_equal(counts_obj, counts_handle)
    np.testing.assert_allclose(R_obj, R_handle)
    np.testing.assert_allclose(Z_obj, Z_handle)


def test_boundary_fixed_point_search_returns_plot_payload():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field(nfp=2)

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        radii=(0.04, 0.08, 0.12),
        n_theta=16,
        DPhi=0.02,
        max_iter=50,
        tol=1.0e-10,
        residual_tol=1.0e-8,
        dedup_tol=1.0e-4,
        n_threads=1,
    )

    assert result.diagnostics["n_seeds"] == 48
    assert len(result.fixed_points) == 1
    fp = result.fixed_points[0]
    assert fp.kind == "X"
    assert fp.field_period == pytest.approx(np.pi)
    assert fp.metadata["field_period"] == pytest.approx(np.pi)
    assert fp.metadata["monodromy_field_period"] == pytest.approx(np.pi)
    np.testing.assert_allclose([fp.R, fp.Z], [1.0, 0.0], atol=1.0e-7)
    assert fp.residual < 1.0e-8
    payload_fp = result.fp_by_sec[0.0]["xpts"][0]
    assert payload_fp.kind == "X"
    assert payload_fp.stable_eigenvec is not None
    assert payload_fp.unstable_eigenvec is not None
    assert abs(payload_fp.stable_eigenvec[1]) > 0.99
    assert abs(payload_fp.unstable_eigenvec[0]) > 0.99
    np.testing.assert_allclose(payload_fp.DPm, fp.DPm)


def test_lower_map_power_filter_removes_axis_from_higher_map_power_search():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(2,),
        radii=(0.04, 0.08),
        n_theta=12,
        DPhi=0.02,
        max_iter=50,
        tol=1.0e-10,
        residual_tol=1.0e-8,
        dedup_tol=1.0e-4,
        lower_map_power_tol=1.0e-7,
        n_threads=1,
    )

    assert result.fixed_points == ()
    assert result.fp_by_sec[0.0] == {"xpts": [], "opts": []}


def test_recurrence_candidates_can_seed_fixed_point_search():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()

    candidates = boundary_recurrence_seed_candidates_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        seed_R=np.array([1.0, 1.0]),
        seed_Z=np.array([0.12, -0.12]),
        N_turns=10,
        DPhi=0.02,
        candidates_per_map_power=6,
        candidate_dedup_tol=1.0e-5,
    )

    seed_R, seed_Z = candidates.seeds_for_map_power(1)
    assert 0 < seed_R.size <= 6
    assert np.min(np.hypot(seed_R - 1.0, seed_Z)) < 1.0e-3

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        radii=(0.12,),
        n_theta=4,
        candidate_strategy="recurrence",
        recurrence_turns=10,
        recurrence_candidates_per_map_power=6,
        candidate_dedup_tol=1.0e-5,
        DPhi=0.02,
        max_iter=50,
        tol=1.0e-10,
        residual_tol=1.0e-8,
        dedup_tol=1.0e-4,
        n_threads=1,
    )

    assert result.diagnostics["candidate_strategy"] == "recurrence"
    assert len(result.fixed_points) == 1
    np.testing.assert_allclose(
        [result.fixed_points[0].R, result.fixed_points[0].Z],
        [1.0, 0.0],
        atol=1.0e-7,
    )


def test_recurrence_candidates_can_prefer_outer_wall_fraction(monkeypatch):
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    wall_R = np.cos(theta)
    wall_Z = np.sin(theta)

    def fake_trace(*args, **kwargs):
        _ = (args, kwargs)
        return [(
            np.asarray([0.20, 0.2001, 0.95, 0.97, 0.70, 0.705]),
            np.zeros(6),
        )]

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain._trace_poincare_points_field",
        fake_trace,
    )

    candidates = boundary_recurrence_seed_candidates_field(
        object(),
        0.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        seed_R=np.asarray([0.9]),
        seed_Z=np.asarray([0.0]),
        wall_R=wall_R,
        wall_Z=wall_Z,
        N_turns=5,
        DPhi=0.02,
        recurrence_tol=0.03,
        candidate_order="outer",
        candidate_wall_fraction_min=0.9,
        candidates_per_map_power=1,
        candidate_dedup_tol=1.0e-6,
    )

    seed_R, seed_Z = candidates.seeds_for_map_power(1)
    np.testing.assert_allclose(seed_R, [0.95], atol=2.0e-4)
    np.testing.assert_allclose(seed_Z, [0.0], atol=1.0e-12)
    assert candidates.diagnostics["candidate_order"] == "outer"
    assert candidates.diagnostics["accepted_candidate_wall_fraction_by_map_power"][1]["min"] > 0.94

    unfiltered = boundary_recurrence_seed_candidates_field(
        object(),
        0.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        seed_R=np.asarray([0.9]),
        seed_Z=np.asarray([0.0]),
        wall_R=wall_R,
        wall_Z=wall_Z,
        N_turns=5,
        DPhi=0.02,
        recurrence_tol=0.03,
        candidate_order="outer",
        candidates_per_map_power=1,
        candidate_dedup_tol=1.0e-6,
    )
    seed_R, _seed_Z = unfiltered.seeds_for_map_power(1)
    np.testing.assert_allclose(seed_R, [0.95], atol=2.0e-4)
    assert unfiltered.diagnostics["best_residual_by_map_power"][1] == pytest.approx(1.0e-4)


def test_field_period_fixed_point_search_uses_arbitrary_phi_span():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        radii=(0.12,),
        n_theta=4,
        candidate_strategy="recurrence",
        recurrence_turns=10,
        recurrence_candidates_per_map_power=6,
        field_period=np.pi,
        python_trust_radius=0.04,
        DPhi=0.02,
        max_iter=40,
        tol=1.0e-9,
        residual_tol=1.0e-7,
        dedup_tol=1.0e-4,
        n_threads=1,
    )

    assert result.diagnostics["field_period"] == pytest.approx(np.pi)
    assert len(result.fixed_points) == 1
    fp = result.fixed_points[0]
    assert fp.kind == "X"
    np.testing.assert_allclose([fp.R, fp.Z], [1.0, 0.0], atol=1.0e-7)


def test_trace_fixed_point_manifolds_returns_plot_payload():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()
    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        map_powers=(1,),
        radii=(0.04,),
        n_theta=8,
        DPhi=0.02,
        max_iter=50,
        tol=1.0e-10,
        residual_tol=1.0e-8,
        dedup_tol=1.0e-4,
        n_threads=1,
    )

    manifolds = trace_fixed_point_manifolds_field(
        field,
        result.fp_by_sec[0.0]["xpts"],
        phi_section=0.0,
        N_turns=4,
        DPhi=0.02,
        eps_min=1.0e-4,
        eps_max=3.0e-4,
        n_eps=3,
    )

    assert len(manifolds) == 1
    man = manifolds[0]
    assert {"u_R", "u_Z", "s_R", "s_Z"}.issubset(man)
    assert man["u_R"].size > 0
    assert man["s_R"].size > 0
    assert np.all(np.isfinite(man["u_R"]))
    assert np.all(np.isfinite(man["s_Z"]))
    assert man["origin_R"] == pytest.approx(result.fp_by_sec[0.0]["xpts"][0].R)
    assert man["origin_Z"] == pytest.approx(result.fp_by_sec[0.0]["xpts"][0].Z)
    assert "map_order_index" in man
    assert man["u_seed_R"].shape == man["u_seed_Z"].shape
    assert man["s_seed_R"].shape == man["s_seed_Z"].shape

    manifolds_with_s = trace_fixed_point_manifolds_field(
        field,
        result.fp_by_sec[0.0]["xpts"],
        phi_section=0.0,
        N_turns=4,
        DPhi=0.02,
        eps_min=1.0e-4,
        eps_max=3.0e-4,
        n_eps=3,
        include_arclength=True,
    )
    man_s = manifolds_with_s[0]
    assert man_s["arclength_coordinate"] == "poloidal_RZ_from_xpoint"
    assert man_s["u_lpol"].shape == man_s["u_R"].shape
    assert man_s["s_lpol"].shape == man_s["s_R"].shape
    assert np.all(man_s["u_lpol"] >= 0.0)


def test_trace_fixed_point_manifolds_uses_eigenvalue_geometric_seed_segment():
    _skip_without_cyna_field_handle()
    field = _hyperbolic_field()
    fp = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([16.0, 1.0 / 16.0]),
    )

    manifolds = trace_fixed_point_manifolds_field(
        field,
        [fp],
        phi_section=0.0,
        N_turns=1,
        DPhi=0.02,
        eps_min=1.0e-5,
        eps_max=1.0e-3,
        n_eps=4,
        include_arclength=True,
    )

    assert len(manifolds) == 1
    man = manifolds[0]
    assert man["seed_spacing"] == "eigenvalue_geometric"
    assert man["unstable_expansion"] == pytest.approx(16.0)
    assert man["stable_backward_expansion"] == pytest.approx(16.0)
    assert man["u_seed_ratio"] == pytest.approx(2.0)
    assert man["s_seed_ratio"] == pytest.approx(2.0)
    u_dist = np.unique(np.round(man["u_seed_distance"], decimals=14))
    s_dist = np.unique(np.round(man["s_seed_distance"], decimals=14))
    np.testing.assert_allclose(u_dist, 1.0e-5 * 2.0 ** np.arange(4))
    np.testing.assert_allclose(s_dist, 1.0e-5 * 2.0 ** np.arange(4))
    assert man["u_seed_next_distance"] > float(np.max(man["u_seed_distance"]))
    assert man["s_seed_next_distance"] > float(np.max(man["s_seed_distance"]))
    assert man["u_seed_next_distance"] / float(np.max(man["u_seed_distance"])) == pytest.approx(2.0)
    assert man["s_seed_next_distance"] / float(np.max(man["s_seed_distance"])) == pytest.approx(2.0)
    assert man["u_generation"].shape == man["u_R"].shape
    assert man["s_generation"].shape == man["s_R"].shape
    assert set(np.unique(man["u_generation"][:4])) == {0}
    assert np.all(np.diff(man["u_lpol"][:4]) > 0.0)
    assert man["seed_generation_order"] == "side_then_generation_then_distance"
    u_gen, u_counts = np.unique(man["u_generation"], return_counts=True)
    s_gen, s_counts = np.unique(man["s_generation"], return_counts=True)
    assert dict(zip(u_gen.tolist(), u_counts.tolist())) == {0: 8, 1: 8}
    assert dict(zip(s_gen.tolist(), s_counts.tolist())) == {0: 8, 1: 8}
    assert set(man["u_point_seed_order"][man["u_generation"] == 1]) == {0, 1, 2, 3}
    assert set(man["s_point_seed_order"][man["s_generation"] == 1]) == {0, 1, 2, 3}


def test_trace_fixed_point_manifolds_exact_linear_map_orders_branches(monkeypatch):
    expansion = 4.0
    origin_R = 1.0
    origin_Z = 0.0

    def fake_trace_map_points(
        field,
        seed_R,
        seed_Z,
        phi_section,
        *,
        N_turns,
        map_span,
        DPhi,
        wall_R=None,
        wall_Z=None,
        extend_phi=True,
        fd_eps=1.0e-4,
    ):
        traces = []
        for r, z in zip(np.asarray(seed_R, dtype=float), np.asarray(seed_Z, dtype=float)):
            dR = float(r) - origin_R
            dZ = float(z) - origin_Z
            R_vals = []
            Z_vals = []
            for generation in range(1, int(N_turns) + 1):
                scale = expansion ** generation
                R_vals.append(origin_R + scale * dR)
                Z_vals.append(origin_Z + scale * dZ)
            traces.append((np.asarray(R_vals, dtype=float), np.asarray(Z_vals, dtype=float)))
        return traces

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain._trace_map_points_field",
        fake_trace_map_points,
    )
    fp = FixedPoint(
        phi=0.0,
        R=origin_R,
        Z=origin_Z,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fp.metadata.update({"orbit_id": 3, "map_order_index": 2, "same_orbit_key": "chain=0:orbit=3:kind=X"})

    manifolds = trace_fixed_point_manifolds_field(
        object(),
        [fp],
        phi_section=0.0,
        N_turns=3,
        field_period=1.0,
        DPhi=0.1,
        eps_min=1.0e-5,
        eps_max=1.0e-3,
        n_eps=4,
        include_arclength=True,
    )

    assert len(manifolds) == 1
    man = manifolds[0]
    assert man["kind"] == "X"
    assert man["orbit_id"] == 3
    assert man["map_order_index"] == 2
    expected_dist = np.unique(np.round(man["u_seed_distance"], decimals=14))

    def branch_offsets(prefix: str, coord: str, generation: int, side_sign: float) -> np.ndarray:
        gen = np.asarray(man[f"{prefix}_generation"])
        side = np.asarray(man[f"{prefix}_point_side"])
        order = np.asarray(man[f"{prefix}_point_seed_order"])
        values = np.asarray(man[coord])
        mask = (gen == generation) & (side == side_sign)
        idx = np.argsort(order[mask])
        base = origin_R if coord.endswith("_R") else origin_Z
        return values[mask][idx] - base

    for generation in range(4):
        factor = expansion ** generation
        np.testing.assert_allclose(
            branch_offsets("u", "u_R", generation, 1.0),
            factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("u", "u_R", generation, -1.0),
            -factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("s", "s_Z", generation, 1.0),
            factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("s", "s_Z", generation, -1.0),
            -factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(branch_offsets("u", "u_Z", generation, 1.0), 0.0, atol=1.0e-15)
        np.testing.assert_allclose(branch_offsets("s", "s_R", generation, 1.0), 0.0, atol=1.0e-15)

    for prefix in ("u", "s"):
        side = np.asarray(man[f"{prefix}_point_side"])
        lpol = np.asarray(man[f"{prefix}_lpol"])
        for side_sign in (-1.0, 1.0):
            branch_lpol = lpol[side == side_sign]
            assert np.all(np.diff(branch_lpol) > 0.0)


def test_trace_fixed_point_manifolds_handles_orientation_reversing_saddle(monkeypatch):
    expansion = 3.0
    origin_R = 1.0
    origin_Z = 0.0

    def fake_trace_map_points(
        field,
        seed_R,
        seed_Z,
        phi_section,
        *,
        N_turns,
        map_span,
        DPhi,
        wall_R=None,
        wall_Z=None,
        extend_phi=True,
        fd_eps=1.0e-4,
    ):
        traces = []
        forward = float(map_span) > 0.0
        for r, z in zip(np.asarray(seed_R, dtype=float), np.asarray(seed_Z, dtype=float)):
            dR = float(r) - origin_R
            dZ = float(z) - origin_Z
            R_vals = []
            Z_vals = []
            for _generation in range(1, int(N_turns) + 1):
                if forward:
                    dR, dZ = -expansion * dR, -dZ / expansion
                else:
                    dR, dZ = -dR / expansion, -expansion * dZ
                R_vals.append(origin_R + dR)
                Z_vals.append(origin_Z + dZ)
            traces.append((np.asarray(R_vals, dtype=float), np.asarray(Z_vals, dtype=float)))
        return traces

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain._trace_map_points_field",
        fake_trace_map_points,
    )
    fp = FixedPoint(
        phi=0.0,
        R=origin_R,
        Z=origin_Z,
        kind="X",
        DPm=np.diag([-expansion, -1.0 / expansion]),
    )

    manifolds = trace_fixed_point_manifolds_field(
        object(),
        [fp],
        phi_section=0.0,
        N_turns=3,
        field_period=1.0,
        DPhi=0.1,
        eps_min=1.0e-5,
        eps_max=1.0e-3,
        n_eps=4,
        include_arclength=True,
    )

    assert len(manifolds) == 1
    man = manifolds[0]
    assert man["unstable_orientation_reversing"] is True
    assert man["stable_orientation_reversing"] is True
    expected_dist = np.unique(np.round(man["u_seed_distance"], decimals=14))

    def branch_offsets(prefix: str, coord: str, generation: int, side_sign: float) -> np.ndarray:
        gen = np.asarray(man[f"{prefix}_generation"])
        side = np.asarray(man[f"{prefix}_point_side"])
        order = np.asarray(man[f"{prefix}_point_seed_order"])
        values = np.asarray(man[coord])
        mask = (gen == generation) & (side == side_sign)
        idx = np.argsort(order[mask])
        base = origin_R if coord.endswith("_R") else origin_Z
        return values[mask][idx] - base

    for generation in range(4):
        factor = expansion ** generation
        np.testing.assert_allclose(
            branch_offsets("u", "u_R", generation, 1.0),
            factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("u", "u_R", generation, -1.0),
            -factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("s", "s_Z", generation, 1.0),
            factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            branch_offsets("s", "s_Z", generation, -1.0),
            -factor * expected_dist,
            rtol=1.0e-9,
            atol=1.0e-15,
        )

    for prefix in ("u", "s"):
        side = np.asarray(man[f"{prefix}_point_side"])
        lpol = np.asarray(man[f"{prefix}_lpol"])
        for side_sign in (-1.0, 1.0):
            branch_lpol = lpol[side == side_sign]
            assert np.all(np.diff(branch_lpol) > 0.0)


def test_trace_fixed_point_manifolds_exact_nonlinear_area_preserving_map(monkeypatch):
    origin_R = 1.0
    origin_Z = 0.0
    expansion = 3.0
    shear_x = 8.0
    shear_y = 8.0

    def forward_map(R, Z):
        x = np.asarray(R, dtype=float) - origin_R
        y = np.asarray(Z, dtype=float) - origin_Z
        x1 = x + shear_x * y * y
        y1 = y
        x2 = expansion * x1
        y2 = y1 / expansion
        x3 = x2
        y3 = y2 + shear_y * x2 * x2
        return origin_R + x3, origin_Z + y3

    def inverse_map(R, Z):
        x3 = np.asarray(R, dtype=float) - origin_R
        y3 = np.asarray(Z, dtype=float) - origin_Z
        x2 = x3
        y2 = y3 - shear_y * x2 * x2
        x1 = x2 / expansion
        y1 = expansion * y2
        x0 = x1 - shear_x * y1 * y1
        y0 = y1
        return origin_R + x0, origin_Z + y0

    def fake_trace_map_points(
        field,
        seed_R,
        seed_Z,
        phi_section,
        *,
        N_turns,
        map_span,
        DPhi,
        wall_R=None,
        wall_Z=None,
        extend_phi=True,
        fd_eps=1.0e-4,
    ):
        traces = []
        step = forward_map if float(map_span) > 0.0 else inverse_map
        for r, z in zip(np.asarray(seed_R, dtype=float), np.asarray(seed_Z, dtype=float)):
            R_vals = []
            Z_vals = []
            rr = float(r)
            zz = float(z)
            for _generation in range(int(N_turns)):
                rr, zz = step(rr, zz)
                R_vals.append(float(rr))
                Z_vals.append(float(zz))
            traces.append((np.asarray(R_vals, dtype=float), np.asarray(Z_vals, dtype=float)))
        return traces

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain._trace_map_points_field",
        fake_trace_map_points,
    )
    fp = FixedPoint(
        phi=0.0,
        R=origin_R,
        Z=origin_Z,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fp.metadata.update({
        "orbit_id": 4,
        "map_order_index": 1,
        "same_orbit_key": "chain=0:orbit=4:kind=X",
    })

    manifolds = trace_fixed_point_manifolds_field(
        object(),
        [fp],
        phi_section=0.0,
        N_turns=4,
        field_period=1.0,
        DPhi=0.1,
        eps_min=1.0e-4,
        eps_max=2.0e-2,
        n_eps=32,
        include_arclength=True,
    )

    assert len(manifolds) == 1
    man = manifolds[0]
    assert man["seed_spacing"] == "eigenvalue_geometric"
    assert man["unstable_expansion"] == pytest.approx(expansion)
    assert man["stable_backward_expansion"] == pytest.approx(expansion)
    for prefix in ("u", "s"):
        seed_distance = np.asarray(man[f"{prefix}_seed_distance"])
        seed_next_distance = float(man[f"{prefix}_seed_next_distance"])
        seed_ratio = float(man[f"{prefix}_seed_ratio"])
        assert seed_next_distance > float(np.max(seed_distance))
        assert seed_next_distance / float(np.max(seed_distance)) == pytest.approx(seed_ratio)

    def branch_points(prefix: str, generation: int, side_sign: float) -> tuple[np.ndarray, np.ndarray]:
        gen = np.asarray(man[f"{prefix}_generation"])
        side = np.asarray(man[f"{prefix}_point_side"])
        order = np.asarray(man[f"{prefix}_point_seed_order"])
        mask = (gen == generation) & (side == side_sign)
        idx = np.argsort(order[mask])
        return np.asarray(man[f"{prefix}_R"])[mask][idx], np.asarray(man[f"{prefix}_Z"])[mask][idx]

    for side_sign in (-1.0, 1.0):
        for generation in range(4):
            u_R, u_Z = branch_points("u", generation, side_sign)
            u_next_R, u_next_Z = branch_points("u", generation + 1, side_sign)
            mapped_R, mapped_Z = forward_map(u_R, u_Z)
            np.testing.assert_allclose(u_next_R, mapped_R, rtol=1.0e-12, atol=1.0e-14)
            np.testing.assert_allclose(u_next_Z, mapped_Z, rtol=1.0e-12, atol=1.0e-14)

            s_R, s_Z = branch_points("s", generation, side_sign)
            s_next_R, s_next_Z = branch_points("s", generation + 1, side_sign)
            mapped_R, mapped_Z = inverse_map(s_R, s_Z)
            np.testing.assert_allclose(s_next_R, mapped_R, rtol=1.0e-12, atol=1.0e-14)
            np.testing.assert_allclose(s_next_Z, mapped_Z, rtol=1.0e-12, atol=1.0e-14)

    for prefix in ("u", "s"):
        side = np.asarray(man[f"{prefix}_point_side"])
        R = np.asarray(man[f"{prefix}_R"])
        Z = np.asarray(man[f"{prefix}_Z"])
        lpol = np.asarray(man[f"{prefix}_lpol"])
        for side_sign in (-1.0, 1.0):
            branch = np.column_stack([R[side == side_sign], Z[side == side_sign]])
            centered = branch - np.mean(branch, axis=0)
            _u, singular_values, _vh = np.linalg.svd(centered, full_matrices=False)
            assert singular_values[-1] / singular_values[0] > 1.0e-2
            assert np.all(np.diff(lpol[side == side_sign]) > 0.0)


def test_trace_fixed_point_manifolds_defaults_to_monodromy_return_map():
    _skip_without_cyna_field_handle()
    lambda_step = 4.0
    field = _hyperbolic_field(rate=np.log(lambda_step) / (2.0 * np.pi))
    fp = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([lambda_step**2, lambda_step**-2]),
    )
    fp.map_power = 2
    fp.metadata.update({"map_power": 2, "field_period": 2.0 * np.pi})

    manifolds = trace_fixed_point_manifolds_field(
        field,
        [fp],
        phi_section=0.0,
        N_turns=1,
        DPhi=0.02,
        eps_min=1.0e-6,
        eps_max=1.0e-4,
        n_eps=4,
        include_arclength=True,
    )

    man = manifolds[0]
    assert man["manifold_field_period"] == pytest.approx(4.0 * np.pi)
    assert man["manifold_field_period_source"] == "map_power_times_field_period"

    def offsets(prefix: str, coord_key: str, generation: int) -> np.ndarray:
        gen = np.asarray(man[f"{prefix}_generation"])
        side = np.asarray(man[f"{prefix}_point_side"])
        order = np.asarray(man[f"{prefix}_point_seed_order"])
        coord = np.asarray(man[coord_key])
        mask = (gen == generation) & (side > 0.0)
        idx = np.argsort(order[mask])
        base = man["origin_R"] if coord_key.endswith("_R") else man["origin_Z"]
        return coord[mask][idx] - float(base)

    u0 = offsets("u", "u_R", 0)
    u1 = offsets("u", "u_R", 1)
    s0 = offsets("s", "s_Z", 0)
    s1 = offsets("s", "s_Z", 1)
    np.testing.assert_allclose(u1, (lambda_step**2) * u0, rtol=8.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(s1, (lambda_step**2) * s0, rtol=8.0e-3, atol=1.0e-8)


def test_trace_fixed_point_manifolds_accepts_explicit_total_map_span(monkeypatch):
    map_spans = []

    def fake_trace_map_points(
        field,
        seed_R,
        seed_Z,
        phi_section,
        *,
        N_turns,
        map_span,
        DPhi,
        wall_R=None,
        wall_Z=None,
        extend_phi=True,
        fd_eps=1.0e-4,
    ):
        map_spans.append(float(map_span))
        return [
            (np.asarray([float(r) + 0.01]), np.asarray([float(z)]))
            for r, z in zip(np.asarray(seed_R), np.asarray(seed_Z))
        ]

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain._trace_map_points_field",
        fake_trace_map_points,
    )
    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.diag([3.0, 1.0 / 3.0]))
    fp.map_power = 5
    fp.metadata.update({
        "map_power": 5,
        "monodromy_map_span": 2.0,
    })

    manifolds = trace_fixed_point_manifolds_field(
        object(),
        [fp],
        phi_section=0.0,
        N_turns=1,
        map_span=2.0,
        DPhi=0.1,
        eps_min=1.0e-5,
        eps_max=1.0e-4,
        n_eps=2,
    )

    assert len(manifolds) == 1
    assert manifolds[0]["manifold_field_period"] == pytest.approx(2.0)
    assert manifolds[0]["manifold_field_period_source"] == "explicit_map_span"
    assert map_spans == [pytest.approx(2.0), pytest.approx(-2.0)]


def test_trace_fixed_point_manifolds_rejects_monodromy_span_mismatch():
    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.diag([3.0, 1.0 / 3.0]))
    fp.map_power = 2
    fp.metadata.update({
        "map_power": 2,
        "monodromy_map_span": 4.0,
    })

    with pytest.raises(ValueError, match="DPm span metadata"):
        trace_fixed_point_manifolds_field(
            object(),
            [fp],
            phi_section=0.0,
            field_period=3.0,
            DPhi=0.1,
        )


def test_trace_fixed_point_manifolds_rejects_nonlocal_cloned_monodromy():
    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.diag([3.0, 1.0 / 3.0]))
    fp.metadata.update({"monodromy_local": False})

    with pytest.raises(ValueError, match="non-local"):
        trace_fixed_point_manifolds_field(
            object(),
            [fp],
            phi_section=0.0,
            map_span=1.0,
            DPhi=0.1,
        )


def test_trace_fixed_point_manifolds_multi_section_rejects_missing_section_key():
    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, kind="X", DPm=np.diag([3.0, 1.0 / 3.0]))

    with pytest.raises(KeyError, match="no section matching"):
        trace_fixed_point_manifolds_multi_section_field(
            object(),
            {0.0: {"xpts": [fp], "opts": []}},
            [0.5],
            map_span=1.0,
            DPhi=0.1,
        )


def test_refine_fixed_points_monodromy_span_field_refreshes_local_eigenvectors(monkeypatch):
    def fake_fixed_points(field, R, Z, phi_start, field_period, DPhi, **kwargs):
        assert field == "field"
        assert phi_start == pytest.approx(0.5)
        assert field_period == pytest.approx(6.0)
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        DPm = np.asarray([
            [[4.0, 0.0], [0.0, 0.25]],
            [[0.25, 0.0], [0.0, 4.0]],
        ])
        return (
            R + np.asarray([1.0e-4, -2.0e-4]),
            Z + np.asarray([2.0e-4, 1.0e-4]),
            np.asarray([2.0e-12, 3.0e-12]),
            np.asarray([True, True]),
            DPm.reshape(2, 4),
            np.asarray([[4.0, 0.25], [0.25, 4.0]]),
            np.zeros((2, 2)),
            np.asarray([1, 1]),
        )

    monkeypatch.setattr(
        "pyna.toroidal.flt.island_chain.find_fixed_points_batch_span_field",
        fake_fixed_points,
    )
    first = FixedPoint(phi=0.5, R=1.0, Z=0.0, DPm=np.diag([2.0, 0.5]), kind="X")
    second = FixedPoint(phi=0.5, R=1.1, Z=0.1, DPm=np.diag([2.0, 0.5]), kind="X")
    for idx, fp in enumerate((first, second)):
        fp.map_power = 3
        fp.metadata.update({
            "orbit_id": 7,
            "map_order_index": idx,
            "same_orbit_key": "chain=0:orbit=7:kind=X",
        })

    refined = refine_fixed_points_monodromy_span_field(
        "field",
        [first, second],
        field_period=2.0,
        DPhi=0.1,
        tol=1.0e-10,
    )

    assert len(refined) == 2
    np.testing.assert_allclose(refined[0].unstable_eigenvec, [1.0, 0.0])
    np.testing.assert_allclose(refined[1].unstable_eigenvec, [0.0, 1.0])
    assert refined[1].metadata["orbit_id"] == 7
    assert refined[1].metadata["map_order_index"] == 1
    assert refined[1].metadata["monodromy_local"] is True
    assert refined[1].metadata["monodromy_refined"] is True
    assert refined[1].metadata["monodromy_map_span"] == pytest.approx(6.0)
    assert refined[1].metadata["monodromy_refine_total_span"] == pytest.approx(6.0)


def test_boundary_island_shape_payload_uses_traced_curves():
    _skip_without_fixed_point_cyna()
    field = _rotation_field()
    opt = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="O",
        DPm=np.array([[0.0, -1.0], [1.0, 0.0]]),
    )

    shapes = trace_boundary_island_shapes_field(
        field,
        [opt],
        0.98,
        0.0,
        phi_section=0.0,
        shape_radius_fractions=(0.5,),
        n_shape_angles=4,
        N_turns=8,
        DPhi=0.02,
        min_points=4,
    )

    assert len(shapes) > 0
    assert all(len(R) >= 4 and len(Z) >= 4 for R, Z in shapes)
    edge_payload = boundary_island_edge_state_payload([shapes])
    assert edge_payload[0]["counts"]["boundary_island"] == len(shapes)
    assert edge_payload[0]["boundary_island"][0][0].ndim == 1
