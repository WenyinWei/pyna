import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import (
    boundary_island_edge_state_payload,
    boundary_recurrence_seed_candidates_field,
    boundary_seed_grid,
    vector_field_cylind_from_field,
    find_boundary_island_fixed_points_field,
    trace_map_batch_span_field,
    trace_boundary_island_shapes_field,
    trace_fixed_point_manifolds_field,
)


def _skip_without_fixed_point_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.find_fixed_points_batch is None:
        pytest.skip("cyna fixed-point search is unavailable")


def _skip_without_cyna_field_handle():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind handle is unavailable")


def _hyperbolic_field():
    axis_R = 1.0
    axis_Z = 0.0
    rate = 0.18
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
        BR=dR_dphi / RR,
        BZ=dZ_dphi / RR,
        BPhi=np.ones_like(RR),
    )


def _rotation_field():
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
        BR=dR_dphi / RR,
        BZ=dZ_dphi / RR,
        BPhi=np.ones_like(RR),
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
    field = _hyperbolic_field()

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        periods=(1,),
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
    np.testing.assert_allclose([fp.R, fp.Z], [1.0, 0.0], atol=1.0e-7)
    assert fp.residual < 1.0e-8
    payload_fp = result.fp_by_sec[0.0]["xpts"][0]
    assert payload_fp.kind == "X"
    assert payload_fp.stable_eigenvec is not None
    assert payload_fp.unstable_eigenvec is not None
    assert abs(payload_fp.stable_eigenvec[1]) > 0.99
    assert abs(payload_fp.unstable_eigenvec[0]) > 0.99
    np.testing.assert_allclose(payload_fp.DPm, fp.DPm)


def test_lower_period_filter_removes_axis_from_higher_period_search():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        periods=(2,),
        radii=(0.04, 0.08),
        n_theta=12,
        DPhi=0.02,
        max_iter=50,
        tol=1.0e-10,
        residual_tol=1.0e-8,
        dedup_tol=1.0e-4,
        lower_period_tol=1.0e-7,
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
        periods=(1,),
        seed_R=np.array([1.0, 1.0]),
        seed_Z=np.array([0.12, -0.12]),
        N_turns=10,
        DPhi=0.02,
        candidates_per_period=6,
        candidate_dedup_tol=1.0e-5,
    )

    seed_R, seed_Z = candidates.seeds_for_period(1)
    assert 0 < seed_R.size <= 6
    assert np.min(np.hypot(seed_R - 1.0, seed_Z)) < 1.0e-3

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        periods=(1,),
        radii=(0.12,),
        n_theta=4,
        candidate_strategy="recurrence",
        recurrence_turns=10,
        recurrence_candidates_per_period=6,
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


def test_field_period_map_fixed_point_search_uses_arbitrary_phi_span():
    _skip_without_fixed_point_cyna()
    field = _hyperbolic_field()

    result = find_boundary_island_fixed_points_field(
        field,
        1.0,
        0.0,
        phi_section=0.0,
        periods=(1,),
        radii=(0.12,),
        n_theta=4,
        candidate_strategy="recurrence",
        recurrence_turns=10,
        recurrence_candidates_per_period=6,
        map_period=np.pi,
        python_trust_radius=0.04,
        DPhi=0.02,
        max_iter=40,
        tol=1.0e-9,
        residual_tol=1.0e-7,
        dedup_tol=1.0e-4,
        n_threads=1,
    )

    assert result.diagnostics["map_period"] == pytest.approx(np.pi)
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
        periods=(1,),
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
    assert set(man) == {"u_R", "u_Z", "s_R", "s_Z"}
    assert man["u_R"].size > 0
    assert man["s_R"].size > 0
    assert np.all(np.isfinite(man["u_R"]))
    assert np.all(np.isfinite(man["s_Z"]))


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
