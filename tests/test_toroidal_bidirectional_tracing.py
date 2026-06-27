import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.flt import (
    trace_fieldline_trajectory,
    trace_orbit_along_phi_field,
    trace_poincare_bidirectional_batch_field,
    trace_strike_line_twall_field,
    trace_wall_hits_twall_field,
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
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
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
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    fR = np.full_like(RR, 0.05)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=fR / RR,
        BZ=np.zeros_like(RR),
        BPhi=np.ones_like(RR),
    )


def _wall_2d():
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    return 1.0 + 0.25 * np.cos(theta), 0.25 * np.sin(theta)


def _toroidal_wall():
    wall_phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    R_wall = np.tile(1.0 + 0.2 * np.cos(theta), (wall_phi.size, 1))
    Z_wall = np.tile(0.2 * np.sin(theta), (wall_phi.size, 1))
    return wall_phi, R_wall, Z_wall


def test_trace_orbit_along_phi_field_accepts_backward_output():
    _skip_without_cyna()
    field = _rotation_field()

    R_t, Z_t, phi_t, _DP_t, alive = trace_orbit_along_phi_field(
        field,
        1.08,
        0.02,
        0.0,
        -0.5,
        0.01,
        dphi_out=0.1,
    )

    assert len(phi_t) == 6
    assert bool(alive[-1])
    np.testing.assert_allclose(phi_t, np.linspace(0.0, -0.5, 6), atol=1.0e-14)
    assert np.isfinite(R_t[-1])
    assert np.isfinite(Z_t[-1])


def test_poincare_bidirectional_counts_match_forward_backward():
    _skip_without_cyna()
    field = _rotation_field()
    wall_R, wall_Z = _wall_2d()

    result = trace_poincare_bidirectional_batch_field(
        field,
        np.array([1.08]),
        np.array([0.02]),
        0.0,
        3,
        0.02,
        wall_R,
        wall_Z,
    )

    f_counts, f_R, f_Z = result["forward"]
    b_counts, b_R, b_Z = result["backward"]
    assert int(f_counts[0]) == 3
    assert int(b_counts[0]) == 3
    assert np.all(np.isfinite(f_R[:3]))
    assert np.all(np.isfinite(f_Z[:3]))
    assert np.all(np.isfinite(b_R[:3]))
    assert np.all(np.isfinite(b_Z[:3]))


def test_toroidal_wall_hits_and_strike_line_object_api():
    _skip_without_cyna()
    field = _outward_field()
    wall_phi, wall_R, wall_Z = _toroidal_wall()

    hits = trace_wall_hits_twall_field(
        field,
        np.array([1.0]),
        np.array([0.0]),
        0.0,
        2,
        0.01,
        wall_phi,
        wall_R,
        wall_Z,
    )
    assert int(hits["term_plus"][0]) == 1
    assert int(hits["term_minus"][0]) == 1
    np.testing.assert_allclose(hits["hit_plus"][0, 0], 1.2, atol=2.0e-3)
    np.testing.assert_allclose(hits["hit_minus"][0, 0], 0.8, atol=2.0e-3)

    strike = trace_strike_line_twall_field(
        field,
        np.array([1.0]),
        np.array([0.0]),
        0.0,
        2,
        0.01,
        wall_phi,
        wall_R,
        wall_Z,
        direction="+",
    )
    np.testing.assert_array_equal(strike["seed_index"], np.array([0]))
    np.testing.assert_allclose(strike["R"], np.array([1.2]), atol=2.0e-3)


def test_toroidal_wall_period_wraps_for_field_period_wall():
    _skip_without_cyna()
    field = _outward_field()
    wall_phi = np.linspace(0.0, np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    minor_radius = np.array([0.10, 0.14, 0.18, 0.22])
    wall_R = 1.0 + minor_radius[:, None] * np.cos(theta)[None, :]
    wall_Z = minor_radius[:, None] * np.sin(theta)[None, :]

    hits = trace_wall_hits_twall_field(
        field,
        np.array([1.095]),
        np.array([0.0]),
        np.pi + 0.01,
        1,
        0.002,
        wall_phi,
        wall_R,
        wall_Z,
    )

    assert int(hits["term_plus"][0]) == 1
    np.testing.assert_allclose(hits["hit_plus"][0, 0], 1.10, atol=2.0e-3)


def test_restartable_dense_trajectory_matches_unchunked(tmp_path):
    _skip_without_cyna()
    field = _rotation_field()

    full = trace_fieldline_trajectory(
        field,
        1.08,
        0.02,
        0.0,
        1.0,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=10.0,
        storage="memory",
    )
    partial = trace_fieldline_trajectory(
        field,
        1.08,
        0.02,
        0.0,
        1.0,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.2,
        checkpoint_dir=tmp_path,
        stop_after_chunks=2,
    )
    assert partial.status == "incomplete"

    resumed = trace_fieldline_trajectory(
        field,
        1.08,
        0.02,
        0.0,
        1.0,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.2,
        checkpoint_dir=tmp_path,
    )
    assert resumed.status == "complete"
    np.testing.assert_allclose(resumed.phi, full.phi, atol=1.0e-14)
    np.testing.assert_allclose(resumed.R, full.R, atol=1.0e-14)
    np.testing.assert_allclose(resumed.Z, full.Z, atol=1.0e-14)

    values = resumed.sol([0.25, 0.75])
    assert values.shape == (2, 2)
    assert np.all(np.isfinite(values))
