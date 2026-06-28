import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.flt import (
    ToroidalWallTraceData,
    connection_length_views_from_wall_hits,
    normalize_connection_length_view,
    trace_toroidal_wall_data_field,
    trace_fieldline_trajectory,
    trace_fieldline_trajectory_bidirectional,
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


def _outward_field(nfp=1):
    R = np.linspace(0.7, 1.3, 33)
    Z = np.linspace(-0.3, 0.3, 17)
    Phi = np.linspace(0.0, 2.0 * np.pi / int(nfp), 8, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    fR = np.full_like(RR, 0.05)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=fR / RR,
        BZ=np.zeros_like(RR),
        BPhi=np.ones_like(RR),
        field_periods=int(nfp),
    )


def _field_period_modulated_radial_field(nfp=2):
    R = np.linspace(0.82, 1.18, 33)
    Z = np.linspace(-0.18, 0.18, 17)
    period = 2.0 * np.pi / int(nfp)
    Phi = np.linspace(0.0, period, 32, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    dR_dphi = 0.012 * np.cos(int(nfp) * PP)
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=dR_dphi / RR,
        BZ=np.zeros_like(RR),
        BPhi=np.ones_like(RR),
        field_periods=int(nfp),
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


def test_wall_trace_post_compute_views_and_cache(tmp_path, monkeypatch):
    wall_hits = {
        "Lc_plus": np.asarray([1.0, 2.0]),
        "Lc_minus": np.asarray([3.0, 0.5]),
        "hit_plus": np.asarray([[1.2, 0.0, 0.1], [1.1, 0.0, 0.2]]),
        "hit_minus": np.asarray([[0.8, 0.0, -0.1], [0.9, 0.0, -0.2]]),
        "term_plus": np.asarray([1, 2]),
        "term_minus": np.asarray([1, 1]),
    }

    views = connection_length_views_from_wall_hits(wall_hits, ("forward", "backward", "total", "max"))
    np.testing.assert_allclose(views["Lc_plus"], [1.0, 2.0])
    np.testing.assert_allclose(views["Lc_minus"], [3.0, 0.5])
    np.testing.assert_allclose(views["Lc_sum"], [4.0, 2.5])
    np.testing.assert_allclose(views["Lc_max"], [3.0, 2.0])
    assert normalize_connection_length_view("Lc+") == "Lc_plus"

    data = ToroidalWallTraceData(
        seed_R=np.asarray([1.0, 1.1]),
        seed_Z=np.asarray([0.0, 0.0]),
        phi_start=0.0,
        max_turns=2,
        DPhi=0.01,
        wall_hits=wall_hits,
        metadata={"case": "synthetic"},
    )
    cache = tmp_path / "wall_trace.npz"
    data.save_npz(cache)
    loaded = ToroidalWallTraceData.load_npz(cache)
    np.testing.assert_allclose(loaded.view("max"), [3.0, 2.0])
    views_cache = tmp_path / "wall_trace_views.npz"
    saved_views = loaded.save_views_npz(views_cache, ("forward", "total"))
    assert views_cache.exists()
    np.testing.assert_allclose(saved_views["Lc_sum"], [4.0, 2.5])
    strike = loaded.strike(direction="-")
    np.testing.assert_allclose(strike["R"], [0.8, 0.9])
    assert loaded.metadata["case"] == "synthetic"

    calls = []

    def fake_trace(*args, **kwargs):
        calls.append((args, kwargs))
        return wall_hits

    monkeypatch.setattr("pyna.toroidal.flt.postcompute.trace_wall_hits_twall_field", fake_trace)
    cache2 = tmp_path / "wall_trace_from_field.npz"
    traced = trace_toroidal_wall_data_field(
        object(),
        [1.0, 1.1],
        [0.0, 0.0],
        0.0,
        2,
        0.01,
        [0.0],
        [[1.2]],
        [[0.0]],
        cache_path=cache2,
    )
    cached = trace_toroidal_wall_data_field(
        object(),
        [1.0, 1.1],
        [0.0, 0.0],
        0.0,
        2,
        0.01,
        [0.0],
        [[1.2]],
        [[0.0]],
        cache_path=cache2,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(traced.view("total"), cached.view("total"))

    with pytest.raises(ValueError, match="does not match requested inputs"):
        trace_toroidal_wall_data_field(
            object(),
            [1.0, 1.2],
            [0.0, 0.0],
            0.0,
            2,
            0.01,
            [0.0],
            [[1.2]],
            [[0.0]],
            cache_path=cache2,
        )


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


def test_backward_trace_crosses_field_period_phi_seam():
    _skip_without_cyna()
    field = _field_period_modulated_radial_field(nfp=2)
    phi0 = 0.85 * np.pi

    R_t, Z_t, phi_t, _DP_t, alive = trace_orbit_along_phi_field(
        field,
        1.0,
        0.0,
        phi0,
        phi0 - 2.0 * np.pi,
        0.01,
        dphi_out=0.05,
    )

    assert bool(alive[-1])
    assert np.all(np.diff(phi_t) <= 1.0e-13)
    np.testing.assert_allclose(R_t[-1], 1.0, atol=5.0e-4)
    np.testing.assert_allclose(Z_t[-1], 0.0, atol=1.0e-12)


def test_field_period_wall_wraps_for_backward_wall_hit():
    _skip_without_cyna()
    field = _outward_field(nfp=2)
    wall_phi = np.linspace(0.0, np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_R = 1.0 + 0.10 * np.cos(theta)[None, :] * np.ones((wall_phi.size, 1))
    wall_Z = 0.10 * np.sin(theta)[None, :] * np.ones((wall_phi.size, 1))

    hits = trace_wall_hits_twall_field(
        field,
        np.array([0.905]),
        np.array([0.0]),
        np.pi + 0.01,
        1,
        0.002,
        wall_phi,
        wall_R,
        wall_Z,
    )

    assert int(hits["term_minus"][0]) == 1
    np.testing.assert_allclose(hits["hit_minus"][0, 0], 0.90, atol=2.0e-3)


def test_bidirectional_dense_trajectory_crosses_field_period_seam():
    _skip_without_cyna()
    field = _field_period_modulated_radial_field(nfp=2)
    phi0 = 0.85 * np.pi

    out = trace_fieldline_trajectory_bidirectional(
        field,
        1.0,
        0.0,
        phi0,
        2.0 * np.pi,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.4,
        storage="memory",
    )

    assert out["forward"].status == "complete"
    assert out["backward"].status == "complete"
    assert out["forward"].phi[0] < out["forward"].phi[-1]
    assert out["backward"].phi[0] > out["backward"].phi[-1]
    assert np.all(np.isfinite(out["forward"].sol([phi0 + 0.2, phi0 + 1.1 * np.pi])))
    assert np.all(np.isfinite(out["backward"].sol([phi0 - 0.2, phi0 - 1.1 * np.pi])))


def test_restart_resume_crosses_field_period_phi_seam(tmp_path):
    _skip_without_cyna()
    field = _field_period_modulated_radial_field(nfp=2)
    phi0 = 0.85 * np.pi

    full = trace_fieldline_trajectory(
        field,
        1.0,
        0.0,
        phi0,
        phi0 - 2.0 * np.pi,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.35,
        storage="memory",
    )
    partial = trace_fieldline_trajectory(
        field,
        1.0,
        0.0,
        phi0,
        phi0 - 2.0 * np.pi,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.35,
        checkpoint_dir=tmp_path,
        stop_after_chunks=3,
    )
    assert partial.status == "incomplete"

    resumed = trace_fieldline_trajectory(
        field,
        1.0,
        0.0,
        phi0,
        phi0 - 2.0 * np.pi,
        0.01,
        dphi_out=0.05,
        chunk_phi_span=0.35,
        checkpoint_dir=tmp_path,
    )

    assert resumed.status == "complete"
    np.testing.assert_allclose(resumed.phi, full.phi, atol=1.0e-14)
    np.testing.assert_allclose(resumed.R, full.R, atol=1.0e-12)
    np.testing.assert_allclose(resumed.Z, full.Z, atol=1.0e-12)


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
