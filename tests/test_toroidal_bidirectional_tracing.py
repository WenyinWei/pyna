import json

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal import GlassWindow, TargetPlate, ToroidalComponentSurface, ToroidalWall
from pyna.toroidal.flt import (
    ToroidalWallTraceData,
    connection_length_views_from_wall_hits,
    normalize_connection_length_view,
    trace_toroidal_wall_data_field,
    trace_fieldline_trajectory,
    trace_fieldline_trajectory_bidirectional,
    trace_map_batch_span_field,
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
        nfp=int(nfp),
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
        nfp=int(nfp),
    )


def _field_period_strong_phi_field(nfp=2):
    R = np.linspace(0.45, 1.55, 65)
    Z = np.linspace(-0.55, 0.55, 65)
    period = 2.0 * np.pi / int(nfp)
    Phi = np.linspace(0.0, period, 97, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    x = RR - 1.0
    y = ZZ
    fR = (
        0.25 * np.sin(int(nfp) * PP + 0.2)
        + 0.35 * np.cos(2 * int(nfp) * PP - 0.1) * x
        + 0.28 * np.sin(3 * int(nfp) * PP + 0.4) * y
        + 0.22 * np.cos(int(nfp) * PP) * (x * x - 0.7 * y * y)
    )
    fZ = (
        -0.18 * np.cos(int(nfp) * PP - 0.3)
        + 0.31 * np.sin(2 * int(nfp) * PP + 0.5) * x
        - 0.24 * np.cos(3 * int(nfp) * PP - 0.2) * y
        + 0.18 * np.sin(int(nfp) * PP + 0.1) * x * y
    )
    return VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=fR / RR,
        BZ=fZ / RR,
        BPhi=np.ones_like(RR),
        nfp=int(nfp),
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


def test_toroidal_component_surface_validation_and_aliases():
    wall_phi, wall_R, wall_Z = _toroidal_wall()
    surface = ToroidalWall(wall_phi, wall_R, wall_Z, name="reference_wall")

    assert surface.kind == "wall"
    assert surface.nfp == 1
    assert surface.n_phi == wall_phi.size
    assert surface.n_poloidal == wall_R.shape[1]
    np.testing.assert_allclose(surface.wall_phi, wall_phi)
    np.testing.assert_allclose(surface.wall_R_all, wall_R)
    np.testing.assert_allclose(surface.section(phi=0.01)[0], wall_R[0])

    target = TargetPlate(wall_phi, wall_R, wall_Z, name="target")
    window = GlassWindow(wall_phi, wall_R, wall_Z, name="window")
    assert target.kind == "target_plate"
    assert window.kind == "glass_window"

    with pytest.raises(ValueError, match="R and Z surface arrays"):
        ToroidalComponentSurface(wall_phi, wall_R, wall_Z[:, :-1])

    fp_phi = np.linspace(0.0, np.pi, 4, endpoint=False)
    fp_R = wall_R[:4]
    fp_Z = wall_Z[:4]
    fp_wall = ToroidalWall(fp_phi, fp_R, fp_Z, nfp=2)
    assert fp_wall.field_period == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="stored toroidal domain"):
        ToroidalWall(wall_phi, wall_R, wall_Z, nfp=2)


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
    wall = ToroidalWall(wall_phi, wall_R, wall_Z)

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

    hits_obj = trace_wall_hits_twall_field(
        field,
        np.array([1.0]),
        np.array([0.0]),
        0.0,
        2,
        0.01,
        wall,
    )
    np.testing.assert_allclose(hits_obj["hit_plus"][0, 0], 1.2, atol=2.0e-3)

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

    strike_obj = trace_strike_line_twall_field(
        field,
        np.array([1.0]),
        np.array([0.0]),
        0.0,
        2,
        0.01,
        wall,
        direction="+",
    )
    np.testing.assert_allclose(strike_obj["R"], np.array([1.2]), atol=2.0e-3)


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
    with np.load(cache, allow_pickle=False) as raw:
        assert str(np.asarray(raw["schema_name"]).item()) == "pyna.toroidal.flt.wall_trace_data"
        assert int(np.asarray(raw["schema_version"]).item()) == 1
        assert str(np.asarray(raw["compatibility"]).item()) == "v1 append-only; readers ignore unknown fields"
        assert bool(np.asarray(raw["complete"]).item()) is True
        assert int(np.asarray(raw["n_seed"]).item()) == 2
        wall_hit_keys = json.loads(str(np.asarray(raw["wall_hit_keys_json"]).item()))
        assert wall_hit_keys == sorted(wall_hits)
        wall_hit_specs = json.loads(str(np.asarray(raw["wall_hit_specs_json"]).item()))
        assert wall_hit_specs["Lc_plus"]["shape"] == [2]
        assert wall_hit_specs["hit_plus"]["shape"] == [2, 3]
        assert wall_hit_specs["term_plus"]["dtype"] == str(wall_hits["term_plus"].dtype)
    loaded = ToroidalWallTraceData.load_npz(cache)
    np.testing.assert_allclose(loaded.view("max"), [3.0, 2.0])
    incomplete_cache = tmp_path / "wall_trace_incomplete.npz"
    with np.load(cache, allow_pickle=False) as raw:
        arrays = {key: np.asarray(raw[key]) for key in raw.files}
    arrays["complete"] = np.asarray(False)
    np.savez(incomplete_cache, **arrays)
    with pytest.raises(ValueError, match="not marked complete"):
        ToroidalWallTraceData.load_npz(incomplete_cache)

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
    fake_wall = ToroidalWall([0.0], [[1.2, 1.2]], [[0.0, 0.1]])
    with pytest.raises(ValueError, match="lacks cache signature"):
        trace_toroidal_wall_data_field(
            object(),
            [1.0, 1.1],
            [0.0, 0.0],
            0.0,
            2,
            0.01,
            fake_wall,
            cache_path=cache,
        )
    legacy = trace_toroidal_wall_data_field(
        object(),
        [1.0, 1.1],
        [0.0, 0.0],
        0.0,
        2,
        0.01,
        fake_wall,
        cache_path=cache,
        allow_legacy_unsigned_cache=True,
    )
    np.testing.assert_allclose(legacy.view("total"), [4.0, 2.5])

    traced = trace_toroidal_wall_data_field(
        object(),
        [1.0, 1.1],
        [0.0, 0.0],
        0.0,
        2,
        0.01,
        fake_wall,
        cache_path=cache2,
    )
    cached = trace_toroidal_wall_data_field(
        object(),
        [1.0, 1.1],
        [0.0, 0.0],
        0.0,
        2,
        0.01,
        fake_wall,
        cache_path=cache2,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(traced.view("total"), cached.view("total"))
    assert "field_signature" in traced.metadata
    assert traced.metadata["cache_signature_inputs"]["extend_phi"] is True
    assert traced.metadata["cache_signature_inputs"]["seed_R"]
    assert traced.metadata["cache_signature_inputs"]["wall_phi"]

    with pytest.raises(ValueError, match="does not match requested inputs"):
        trace_toroidal_wall_data_field(
            object(),
            [1.0, 1.2],
            [0.0, 0.0],
            0.0,
            2,
            0.01,
            fake_wall,
            cache_path=cache2,
        )
    with pytest.raises(ValueError, match="does not match requested inputs"):
        trace_toroidal_wall_data_field(
            object(),
            [1.0, 1.1],
            [0.0, 0.0],
            0.0,
            2,
            0.01,
            fake_wall,
            cache_path=cache2,
            field_signature={"case": "different_field"},
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


def test_forward_backward_roundtrip_keeps_rk4_order_across_field_period_seam():
    _skip_without_cyna()
    field = _field_period_strong_phi_field(nfp=2)
    period = field.field_period
    phi_crossing = period - 0.137
    phi_no_seam = 0.645
    span = 0.384
    steps = span / np.asarray([4, 8, 16, 32], dtype=float)
    R0 = np.asarray([0.98, 1.05, 1.12], dtype=float)
    Z0 = np.asarray([0.03, -0.025, 0.055], dtype=float)

    def roundtrip_error(phi0, dphi):
        _counts, Rf, Zf = trace_map_batch_span_field(
            field, R0, Z0, phi0, span, 1, dphi, n_threads=1
        )
        Rf = np.asarray(Rf[: R0.size], dtype=float)
        Zf = np.asarray(Zf[: Z0.size], dtype=float)
        _counts, Rb, Zb = trace_map_batch_span_field(
            field, Rf, Zf, phi0 + span, -span, 1, dphi, n_threads=1
        )
        Rb = np.asarray(Rb[: R0.size], dtype=float)
        Zb = np.asarray(Zb[: Z0.size], dtype=float)
        return float(np.max(np.hypot(Rb - R0, Zb - Z0)))

    seam_error = np.asarray([roundtrip_error(phi_crossing, h) for h in steps])
    no_seam_error = np.asarray([roundtrip_error(phi_no_seam, h) for h in steps])
    seam_slope = np.polyfit(np.log(steps), np.log(seam_error), 1)[0]

    assert phi_crossing < period < phi_crossing + span
    assert seam_slope > 3.0
    assert seam_error[-1] < 1.0e-3 * seam_error[0]
    assert np.max(seam_error / no_seam_error) < 10.0


def test_cyna_interpolation_wraps_at_field_period_seam():
    _skip_without_cyna()
    import pyna._cyna as cyna

    ext = getattr(cyna, "_cyna_ext", None)
    interp = getattr(ext, "interp3d_test", None)
    if interp is None:
        pytest.skip("cyna interpolation test hook is unavailable")

    field = _field_period_strong_phi_field(nfp=2)
    arrays = field.cyna_arrays(extend_phi=True)
    period = field.field_period
    sample_rz = [(0.97, 0.04), (1.13, -0.07), (1.24, 0.09)]
    eps_values = (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-8)

    for values in (arrays.BR, arrays.BZ, arrays.BPhi):
        for R0, Z0 in sample_rz:
            args = (values, arrays.R_grid, arrays.Z_grid, arrays.Phi_grid)
            for eps in eps_values:
                np.testing.assert_allclose(
                    interp(R0, Z0, period + eps, *args, field.nfp),
                    interp(R0, Z0, eps, *args, field.nfp),
                    rtol=0.0,
                    atol=5.0e-14,
                )
                np.testing.assert_allclose(
                    interp(R0, Z0, -eps, *args, field.nfp),
                    interp(R0, Z0, period - eps, *args, field.nfp),
                    rtol=0.0,
                    atol=5.0e-14,
                )


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
    checkpoint = json.loads((tmp_path / "fieldline_trajectory_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["schema_name"] == "pyna.toroidal.flt.fieldline_trajectory_checkpoint"
    assert checkpoint["compatibility"] == "v1 append-only; readers ignore unknown fields"
    assert checkpoint["options"]["extend_phi"] is True
    assert checkpoint["field_signature"]["type"].endswith("VectorFieldCylind")
    assert checkpoint["array_specs"]["R"]["shape"] == [partial.metadata["n_total"]]
    assert checkpoint["array_specs"]["alive"]["dtype"] == "int8"

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


def test_restartable_dense_trajectory_rejects_memory_storage_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_dir"):
        trace_fieldline_trajectory(
            object(),
            1.08,
            0.02,
            0.0,
            1.0,
            0.01,
            dphi_out=0.05,
            checkpoint_dir=tmp_path,
            storage="memory",
        )


def test_restartable_dense_trajectory_rejects_mismatched_resume_inputs(tmp_path):
    _skip_without_cyna()
    field = _rotation_field(omega=0.25)

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
        stop_after_chunks=1,
    )
    assert partial.status == "incomplete"

    with pytest.raises(ValueError, match="field signature"):
        trace_fieldline_trajectory(
            _rotation_field(omega=0.30),
            1.08,
            0.02,
            0.0,
            1.0,
            0.01,
            dphi_out=0.05,
            chunk_phi_span=0.2,
            checkpoint_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="extend_phi"):
        trace_fieldline_trajectory(
            field,
            1.08,
            0.02,
            0.0,
            1.0,
            0.01,
            dphi_out=0.05,
            chunk_phi_span=0.2,
            checkpoint_dir=tmp_path,
            extend_phi=False,
        )

    checkpoint_path = tmp_path / "fieldline_trajectory_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["last_R"] = float(checkpoint["last_R"]) + 0.1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="last_R"):
        trace_fieldline_trajectory(
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
