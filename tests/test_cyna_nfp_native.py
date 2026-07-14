import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.flt import (
    evolve_delta_X_cycle_along_cycle_field,
    find_fixed_points_batch_span_field,
    progress_DX_pol_along_orbit_field,
    progress_delta_X_along_orbit_field,
    trace_map_batch_span,
    trace_map_batch_span_field,
    trace_orbit_along_phi_field,
    trace_poincare_batch_field,
    vector_field_cylind_from_field,
)


NFP = 5
NPHI_NATIVE = 64


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available():
        pytest.skip("cyna extension is unavailable")
    return cyna


def _native_and_full_fields(*, perturbation=False):
    period = 2.0 * np.pi / NFP
    R = np.linspace(0.82, 1.18, 21)
    Z = np.linspace(-0.18, 0.18, 21)
    Phi = np.linspace(0.0, period, NPHI_NATIVE, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    x = RR - 1.0
    y = ZZ

    if perturbation:
        BR = (0.003 * np.cos(NFP * PP + 0.2) + 0.002 * y) / RR
        BZ = (-0.0025 * np.sin(2 * NFP * PP - 0.1) + 0.0015 * x) / RR
        BPhi = 0.001 * np.cos(NFP * PP - 0.3) * np.ones_like(RR)
    else:
        fR = (0.025 + 0.012 * np.cos(NFP * PP)) * x + 0.040 * y
        fZ = -0.050 * x + (0.018 * np.sin(NFP * PP + 0.3)) * y
        BR = fR / RR
        BZ = fZ / RR
        BPhi = 1.0 + 0.010 * x + 0.015 * np.cos(NFP * PP - 0.2)

    native = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=BR,
        BZ=BZ,
        BPhi=BPhi,
        nfp=NFP,
    )
    Phi_full = np.linspace(0.0, 2.0 * np.pi, NFP * NPHI_NATIVE, endpoint=False)
    full = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi_full,
        BR=np.tile(BR, (1, 1, NFP)),
        BZ=np.tile(BZ, (1, 1, NFP)),
        BPhi=np.tile(BPhi, (1, 1, NFP)),
        nfp=1,
    )
    return native, full


def _assert_float_tuples_close(native, full, *, atol=3.0e-12):
    assert len(native) == len(full)
    for lhs, rhs in zip(native, full):
        lhs = np.asarray(lhs)
        rhs = np.asarray(rhs)
        if lhs.dtype.kind in "biu" or rhs.dtype.kind in "biu":
            np.testing.assert_array_equal(lhs, rhs)
        else:
            np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=atol, equal_nan=True)


def test_nfp5_native_storage_closes_64_planes_to_65_only():
    cyna = _require_cyna()
    native, full = _native_and_full_fields()

    native_arrays = native.cyna_arrays(extend_phi=True)
    full_arrays = full.cyna_arrays(extend_phi=True)
    handle = vector_field_cylind_from_field(native)

    assert native_arrays.nfp == NFP
    assert native_arrays.Phi_grid.size == NPHI_NATIVE + 1
    assert full_arrays.Phi_grid.size == NFP * NPHI_NATIVE + 1
    assert handle.nfp == NFP
    assert handle.nPhi == NPHI_NATIVE + 1

    with pytest.raises(ValueError, match="explicit period"):
        trace_map_batch_span(
            np.array([1.02]),
            np.array([0.01]),
            0.0,
            native.field_period,
            1,
            0.02,
            native_arrays.R_grid,
            native_arrays.Z_grid,
            native_arrays.Phi_grid,
            native_arrays.BR_flat,
            native_arrays.BZ_flat,
            native_arrays.BPhi_flat,
            np.empty(0),
            np.empty(0),
        )

    with pytest.raises(RuntimeError, match="2\\*pi/nfp|field period"):
        cyna.VectorFieldCylind(
            native_arrays.R_grid,
            native_arrays.Z_grid,
            native_arrays.Phi_grid,
            native_arrays.BR_flat,
            native_arrays.BZ_flat,
            native_arrays.BPhi_flat,
            nfp=1,
        )


def test_nfp5_native_map_poincare_and_orbit_match_full_torus_reference():
    _require_cyna()
    native, full = _native_and_full_fields()
    R0 = np.array([0.96, 1.03, 1.08])
    Z0 = np.array([0.025, -0.020, 0.035])
    period = native.field_period

    native_map = trace_map_batch_span_field(
        native, R0, Z0, 0.17, period, 7, 0.015, n_threads=1
    )
    full_map = trace_map_batch_span_field(
        full, R0, Z0, 0.17, period, 7, 0.015, n_threads=1
    )
    _assert_float_tuples_close(native_map, full_map)

    empty = np.empty(0)
    native_poi = trace_poincare_batch_field(
        native, R0, Z0, 0.0, 2, 0.015, empty, empty
    )
    full_poi = trace_poincare_batch_field(
        full, R0, Z0, 0.0, 2, 0.015, empty, empty
    )
    _assert_float_tuples_close(native_poi, full_poi)
    np.testing.assert_array_equal(native_poi[0], np.full(R0.size, 2))

    native_orbit = trace_orbit_along_phi_field(
        native, 1.06, -0.025, 0.11, 2.0 * np.pi + 0.37, 0.015, dphi_out=0.07
    )
    full_orbit = trace_orbit_along_phi_field(
        full, 1.06, -0.025, 0.11, 2.0 * np.pi + 0.37, 0.015, dphi_out=0.07
    )
    _assert_float_tuples_close(native_orbit, full_orbit)


def test_nfp5_native_fixed_point_and_variational_responses_match_full_torus():
    cyna = _require_cyna()
    native, full = _native_and_full_fields()
    delta_native, delta_full = _native_and_full_fields(perturbation=True)
    period = native.field_period

    native_fp = find_fixed_points_batch_span_field(
        native,
        np.array([1.015, 0.985]),
        np.array([0.012, -0.010]),
        0.13,
        period,
        0.012,
        fd_eps=2.0e-5,
        max_iter=30,
        tol=1.0e-11,
        n_threads=1,
    )
    full_fp = find_fixed_points_batch_span_field(
        full,
        np.array([1.015, 0.985]),
        np.array([0.012, -0.010]),
        0.13,
        period,
        0.012,
        fd_eps=2.0e-5,
        max_iter=30,
        tol=1.0e-11,
        n_threads=1,
    )
    _assert_float_tuples_close(native_fp, full_fp, atol=2.0e-10)
    np.testing.assert_array_equal(native_fp[3], np.ones(2, dtype=int))
    np.testing.assert_allclose(native_fp[0], 1.0, atol=2.0e-9)
    np.testing.assert_allclose(native_fp[1], 0.0, atol=2.0e-9)

    orbit = trace_orbit_along_phi_field(
        native, 1.055, -0.020, 0.09, 3.4 * period, 0.012, dphi_out=0.04
    )
    R_t, Z_t, phi_t = orbit[:3]
    native_DX = progress_DX_pol_along_orbit_field(native, R_t, Z_t, phi_t, max_step=0.012)
    full_DX = progress_DX_pol_along_orbit_field(full, R_t, Z_t, phi_t, max_step=0.012)
    np.testing.assert_allclose(native_DX, full_DX, rtol=0.0, atol=3.0e-12)

    dX0 = np.array([0.001, -0.002])
    native_dX = progress_delta_X_along_orbit_field(
        native, delta_native, R_t, Z_t, phi_t, dX0, max_step=0.012
    )
    full_dX = progress_delta_X_along_orbit_field(
        full, delta_full, R_t, Z_t, phi_t, dX0, max_step=0.012
    )
    np.testing.assert_allclose(native_dX, full_dX, rtol=0.0, atol=3.0e-12)

    native_cycle = evolve_delta_X_cycle_along_cycle_field(
        native, delta_native, R_t, Z_t, phi_t, dX0, max_step=0.012
    )
    full_cycle = evolve_delta_X_cycle_along_cycle_field(
        full, delta_full, R_t, Z_t, phi_t, dX0, max_step=0.012
    )
    np.testing.assert_allclose(native_cycle, full_cycle, rtol=0.0, atol=3.0e-12)

    native_arrays = native.cyna_arrays(extend_phi=True)
    full_arrays = full.cyna_arrays(extend_phi=True)
    DPm0 = np.array([[1.1, 0.2], [-0.1, 0.9]])
    native_DPm = cyna.evolve_DPm_along_cycle(
        R_t,
        Z_t,
        phi_t,
        DPm0,
        native_arrays.BR_flat,
        native_arrays.BZ_flat,
        native_arrays.BPhi_flat,
        native_arrays.R_grid,
        native_arrays.Z_grid,
        native_arrays.Phi_grid,
        NFP,
    )
    full_DPm = cyna.evolve_DPm_along_cycle(
        R_t,
        Z_t,
        phi_t,
        DPm0,
        full_arrays.BR_flat,
        full_arrays.BZ_flat,
        full_arrays.BPhi_flat,
        full_arrays.R_grid,
        full_arrays.Z_grid,
        full_arrays.Phi_grid,
        1,
    )
    np.testing.assert_allclose(native_DPm, full_DPm, rtol=0.0, atol=3.0e-12)


def test_nfp5_native_beta_sweep_matches_full_torus_reference():
    cyna = _require_cyna()
    native, full = _native_and_full_fields()
    native_arrays = native.cyna_arrays(extend_phi=True)
    full_arrays = full.cyna_arrays(extend_phi=True)
    R0 = np.array([0.97, 1.04])
    Z0 = np.array([0.02, -0.03])
    sections = np.array([0.0])
    empty = np.empty(0)

    def run(arrays, nfp):
        return cyna.trace_poincare_beta_sweep(
            R0,
            Z0,
            sections,
            1,
            0.015,
            arrays.BR_flat,
            arrays.BZ_flat,
            arrays.BPhi_flat,
            arrays.R_grid,
            arrays.Z_grid,
            arrays.Phi_grid,
            empty,
            empty,
            0.012,
            1.0,
            0.0,
            0.3,
            2.0,
            1.0,
            1,
            nfp,
        )

    native_result = run(native_arrays, NFP)
    full_result = run(full_arrays, 1)
    _assert_float_tuples_close(native_result, full_result, atol=5.0e-12)


def test_nfp5_native_fast_surface_metrics_match_full_torus_reference():
    cyna = _require_cyna()
    native, full = _native_and_full_fields()
    native_arrays = native.cyna_arrays(extend_phi=True)
    full_arrays = full.cyna_arrays(extend_phi=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_phi = np.array([0.0])
    wall_R = (1.0 + 0.17 * np.cos(theta))[None, :]
    wall_Z = (0.17 * np.sin(theta))[None, :]
    R0 = np.array([1.03, 1.06])
    Z0 = np.array([0.0, 0.0])

    def run(arrays, nfp):
        return cyna.trace_surface_metrics_batch_twall(
            R0,
            Z0,
            1.0,
            0.0,
            0.0,
            1,
            0.015,
            arrays.BR_flat,
            arrays.BZ_flat,
            arrays.BPhi_flat,
            arrays.R_grid,
            arrays.Z_grid,
            arrays.Phi_grid,
            wall_phi,
            wall_R,
            wall_Z,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1,
            nfp,
        )

    native_result = run(native_arrays, NFP)
    full_result = run(full_arrays, 1)
    _assert_float_tuples_close(native_result, full_result, atol=5.0e-8)
