import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.flt import (
    axis_cycle_shift_from_fields,
    cycle_points_shift_from_fields,
    field_period_cache_from_components,
    find_fixed_points_batch_field,
    trace_orbit_along_phi_field,
)


def _constant_fields(nfp=5):
    R = np.linspace(0.8, 1.2, 5)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, 2.0 * np.pi / nfp, 8, endpoint=False)
    shape = (R.size, Z.size, Phi.size)
    base = VectorFieldCylind(
        R,
        Z,
        Phi,
        BR=np.zeros(shape),
        BZ=np.zeros(shape),
        BPhi=np.ones(shape),
        nfp=nfp,
    )
    delta = VectorFieldCylind.zero_like(base, label="zero_delta")
    return base, delta


def test_field_period_cache_requires_explicit_nfp():
    R = np.linspace(0.8, 1.2, 3)
    Z = np.linspace(-0.1, 0.1, 3)
    Phi = np.linspace(0.0, 1.0, 4, endpoint=False)
    arr = np.zeros((R.size, Z.size, Phi.size))

    with pytest.raises(TypeError):
        field_period_cache_from_components(R, Z, Phi, BR=arr, BZ=arr, BPhi=arr)


def test_cyna_trace_uses_field_period_phi_grid_period():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.trace_orbit_along_phi is None:
        pytest.skip("cyna orbit tracing is unavailable")

    nfp = 2
    period = 2.0 * np.pi / nfp
    R = np.linspace(0.8, 1.2, 9)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, period, 32, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    eps = 0.01
    base = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=eps * np.cos(2.0 * PP),
        BZ=np.zeros_like(RR),
        BPhi=np.ones_like(RR),
        nfp=nfp,
    )

    R_t, Z_t, _phi_t, _DP_t, alive = trace_orbit_along_phi_field(
        base,
        1.0,
        0.0,
        0.0,
        2.0 * np.pi,
        0.01,
        dphi_out=2.0 * np.pi,
    )

    assert bool(alive[-1])
    np.testing.assert_allclose(R_t[-1], 1.0, atol=5.0e-4)
    np.testing.assert_allclose(Z_t[-1], 0.0, atol=1.0e-12)


def test_find_fixed_points_batch_field_matches_current_cyna_abi():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.find_fixed_points_batch is None:
        pytest.skip("cyna fixed-point search is unavailable")

    R0 = 1.0
    Z0 = 0.0
    omega = 0.25
    R = np.linspace(0.8, 1.2, 17)
    Z = np.linspace(-0.2, 0.2, 17)
    Phi = np.array([0.0])
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    field = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=-omega * (ZZ - Z0) / RR,
        BZ=omega * (RR - R0) / RR,
        BPhi=np.ones_like(RR),
    )

    R_out, Z_out, residual, converged, _DPm, _eig_r, _eig_i, point_type = (
        find_fixed_points_batch_field(
            field,
            np.array([1.04]),
            np.array([0.03]),
            0.0,
            1,
            1,
            0.02,
            max_iter=40,
            tol=1.0e-10,
            n_threads=1,
        )
    )

    assert bool(converged[0])
    assert int(point_type[0]) == 0
    assert residual[0] < 1.0e-10
    np.testing.assert_allclose([R_out[0], Z_out[0]], [R0, Z0], atol=1.0e-8)


def test_axis_cycle_shift_zero_delta_if_cyna_available():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.compute_cycle_perturbation_shift is None:
        pytest.skip("cyna cycle perturbation shift is unavailable")

    nfp = 5
    base, delta = _constant_fields(nfp=nfp)
    phi = np.asarray(base.Phi, dtype=float)
    axis_R = np.full(phi.shape, 1.0)
    axis_Z = np.zeros(phi.shape)

    shifted = axis_cycle_shift_from_fields(
        axis_R,
        axis_Z,
        phi,
        base,
        delta,
        nfp=nfp,
        field_periods=1.0,
        steps_per_field_period=32,
        fd_eps=1.0e-4,
    )

    np.testing.assert_allclose(shifted.axis_R, axis_R, atol=1.0e-13)
    np.testing.assert_allclose(shifted.axis_Z, axis_Z, atol=1.0e-13)
    np.testing.assert_allclose(shifted.cycle_shift.delta_X_cyc, 0.0, atol=1.0e-13)
    assert shifted.diagnostics["method"] == "cyna_evolve_delta_X_cycle_along_orbit"


def test_axis_cycle_shift_adds_delta_to_each_query_axis_sample(monkeypatch):
    import pyna.toroidal.flt.cycle_shift as mod

    class FakeCycleShift:
        phi = np.linspace(0.0, 2.0 * np.pi / 3.0, 5)
        R = np.linspace(1.0, 1.1, 5)
        Z = np.linspace(-0.2, 0.2, 5)
        delta_X_cyc = np.column_stack([
            np.full(5, 0.01),
            np.full(5, -0.02),
        ])
        delta_X_cyc0 = np.asarray([0.01, -0.02])
        periodic_residual = np.asarray([0.0, 0.0])
        alive = np.ones(5, dtype=bool)

    monkeypatch.setattr(mod, "cycle_shift_from_fields", lambda *args, **kwargs: FakeCycleShift())
    phi = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    axis_R = 1.5 + 0.2 * np.cos(phi)
    axis_Z = -0.1 + 0.05 * np.sin(phi)

    shifted = mod.axis_cycle_shift_from_fields(
        axis_R,
        axis_Z,
        phi,
        object(),
        object(),
        nfp=3,
    )

    np.testing.assert_allclose(shifted.axis_R, axis_R + 0.01)
    np.testing.assert_allclose(shifted.axis_Z, axis_Z - 0.02)
    assert shifted.diagnostics["axis_shift_norm_max_m"] == pytest.approx(np.hypot(0.01, 0.02))


def test_cycle_points_shift_zero_delta_if_cyna_available():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.compute_cycle_perturbation_shift is None:
        pytest.skip("cyna cycle perturbation shift is unavailable")

    nfp = 5
    base, delta = _constant_fields(nfp=nfp)
    seeds = np.asarray([[0.95, -0.02], [1.05, 0.03]], dtype=float)
    phi_sections = [0.0, 0.25 * 2.0 * np.pi / nfp]

    shifted = cycle_points_shift_from_fields(
        seeds,
        phi_sections,
        base,
        delta,
        nfp=nfp,
        field_periods=1.0,
        steps_per_field_period=32,
        fd_eps=1.0e-4,
    )

    assert len(shifted.sections) == 2
    np.testing.assert_allclose(shifted.sections[0], seeds, atol=1.0e-13)
    np.testing.assert_allclose(shifted.sections[1], seeds, atol=1.0e-13)
    assert shifted.diagnostics["n_cycles"] == 2
