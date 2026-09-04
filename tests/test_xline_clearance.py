"""Tests for production periodic-X-line wall-clearance validation."""

from __future__ import annotations

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.xline_clearance import validate_periodic_xline_clearance
from pyna.toroidal.geometry import ToroidalComponentSurface


def _branch(anchor=(1.05, 0.0), *, section_phi=0.2, map_span=0.8):
    return ManifoldBranchReference(
        origin_label="xline",
        stability="unstable",
        seed_side=1,
        section_phi=section_phi,
        map_span=map_span,
        anchor_RZ_m=np.asarray(anchor),
        direction_RZ=np.asarray([1.0, 0.0]),
        expansion=1.2,
        orientation_reversing=False,
        seed_orders=np.asarray([0]),
        seed_distances_m=np.asarray([0.01]),
        points_RZ_m=np.asarray([[[anchor[0] + 0.01, anchor[1]]]]),
        available=np.asarray([[True]]),
    )


def _wall(nfp=1):
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, 8, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    R = np.broadcast_to(1.0 + 0.3 * np.cos(theta), (phi.size, theta.size))
    Z = np.broadcast_to(0.3 * np.sin(theta), (phi.size, theta.size))
    return ToroidalComponentSurface(phi=phi, R=R, Z=Z, nfp=nfp)


def _constant_orbit_trace(
    field,
    R0,
    Z0,
    phi_start,
    phi_end,
    DPhi,
    **kwargs,
):
    del field, DPhi, kwargs
    phi = np.linspace(phi_start, phi_end, 17)
    R = np.full(phi.size, R0)
    Z = np.full(phi.size, Z0)
    DPm = np.broadcast_to(np.eye(2), (phi.size, 2, 2)).copy()
    return R, Z, phi, DPm, np.ones(phi.size, dtype=bool)


def test_production_clearance_accepts_closed_inside_orbit():
    report = validate_periodic_xline_clearance(
        object(),
        _branch(),
        _wall(),
        DPhi=0.01,
        required_minimum_clearance_m=0.2,
        maximum_closure_error_m=1.0e-6,
        trace_function=_constant_orbit_trace,
    )

    assert report.accepted
    assert report.trace_complete
    assert report.rejection_reason is None
    assert report.minimum_sample_index is not None
    assert report.minimum_clearance_m == pytest.approx(0.25, abs=2.0e-4)
    assert report.closure_error_m == pytest.approx(0.0)


def test_production_clearance_uses_native_cyna_grid_trace():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")
    radius = np.linspace(0.5, 1.5, 25)
    height = np.linspace(-0.5, 0.5, 25)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rr, _, _ = np.meshgrid(radius, height, phi, indexing="ij")
    field = VectorFieldCylind(
        R=radius,
        Z=height,
        Phi=phi,
        BR=np.zeros_like(rr),
        BZ=np.zeros_like(rr),
        BPhi=rr,
        nfp=1,
    )

    report = validate_periodic_xline_clearance(
        field,
        _branch(),
        _wall(),
        DPhi=0.01,
        required_minimum_clearance_m=0.2,
        maximum_closure_error_m=1.0e-10,
    )

    assert report.accepted
    assert report.point_RZPhi_m_rad.shape[0] == 81
    assert report.minimum_clearance_m == pytest.approx(0.25, abs=2.0e-4)


def test_production_clearance_preserves_sign_across_nfp_seams():
    report = validate_periodic_xline_clearance(
        object(),
        _branch(map_span=2.0 * np.pi + 0.4),
        _wall(nfp=2),
        DPhi=0.01,
        required_minimum_clearance_m=0.2,
        maximum_closure_error_m=1.0e-6,
        trace_function=_constant_orbit_trace,
    )

    assert report.accepted
    assert report.point_RZPhi_m_rad[-1, 2] > 2.0 * np.pi
    np.testing.assert_allclose(report.signed_clearance_m, 0.25, atol=2.0e-4)


def test_production_clearance_rejects_outside_orbit_with_signed_distance():
    report = validate_periodic_xline_clearance(
        object(),
        _branch(anchor=(1.35, 0.0)),
        _wall(),
        DPhi=0.01,
        required_minimum_clearance_m=0.0,
        maximum_closure_error_m=1.0e-6,
        trace_function=_constant_orbit_trace,
    )

    assert not report.accepted
    assert report.rejection_reason == "production_xline_clearance_below_limit"
    assert report.minimum_clearance_m == pytest.approx(-0.05, abs=2.0e-4)


def test_production_clearance_rejects_closure_drift_and_incomplete_trace():
    def drifting_trace(*args, **kwargs):
        R, Z, phi, DPm, alive = _constant_orbit_trace(*args, **kwargs)
        R[-1] += 2.0e-3
        return R, Z, phi, DPm, alive

    closure_report = validate_periodic_xline_clearance(
        object(),
        _branch(),
        _wall(),
        DPhi=0.01,
        required_minimum_clearance_m=0.0,
        maximum_closure_error_m=1.0e-3,
        trace_function=drifting_trace,
    )
    assert not closure_report.accepted
    assert (
        closure_report.rejection_reason
        == "production_orbit_closure_exceeds_limit"
    )

    def incomplete_trace(*args, **kwargs):
        R, Z, phi, DPm, alive = _constant_orbit_trace(*args, **kwargs)
        alive[-1] = False
        return R, Z, phi, DPm, alive

    incomplete_report = validate_periodic_xline_clearance(
        object(),
        _branch(),
        _wall(),
        DPhi=0.01,
        required_minimum_clearance_m=0.0,
        maximum_closure_error_m=1.0e-3,
        trace_function=incomplete_trace,
    )
    assert not incomplete_report.accepted
    assert incomplete_report.rejection_reason == "production_orbit_trace_incomplete"
    assert np.isinf(incomplete_report.closure_error_m)


def test_production_clearance_validates_configuration():
    with pytest.raises(ValueError, match="DPhi"):
        validate_periodic_xline_clearance(
            object(),
            _branch(),
            _wall(),
            DPhi=0.0,
            required_minimum_clearance_m=0.0,
            maximum_closure_error_m=1.0e-3,
            trace_function=_constant_orbit_trace,
        )
    with pytest.raises(ValueError, match="required_minimum_clearance_m"):
        validate_periodic_xline_clearance(
            object(),
            _branch(),
            _wall(),
            DPhi=0.01,
            required_minimum_clearance_m=-1.0,
            maximum_closure_error_m=1.0e-3,
            trace_function=_constant_orbit_trace,
        )
