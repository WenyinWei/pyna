"""First-hit identity and independently sampled production leg margins."""

from dataclasses import replace

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_leg_clearance import validate_manifold_leg_clearance
from pyna.topo.manifold_strike_contracts import ManifoldStrikeLabel, ManifoldStrikeMatch
from pyna.toroidal.geometry import ToroidalComponentSurface
from pyna.toroidal.geometry import project_points_to_toroidal_surface
from pyna.topo.xline_clearance import _signed_clearance_from_projection


def _case(stability="unstable", seed_side=1):
    stable = stability == "stable"
    direction = np.array([0., 1.]) if stable else np.array([1., 0.])
    distances = np.array([.01, .02])
    branch = ManifoldBranchReference(
        origin_label="leg:X0", stability=stability, seed_side=seed_side,
        section_phi=6.1, map_span=np.pi, anchor_RZ_m=np.array([1., 0.]),
        direction_RZ=direction, expansion=np.exp(.35 * np.pi),
        orientation_reversing=False, seed_orders=np.array([4, 9]),
        seed_distances_m=distances,
        points_RZ_m=(np.array([1., 0.]) + seed_side * distances[:, None] * direction)[None],
        available=np.ones((1, 2), dtype=bool),
    )
    phi = np.linspace(0, np.pi, 8, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    wall = ToroidalComponentSurface(
        phi=phi, R=np.broadcast_to(1 + .3 * np.cos(theta), (8, 128)),
        Z=np.broadcast_to(.3 * np.sin(theta), (8, 128)), nfp=2,
    )
    sign = -1 if stable else 1
    endpoint = np.array([1., 0.]) + seed_side * .3 * direction
    matches = tuple(ManifoldStrikeMatch(
        label=ManifoldStrikeLabel(branch.label, "-" if stable else "+", int(order)),
        point_RZPhi_m_rad=np.r_[endpoint, branch.section_phi + sign * np.log(.3 / distance) / .35],
        target_distance_m=0., connection_length_m=10., bundle_seed_index=i,
        distance_mode="rz",
    ) for i, (order, distance) in enumerate(zip(branch.seed_orders, distances)))
    return branch, wall, matches


def _strike_trace(matches, *, missing=False, phase_shift=0.):
    def trace(field, R, Z, phi_start, max_turns, DPhi, wall_phi, wall_R, wall_Z, *, direction, extend_phi):
        suffix = "plus" if direction == "+" else "minus"
        hits = np.array([match.point_RZPhi_m_rad for match in matches])
        hits[:, 2] += phase_shift
        return {f"hit_{suffix}": hits, f"term_{suffix}": np.array([1, 0 if missing else 1]),
                f"Lc_{suffix}": np.ones(2) * 10.}
    return trace


def _trajectory(*, mode=None):
    def trace(field, R0, Z0, start, end, step, **kwargs):
        phi = np.linspace(start, end, 101)
        R = 1 + (R0 - 1) * np.exp(.35 * (phi - start))
        Z = Z0 * np.exp(-.35 * (phi - start))
        alive = np.ones(len(phi), dtype=bool)
        if mode == "incomplete":
            alive[30] = False
        elif mode == "endpoint":
            R[-1] += .01
        elif mode == "outside":
            R[40] = 1.31
        return R, Z, phi, None, alive
    return trace


def _validate(stability="unstable", **overrides):
    branch, wall, matches = _case(stability)
    kwargs = dict(
        field=object(), branch_reference=branch, wall=wall, previous_matches=matches,
        required_minimum_clearance_m=.01, terminal_exclusion_fraction=.2,
        maximum_hit_displacement_m=.01, maximum_hit_phase_shift_rad=.1,
        maximum_endpoint_error_m=1e-5, maximum_projection_distance_m=1e-5,
        DPhi=.01, max_turns=4, strike_trace_function=_strike_trace(matches),
        trajectory_trace_function=_trajectory(),
    )
    kwargs.update(overrides)
    return validate_manifold_leg_clearance(**kwargs)


@pytest.mark.parametrize("stability", ["stable", "unstable"])
def test_labelled_leg_clearance_handles_signed_multiturn_seams(stability):
    report = _validate(stability)
    assert report.accepted
    assert tuple(leg.label.seed_order for leg in report.legs) == (4, 9)
    assert report.minimum_clearance_m > .1
    for leg, match in zip(report.legs, report.strike_matches):
        start = 6.1
        fraction = (leg.point_RZPhi_m_rad[:, 2] - start) / (match.point_RZPhi_m_rad[2] - start)
        assert np.all(fraction < .8)
        assert np.all(leg.signed_clearance_m > 0)
        assert not leg.signed_clearance_m.flags.writeable


@pytest.mark.parametrize("mode,reason", [
    ("incomplete", "production_leg_trace_incomplete"),
    ("endpoint", "production_leg_endpoint_mismatch"),
    ("outside", "production_leg_clearance_below_limit"),
])
def test_leg_clearance_rejects_bad_trajectory(mode, reason):
    report = _validate(trajectory_trace_function=_trajectory(mode=mode))
    assert not report.accepted
    assert report.rejection_reason == reason
    assert len(report.legs) == 2


def test_leg_clearance_rejects_missing_seed_and_same_hit_on_wrong_turn():
    _, _, matches = _case()
    missing = _validate(strike_trace_function=_strike_trace(matches, missing=True))
    assert not missing.accepted
    assert missing.legs[0].accepted
    assert "strike_unresolved" in missing.legs[1].rejection_reason
    wrong_turn = _validate(strike_trace_function=_strike_trace(matches, phase_shift=-2 * np.pi))
    assert wrong_turn.rejection_reason == "production_leg_hit_phase_shift_exceeds_limit"


def test_clearance_limit_and_order_are_mandatory():
    report = _validate(required_minimum_clearance_m=.29)
    assert not report.accepted
    with pytest.raises(ValueError, match="terminal_exclusion_fraction"):
        _validate(terminal_exclusion_fraction=0.)
    _, _, matches = _case()
    with pytest.raises(ValueError, match="ordered branch identity"):
        _validate(previous_matches=matches[::-1])
    with pytest.raises(ValueError, match="required minimum"):
        replace(_validate(), required_minimum_clearance_m=.3)


@pytest.mark.parametrize("reverse", [False, True])
def test_concave_wall_vertex_clearance_uses_containment_not_normal_roundoff(reverse):
    phi = np.linspace(0, np.pi, 32, endpoint=False)
    section_R = np.array([1., 4., 4., 2., 2., 1.])
    section_Z = np.array([0., 0., 1., 1., 3., 3.])
    if reverse:
        section_R, section_Z = section_R[::-1], section_Z[::-1]
    wall = ToroidalComponentSurface(
        phi=phi, R=np.broadcast_to(section_R, (len(phi), 6)),
        Z=section_Z + .2 * np.sin(2 * phi[:, None]), nfp=2,
    )
    # These interior points lie on the extension of an edge at the reflex
    # vertex. Their normal offset can vanish while true clearance is 0.25 m.
    unwrapped = phi - 4 * np.pi
    points = np.column_stack((np.full(len(phi), 1.75),
                              1 + .2 * np.sin(2 * phi), unwrapped))
    projection = project_points_to_toroidal_surface(*points.T, wall)
    np.testing.assert_allclose(
        _signed_clearance_from_projection(points, projection), .25, atol=1e-13,
    )
    outside = points.copy()
    outside[:, 0] = 2.25
    outside[:, 1] += .25
    projection = project_points_to_toroidal_surface(*outside.T, wall)
    np.testing.assert_allclose(projection.signed_distance, -.25, atol=1e-13)


@pytest.mark.parametrize("stability,side", [("unstable", 1), ("stable", 1), ("stable", -1)])
def test_native_cyna_leg_clearance_and_sampling_convergence(stability, side):
    import pyna._cyna as cyna
    if not cyna.is_available():
        pytest.skip("Cyna is unavailable")
    branch, wall, matches = _case(stability, side)
    R, Z, phi = np.linspace(.5, 1.5, 33), np.linspace(-.5, .5, 33), np.linspace(0, np.pi, 12, endpoint=False)
    rr, zz, _ = np.meshgrid(R, Z, phi, indexing="ij")
    field = VectorFieldCylind(R=R, Z=Z, Phi=phi, BR=.35 * (rr - 1), BZ=-.35 * zz, BPhi=rr, nfp=2)
    reports = []
    for step in (.02, .01):
        reports.append(_validate(
            stability, field=field, branch_reference=branch, wall=wall, previous_matches=matches,
            DPhi=step, strike_trace_function=None, trajectory_trace_function=None,
            maximum_endpoint_error_m=1e-4,
        ))
    assert all(report.accepted for report in reports), [r.rejection_reason for r in reports]
    assert abs(reports[0].minimum_clearance_m - reports[1].minimum_clearance_m) < .002
