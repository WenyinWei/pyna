import numpy as np
import pytest

from pyna.toroidal.control.boundary_nonlinear_validation import (
    BoundaryNonlinearValidationState,
    DPKGrowthValidation,
    FixedPointChainValidation,
    HealedSurfaceSectionChart,
    SeparatrixEnvelopeBranchPair,
    LocalManifoldMeasurementLine,
    MagneticAxisCoreShift,
    ManifoldBranchSamples,
    ManifoldTraceProvenance,
    NewtonFixedPointState,
    StableUnstableBranchPair,
    boundary_control_key,
    boundary_nonlinear_observable_builder,
    boundary_nonlinear_validation_observables,
    fixed_point_chain_coherence,
    fixed_point_chain_helical_coherence,
    fixed_point_chain_helical_phase,
    fixed_point_chain_helical_phasor,
    fixed_point_chain_phase,
    greene_residue,
    manifold_branches_from_trace,
    nardon_fixed_point_phase_closure_error,
    periodic_phase_difference,
    polyline_line_intersections,
    separatrix_width_from_manifolds,
    solve_periodic_point_phase_response,
    stable_unstable_splitting_from_manifolds,
)
from pyna.toroidal.control.boundary_plasma_response import (
    BoundaryPlasmaResponseInput,
    BoundaryPlasmaResponseSnapshot,
)
from pyna.toroidal.control.boundary_topology_design import BoundaryResponseObservables


TWOPI = 2.0 * np.pi


def _fixed_point(kind, theta, DPm, *, m=2, label="", section_phi=None):
    return NewtonFixedPointState(
        kind=kind,
        R=5.5 + 0.1 * np.cos(0.0 if theta is None else theta),
        Z=0.1 * np.sin(0.0 if theta is None else theta),
        map_power=m,
        DPm=np.asarray(DPm, dtype=float),
        residual=1.0e-12,
        healed_theta=theta,
        section_phi=section_phi,
        label=label,
    )


def _analytic_healed_chart(*, radial_phase=None, canonical=False, symplectic=False):
    R_axis = 3.1
    Z_axis = -0.2
    elongation_R = 1.4
    elongation_Z = 0.8

    def s_theta_to_x(s_theta):
        values = np.asarray(s_theta, dtype=float)
        s = values[..., 0]
        theta = values[..., 1]
        return np.stack(
            [
                R_axis + elongation_R * s * np.cos(theta),
                Z_axis + elongation_Z * s * np.sin(theta),
            ],
            axis=-1,
        )

    def x_to_s_theta(x_RZ):
        values = np.asarray(x_RZ, dtype=float)
        u = (values[..., 0] - R_axis) / elongation_R
        v = (values[..., 1] - Z_axis) / elongation_Z
        return np.stack([np.hypot(u, v), np.arctan2(v, u)], axis=-1)

    def jacobian_s_theta(s_theta):
        s, theta = np.asarray(s_theta, dtype=float)
        return np.asarray(
            [
                [elongation_R * np.cos(theta), -elongation_R * s * np.sin(theta)],
                [elongation_Z * np.sin(theta), elongation_Z * s * np.cos(theta)],
            ]
        )

    return HealedSurfaceSectionChart(
        s_theta_to_x,
        x_to_s_theta,
        jacobian_s_theta,
        radial_phase=radial_phase,
        canonical=canonical,
        symplectic=symplectic,
        metadata={"surface_family": "analytic_ellipse"},
    )


def _exact_width_metric():
    provenance = ManifoldTraceProvenance(
        n_turns=12,
        integration_step=0.01,
        seed_distances_m=np.asarray([1.0e-4, 2.0e-4]),
        metadata={"seed_spacing": "user"},
    )
    stable = ManifoldBranchSamples(
        label="x0.stable.plus",
        stability="stable",
        points_RZ_m=np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        side=1,
        provenance=provenance,
    )
    unstable = ManifoldBranchSamples(
        label="x0.unstable.plus",
        stability="unstable",
        points_RZ_m=np.asarray([[-1.0, 0.2], [0.0, 0.4], [1.0, 0.6]]),
        side=1,
        provenance=provenance,
    )
    lines = tuple(
        LocalManifoldMeasurementLine(
            label=f"normal{index}",
            origin_RZ_m=np.asarray([R, 0.0]),
            direction_RZ=np.asarray([0.0, 2.0]),
            offset_bounds_m=(-0.1, 0.8),
        )
        for index, R in enumerate((-1.0, 0.0, 1.0))
    )
    metric = separatrix_width_from_manifolds(
        (SeparatrixEnvelopeBranchPair("plus", stable, unstable),),
        lines,
        label="edge",
        require_all=True,
    )
    return stable, unstable, lines, metric


def test_chain_phase_wrap_uses_healed_theta_harmonic_convention():
    m = 5
    epsilon = 1.0e-8
    phase_before = np.pi / m - epsilon
    phase_after = -np.pi / m + epsilon
    theta_before = (phase_before + TWOPI * np.arange(m) / m) % TWOPI
    theta_after = (phase_after + TWOPI * np.arange(m) / m) % TWOPI

    measured_before = fixed_point_chain_phase(theta_before, m)
    measured_after = fixed_point_chain_phase(theta_after, m)

    assert measured_before == pytest.approx(phase_before, abs=2.0e-15)
    assert measured_after == pytest.approx(phase_after, abs=2.0e-15)
    assert periodic_phase_difference(measured_after, measured_before, TWOPI / m) == pytest.approx(
        2.0 * epsilon,
        abs=2.0e-15,
    )
    assert fixed_point_chain_coherence(theta_before, m) == pytest.approx(1.0, abs=1.0e-15)


def test_multisection_helical_phasor_is_invariant_for_nardon_branch():
    m = 5
    n_nardon = -5
    phase = 0.17
    section_phi = np.linspace(0.0, TWOPI / 5.0, 7, endpoint=False)
    branch = TWOPI * np.arange(m) / m
    phi_grid, branch_grid = np.meshgrid(section_phi, branch, indexing="ij")
    theta = phase - (n_nardon / m) * phi_grid + branch_grid

    phasor = fixed_point_chain_helical_phasor(
        theta.ravel(),
        phi_grid.ravel(),
        m,
        n_nardon,
    )

    assert abs(phasor) == pytest.approx(1.0, abs=2.0e-15)
    assert np.angle(phasor) == pytest.approx(m * phase, abs=2.0e-15)
    assert fixed_point_chain_helical_phase(
        theta.ravel(), phi_grid.ravel(), m, n_nardon
    ) == pytest.approx(phase, abs=2.0e-15)
    assert fixed_point_chain_helical_coherence(
        theta.ravel(), phi_grid.ravel(), m, n_nardon
    ) == pytest.approx(1.0, abs=2.0e-15)
    assert fixed_point_chain_helical_coherence(
        theta.ravel(), phi_grid.ravel(), m, +5
    ) < 0.2


def test_nardon_fixed_point_phase_closure_uses_helical_plus_coefficient_phase():
    m = 5
    b0 = 2.0e-3 * np.exp(0.3j)
    helical0 = 0.97 * np.exp(-0.8j)
    coefficient_shift = np.deg2rad(15.0)
    physical_error = np.deg2rad(0.4)
    b = b0 * np.exp(1j * coefficient_shift)
    helical = helical0 * np.exp(1j * (-coefficient_shift + m * physical_error))

    error = nardon_fixed_point_phase_closure_error(
        b,
        b0,
        helical,
        helical0,
        m,
    )

    assert error == pytest.approx(physical_error)
    with pytest.raises(ValueError, match="nonzero"):
        nardon_fixed_point_phase_closure_error(b, b0, 0.0j, helical0, m)


def test_multisection_validated_chain_requires_and_uses_explicit_nardon_branch():
    m = 5
    n0 = 5
    n_nardon = -n0
    phase = 0.17
    elliptic = np.asarray([[0.8, -0.6], [0.6, 0.8]])
    hyperbolic = np.diag([2.0, 0.5])
    phi = np.linspace(0.0, TWOPI / n0, m, endpoint=False)
    theta = phase - (n_nardon / m) * phi + TWOPI * np.arange(m) / m
    o_points = tuple(
        _fixed_point(
            "O",
            theta_i,
            elliptic,
            m=m,
            label=f"o{index}",
            section_phi=phi_i,
        )
        for index, (theta_i, phi_i) in enumerate(zip(theta, phi))
    )
    x_point = _fixed_point("X", None, hyperbolic, m=m, label="x0")

    with pytest.raises(ValueError, match="nardon_n"):
        FixedPointChainValidation(
            label="m5.n5",
            m=m,
            n=n0,
            fixed_points=o_points + (x_point,),
        )

    chain = FixedPointChainValidation(
        label="m5.n5",
        m=m,
        n=n0,
        fixed_points=o_points + (x_point,),
        metadata={"nardon_n": n_nardon},
    )

    assert chain.phase == pytest.approx(phase, abs=2.0e-15)
    assert chain.resonant_phase == pytest.approx(m * phase, abs=2.0e-15)
    assert chain.coherence == pytest.approx(1.0, abs=2.0e-15)
    assert chain.metadata["multi_section"] is True


def test_healed_chart_explicit_radial_phase_and_canonical_metadata():
    s_chart = _analytic_healed_chart(radial_phase="s")
    psi_chart = _analytic_healed_chart(canonical=True)
    s_theta = np.asarray([0.7, 0.4])
    z_psi = np.asarray([s_theta[0] ** 2, s_theta[1]])

    assert s_chart.radial_phase == "s"
    assert s_chart.metadata["coordinate_choice"] == "z=(s, theta*)"
    assert s_chart.metadata["canonical"] is False
    assert psi_chart.radial_phase == "psi"
    assert psi_chart.metadata["coordinate_choice"] == "z=(psi=s^2, theta*)"
    assert psi_chart.metadata["canonical"] is True
    assert psi_chart.metadata["symplectic"] is True
    np.testing.assert_allclose(psi_chart.z_to_x(z_psi), s_chart.z_to_x(s_theta))
    np.testing.assert_allclose(
        psi_chart.jacobian(z_psi),
        s_chart.jacobian(s_theta) @ np.diag([1.0 / (2.0 * s_theta[0]), 1.0]),
    )

    with pytest.raises(ValueError, match="not automatically a canonical"):
        _analytic_healed_chart(radial_phase="s", canonical=True)


def test_periodic_point_phase_response_is_coordinate_invariant_between_s_and_psi():
    s_chart = _analytic_healed_chart(radial_phase="s")
    psi_chart = _analytic_healed_chart(canonical=True)
    z_s = np.asarray([0.7, 0.4])
    x0 = s_chart.z_to_x(z_s)
    J_s = s_chart.jacobian(z_s)
    DP_s = np.asarray([[0.25, 0.12], [-0.18, 0.31]])
    expected_delta_s_theta = np.asarray([0.03, -0.12])
    forcing_s = (np.eye(2) - DP_s) @ expected_delta_s_theta
    DP_x = J_s @ DP_s @ np.linalg.inv(J_s)
    forcing_x = J_s @ forcing_s

    response_s = solve_periodic_point_phase_response(
        DP_x,
        forcing_x,
        s_chart,
        x0_RZ_m=x0,
        kind="O",
        map_power=5,
    )
    response_psi = solve_periodic_point_phase_response(
        DP_x,
        forcing_x,
        psi_chart,
        x0_RZ_m=x0,
        kind="O",
        map_power=5,
    )

    np.testing.assert_allclose(response_s.delta_z, expected_delta_s_theta, atol=1.0e-13)
    np.testing.assert_allclose(
        response_psi.delta_z,
        [2.0 * z_s[0] * expected_delta_s_theta[0], expected_delta_s_theta[1]],
        atol=1.0e-13,
    )
    assert response_s.delta_s == pytest.approx(expected_delta_s_theta[0])
    assert response_psi.delta_s == pytest.approx(expected_delta_s_theta[0])
    assert response_psi.delta_psi == pytest.approx(2.0 * z_s[0] * expected_delta_s_theta[0])
    np.testing.assert_allclose(
        response_s.geometric_displacement_RZ_m,
        response_psi.geometric_displacement_RZ_m,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(response_s.DPk_z, DP_s, atol=2.0e-14)
    assert response_s.phase_space_valid
    assert response_psi.phase_space_valid
    assert response_s.status == "valid"
    assert response_s.relative_residual < 1.0e-13
    assert response_s.metadata["linearization"] == "phase-space periodic-point response"
    assert "not a phase response" in response_s.metadata["geometric_displacement_semantics"]


def test_periodic_point_phase_response_wraps_healed_theta_endpoint():
    chart = _analytic_healed_chart(radial_phase="s")
    z0 = np.asarray([0.8, TWOPI - 0.05])
    J = chart.jacobian(z0)
    DP_z = np.asarray([[0.2, 0.0], [0.0, 0.4]])
    expected_delta = np.asarray([0.0, 0.2])
    response = solve_periodic_point_phase_response(
        J @ DP_z @ np.linalg.inv(J),
        J @ ((np.eye(2) - DP_z) @ expected_delta),
        chart,
        z0=z0,
    )

    assert response.delta_theta_star == pytest.approx(0.2)
    assert response.delta_theta_star_wrapped == pytest.approx(0.2)
    assert response.theta_star1_wrapped == pytest.approx(0.15)
    assert 0.0 <= response.z1[1] < TWOPI


def test_periodic_point_phase_response_reports_rank_deficiency_and_chart_singularity():
    chart = _analytic_healed_chart(radial_phase="s")
    z0 = np.asarray([0.7, 0.2])
    J = chart.jacobian(z0)
    DP_z = np.diag([1.0 - 1.0e-14, 0.0])
    forcing_z = np.asarray([1.0, 2.0])
    response = solve_periodic_point_phase_response(
        J @ DP_z @ np.linalg.inv(J),
        J @ forcing_z,
        chart,
        z0=z0,
        svd_rcond=1.0e-10,
    )

    assert response.status == "rank_deficient"
    assert not response.phase_space_valid
    assert response.regularized
    assert response.svd_rank == 1
    assert response.solve_condition_number > 1.0e13
    assert response.relative_residual > 0.4

    singular_chart = HealedSurfaceSectionChart(
        chart.s_theta_to_x,
        chart.x_to_s_theta,
        lambda _z: np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        radial_phase="s",
    )
    with pytest.raises(ValueError, match="Jacobian is singular"):
        solve_periodic_point_phase_response(
            np.zeros((2, 2)),
            np.zeros(2),
            singular_chart,
            z0=z0,
        )


def test_greene_residue_and_dpk_growth_on_exact_linear_maps():
    angle = np.pi / 3.0
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    hyperbolic = np.diag([2.0, 0.5])

    assert greene_residue(np.eye(2)) == pytest.approx(0.0)
    assert greene_residue(rotation) == pytest.approx(0.25)
    assert greene_residue(hyperbolic) == pytest.approx(-0.125)

    k = 4
    rate = 0.55
    reciprocal_growth = DPKGrowthValidation(
        label="reciprocal",
        k=k,
        DPk=np.diag([np.exp(rate * k), np.exp(-rate * k)]),
    )
    assert reciprocal_growth.svd_growth_factor == pytest.approx(np.exp(rate))
    assert reciprocal_growth.eigen_growth_factor == pytest.approx(np.exp(rate))
    assert reciprocal_growth.svd_log_growth_per_iter == pytest.approx(rate)

    nonnormal = DPKGrowthValidation(
        label="nonnormal",
        k=1,
        DPk=np.asarray([[1.0, 5.0], [0.0, 1.0]]),
    )
    assert nonnormal.eigen_growth_factor == pytest.approx(1.0)
    assert nonnormal.svd_growth_factor > 5.0


def test_newton_state_rejects_failed_or_misclassified_fixed_points():
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    hyperbolic = np.diag([2.0, 0.5])
    common = {
        "R": 5.5,
        "Z": 0.0,
        "map_power": 2,
        "residual": 1.0e-12,
    }

    with pytest.raises(ValueError, match="must be converged"):
        NewtonFixedPointState(kind="O", DPm=rotation, converged=False, **common)
    with pytest.raises(ValueError, match="elliptic"):
        NewtonFixedPointState(kind="O", DPm=hyperbolic, **common)
    with pytest.raises(ValueError, match="hyperbolic"):
        NewtonFixedPointState(kind="X", DPm=rotation, **common)
    with pytest.raises(ValueError, match="exceeds residual_tolerance"):
        NewtonFixedPointState(
            kind="X",
            DPm=hyperbolic,
            residual=1.0e-5,
            residual_tolerance=1.0e-8,
            R=5.5,
            Z=0.0,
            map_power=2,
        )


def test_public_manifold_payload_is_split_into_stable_unstable_sides():
    payload = {
        "s_R": np.asarray([-1.0, 0.0, -1.0, 0.0]),
        "s_Z": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "s_point_side": np.asarray([-1.0, -1.0, 1.0, 1.0]),
        "s_seed_distance": np.asarray([1.0e-4, 2.0e-4]),
        "s_seed_side": np.asarray([-1.0, 1.0]),
        "s_generation": np.asarray([0, 1, 0, 1]),
        "u_R": np.asarray([-1.0, 0.0, -1.0, 0.0]),
        "u_Z": np.asarray([0.2, 0.2, 1.2, 1.2]),
        "u_point_side": np.asarray([-1.0, -1.0, 1.0, 1.0]),
        "u_seed_distance": np.asarray([1.5e-4, 2.5e-4]),
        "u_seed_side": np.asarray([-1.0, 1.0]),
        "u_generation": np.asarray([0, 1, 0, 1]),
        "origin_phi": 0.25,
        "manifold_origin_label": "orbit3:P1",
        "manifold_field_period": TWOPI,
        "manifold_field_period_source": "fixed_point_metadata",
        "seed_spacing": "user",
    }

    branches = manifold_branches_from_trace(payload, n_turns=8, integration_step=0.02)

    assert len(branches) == 4
    assert {(branch.stability, branch.side) for branch in branches} == {
        ("stable", -1),
        ("stable", 1),
        ("unstable", -1),
        ("unstable", 1),
    }
    assert all(branch.points_RZ_m.shape == (2, 2) for branch in branches)
    assert all(branch.provenance.n_turns == 8 for branch in branches)
    assert all(
        branch.provenance.source == "pyna.toroidal.flt.trace_fixed_point_manifolds_field"
        for branch in branches
    )


def test_transverse_manifold_intersections_define_width_in_metres():
    stable, _unstable, lines, metric = _exact_width_metric()

    shared_vertex_hits = polyline_line_intersections(stable.points_RZ_m, lines[1])
    assert len(shared_vertex_hits) == 1
    np.testing.assert_allclose(shared_vertex_hits[0].point_RZ_m, [0.0, 0.0], atol=0.0)
    assert shared_vertex_hits[0].offset_m == pytest.approx(0.0)
    near_tangent = np.asarray([[-1.0e-6, -0.5], [1.0e-6, 0.5]])
    transverse_only = LocalManifoldMeasurementLine(
        label="transverse_only",
        origin_RZ_m=np.asarray([0.0, 0.0]),
        direction_RZ=np.asarray([0.0, 1.0]),
        offset_bounds_m=(-1.0, 1.0),
        minimum_crossing_sine=1.0e-3,
    )
    assert polyline_line_intersections(near_tangent, transverse_only) == ()

    np.testing.assert_allclose(metric.widths_m, [0.2, 0.4, 0.6], atol=1.0e-15)
    assert metric.median_m == pytest.approx(0.4)
    assert metric.spread_m == pytest.approx(0.2)
    assert metric.branch_medians_m == {"plus": pytest.approx(0.4)}
    assert metric.branch_spreads_m == {"plus": pytest.approx(0.2)}
    provenance = metric.resolution_provenance
    assert provenance["algorithm"] == "piecewise_linear_inner_outer_separatrix_envelope"
    assert provenance["quantity_definition"] == (
        "s_outer - s_inner in physical metres; full island envelope width"
    )
    assert provenance["trace_resolution"]["plus"]["inner"]["n_turns"] == 12
    assert provenance["trace_resolution"]["plus"]["inner"]["max_segment_length_m"] == pytest.approx(1.0)


def test_integrable_coincident_manifolds_have_zero_splitting_but_nonzero_envelope_width():
    provenance = ManifoldTraceProvenance(n_turns=16, integration_step=0.005)

    def branch(label, stability, height):
        return ManifoldBranchSamples(
            label=label,
            stability=stability,
            points_RZ_m=np.asarray([[-1.0, height], [1.0, height]]),
            section_phi=0.0,
            provenance=provenance,
        )

    inner_stable = branch("inner.stable", "stable", -0.5)
    inner_unstable = branch("inner.unstable", "unstable", -0.5)
    outer_stable = branch("outer.stable", "stable", 0.5)
    outer_unstable = branch("outer.unstable", "unstable", 0.5)
    wrong_section = ManifoldBranchSamples(
        label="inner.unstable.wrong_section",
        stability="unstable",
        points_RZ_m=inner_unstable.points_RZ_m,
        section_phi=0.25,
        provenance=provenance,
    )
    with pytest.raises(ValueError, match="same toroidal section"):
        StableUnstableBranchPair("invalid", inner_stable, wrong_section)
    line = LocalManifoldMeasurementLine(
        label="healed_theta_0",
        origin_RZ_m=np.asarray([0.0, 0.0]),
        direction_RZ=np.asarray([0.0, 1.0]),
        offset_bounds_m=(-0.75, 0.75),
        kind="radial",
        section_phi=0.0,
    )

    splitting = stable_unstable_splitting_from_manifolds(
        (
            StableUnstableBranchPair("inner", inner_stable, inner_unstable),
            StableUnstableBranchPair("outer", outer_stable, outer_unstable),
        ),
        (line,),
        label="integrable",
        require_all=True,
    )
    envelope = separatrix_width_from_manifolds(
        (
            SeparatrixEnvelopeBranchPair(
                "full",
                inner=inner_stable,
                outer=outer_stable,
            ),
        ),
        (line,),
        label="integrable",
        require_all=True,
    )

    np.testing.assert_allclose(splitting.splittings_m, [0.0, 0.0], atol=0.0)
    assert splitting.median_m == pytest.approx(0.0)
    assert envelope.median_m == pytest.approx(1.0)
    assert envelope.branches[0].spread_m == pytest.approx(0.0)
    assert "not island width" in splitting.resolution_provenance["quantity_definition"]
    assert envelope.resolution_provenance["forbidden_width_surrogates"] == (
        "O-X distance",
        "Greene residue",
    )


def test_observable_labels_are_wrap_safe_and_include_available_metrics():
    angle = np.pi / 3.0
    elliptic = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    hyperbolic = np.diag([2.0, 0.5])
    chain = FixedPointChainValidation(
        label="m2.n1",
        m=2,
        n=1,
        fixed_points=(
            _fixed_point("O", 0.1, elliptic, label="o0"),
            _fixed_point("O", np.pi + 0.1, elliptic, label="o1"),
            _fixed_point("X", None, hyperbolic, label="x0"),
        ),
    )
    stable, unstable, lines, width = _exact_width_metric()
    splitting = stable_unstable_splitting_from_manifolds(
        (StableUnstableBranchPair("edge", stable, unstable),),
        lines,
        label="edge",
        require_all=True,
    )
    shift = MagneticAxisCoreShift(
        axis_reference_RZ_m=np.asarray([1.0, 0.0]),
        axis_current_RZ_m=np.asarray([1.03, 0.04]),
        core_reference_RZ_m=np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        core_current_RZ_m=np.asarray([[1.0, 0.03], [2.0, 0.04]]),
    )
    growth = DPKGrowthValidation(
        label="edge",
        k=3,
        DPk=np.diag([np.exp(0.2 * 3), np.exp(-0.2 * 3)]),
        alive_fraction=0.75,
    )
    state = BoundaryNonlinearValidationState(
        chains=(chain,),
        axis_core_shift=shift,
        stable_unstable_splittings=(splitting,),
        separatrix_widths=(width,),
        dpk_growth=(growth,),
        wall_metrics={"strike_fraction": 0.2},
        heat_metrics={"peak_flux": 4.0},
        wall_metric_units={"strike_fraction": "1"},
        heat_metric_units={"peak_flux": "W/m^2"},
        open_loss_fraction=0.1,
    )

    rows = boundary_nonlinear_validation_observables(
        state,
        weights={
            "nonlinear.axis.shift_m": 4.0,
            "wall.strike_fraction": 2.0,
        },
    )

    assert rows.labels == (
        "nonlinear.chain.m2.n1.phase_sin",
        "nonlinear.chain.m2.n1.phase_cos",
        "nonlinear.chain.m2.n1.coherence",
        "nonlinear.chain.m2.n1.residue.o_median",
        "nonlinear.chain.m2.n1.residue.x_median",
        "nonlinear.manifold.edge.stable_unstable_splitting_m",
        "nonlinear.manifold.edge.stable_unstable_splitting_spread_m",
        "nonlinear.separatrix.edge.island_envelope_full_width_m",
        "nonlinear.separatrix.edge.island_envelope_full_width_spread_m",
        "nonlinear.axis.shift_m",
        "nonlinear.core.shift_rms_m",
        "nonlinear.core.shift_max_m",
        "nonlinear.dpk.edge.svd_growth_factor",
        "nonlinear.dpk.edge.eigen_growth_factor",
        "nonlinear.dpk.edge.open_loss_fraction",
        "nonlinear.wall.strike_fraction",
        "nonlinear.heat.peak_flux",
        "nonlinear.transport.open_loss_fraction",
    )
    by_label = dict(zip(rows.labels, rows.values))
    assert by_label["nonlinear.chain.m2.n1.phase_sin"] == pytest.approx(np.sin(0.2))
    assert by_label["nonlinear.chain.m2.n1.phase_cos"] == pytest.approx(np.cos(0.2))
    assert by_label["nonlinear.chain.m2.n1.coherence"] == pytest.approx(1.0)
    assert by_label["nonlinear.manifold.edge.stable_unstable_splitting_m"] == pytest.approx(0.4)
    assert by_label["nonlinear.separatrix.edge.island_envelope_full_width_m"] == pytest.approx(0.4)
    assert by_label["nonlinear.axis.shift_m"] == pytest.approx(0.05)
    assert by_label["nonlinear.dpk.edge.svd_growth_factor"] == pytest.approx(np.exp(0.2))
    assert by_label["nonlinear.dpk.edge.open_loss_fraction"] == pytest.approx(0.25)
    assert rows.weights[rows.labels.index("nonlinear.axis.shift_m")] == pytest.approx(4.0)
    assert rows.weights[rows.labels.index("nonlinear.wall.strike_fraction")] == pytest.approx(2.0)
    assert rows.metadata["units"]["nonlinear.separatrix.edge.island_envelope_full_width_m"] == "m"
    assert rows.metadata["units"]["nonlinear.heat.peak_flux"] == "W/m^2"
    assert rows.metadata["phase_observables"] == "sin(m*phase), cos(m*phase)"


def test_cached_extra_builder_keys_snapshot_content_and_controls_by_label():
    calls = []

    def evaluator(snapshot, request):
        calls.append((snapshot, request))
        return BoundaryNonlinearValidationState(
            wall_metrics={"command_sum": float(np.sum(request.controls))}
        )

    cache = {}
    builder = boundary_nonlinear_observable_builder(
        evaluator,
        cache=cache,
        evaluator_key="synthetic-map-v1",
    )
    snapshot_a = BoundaryPlasmaResponseSnapshot(
        metadata={
            "nonlinear_validation_content_key": {
                "equilibrium": "A",
                "grid": np.asarray([1.0, 2.0, 3.0]),
            }
        }
    )
    snapshot_a_copy = BoundaryPlasmaResponseSnapshot(
        metadata={
            "nonlinear_validation_content_key": {
                "equilibrium": "A",
                "grid": np.asarray([1.0, 2.0, 3.0]).copy(),
            }
        }
    )
    request_ba = BoundaryPlasmaResponseInput(
        controls=np.asarray([1.0, 2.0]),
        control_labels=("b", "a"),
    )
    request_ab = BoundaryPlasmaResponseInput(
        controls=np.asarray([2.0, 1.0]),
        control_labels=("a", "b"),
    )

    first = builder(snapshot_a, request_ba)
    second = builder(snapshot_a_copy, request_ab)

    assert isinstance(first, BoundaryResponseObservables)
    assert first.labels == ("nonlinear.wall.command_sum",)
    np.testing.assert_allclose(second.values, first.values)
    assert len(calls) == 1
    assert calls[0][0] is snapshot_a
    assert calls[0][1] is request_ba
    assert boundary_control_key(request_ba) == boundary_control_key(request_ab)

    changed_context = BoundaryPlasmaResponseInput(
        controls=np.asarray([2.0, 1.0]),
        control_labels=("a", "b"),
        metadata={"nonlinear_solver": "different"},
    )
    builder(snapshot_a_copy, changed_context)
    changed_control = BoundaryPlasmaResponseInput(
        controls=np.asarray([2.1, 1.0]),
        control_labels=("a", "b"),
    )
    builder(snapshot_a_copy, changed_control)
    snapshot_b = BoundaryPlasmaResponseSnapshot(
        metadata={"nonlinear_validation_content_key": {"equilibrium": "B"}}
    )
    builder(snapshot_b, request_ab)
    assert len(calls) == 4
    assert len(cache) == 4


def test_disabled_cache_does_not_evaluate_injected_key_builders():
    key_calls = []

    def evaluator(snapshot, request):
        return BoundaryNonlinearValidationState(heat_metrics={"total": 1.0})

    def content_key(snapshot):
        key_calls.append(snapshot)
        return "content"

    def control_key(request):
        key_calls.append(request)
        return "control"

    builder = boundary_nonlinear_observable_builder(
        evaluator,
        cache=None,
        content_key_builder=content_key,
        control_key_builder=control_key,
    )

    rows = builder(object(), object())

    assert rows.labels == ("nonlinear.heat.total",)
    assert key_calls == []
