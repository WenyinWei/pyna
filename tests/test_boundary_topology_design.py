from types import SimpleNamespace

import numpy as np
import pytest

from pyna.toroidal.control.boundary_topology_design import (
    BoundaryTopologyDesignTarget,
    boundary_dpk_growth_metrics,
    boundary_response_matrix_diagnostics,
    boundary_response_observables,
    boundary_topology_metrics,
    chaotic_layer_region_observables,
    dpk_growth_observables,
    finite_difference_boundary_response_system,
    resonant_chain_observables,
    score_boundary_topology_payload,
    solve_boundary_response_matrix,
    stack_boundary_linear_response_systems,
    stack_boundary_response_observables,
    wall_heat_region_observables,
)


def _payload(*, strike_shift=0.0):
    return {
        "fixed_points": [
            SimpleNamespace(kind="X"),
            SimpleNamespace(kind="X"),
            SimpleNamespace(kind="O"),
            SimpleNamespace(kind="O"),
        ],
        "edge_state_by_sec": [
            {
                "counts": {
                    "closed_core": 0,
                    "boundary_island": 3,
                    "open_loss": 0,
                    "chaotic_edge": 1,
                }
            }
        ],
        "strike_points": {
            "R": [1.0 + strike_shift, 1.05 + strike_shift, 0.95 + strike_shift],
            "Z": [0.0, 0.03, -0.02],
            "phi": [0.0, 0.04, -0.04],
            "weights": [1.0, 2.0, 1.0],
        },
    }


def test_boundary_topology_design_score_accepts_matching_target():
    payload = _payload()
    target = BoundaryTopologyDesignTarget(
        x_points=2,
        o_points=2,
        boundary_island_fraction=(0.70, 0.80),
        chaotic_fraction=(0.20, 0.30),
        open_loss_fraction=0.0,
        acceptance=1.0e-12,
    )

    score = score_boundary_topology_payload(payload, target)

    assert score.total == pytest.approx(0.0)
    assert score.accepted is True
    assert score.metrics.x_points == 2
    assert score.metrics.o_points == 2
    assert score.metrics.edge_fractions["boundary_island"] == pytest.approx(0.75)
    assert score.metrics.edge_fractions["chaotic_edge"] == pytest.approx(0.25)


def test_boundary_topology_design_score_penalizes_strike_drift_from_reference():
    reference = _payload(strike_shift=0.0)
    candidate = _payload(strike_shift=0.2)
    target = BoundaryTopologyDesignTarget(
        preserve_strike_centroid=True,
        preserve_strike_spread=True,
        strike_scale=0.1,
    )

    reference_score = score_boundary_topology_payload(reference, target, reference_payload=reference)
    drift_score = score_boundary_topology_payload(candidate, target, reference_payload=reference)
    metrics = boundary_topology_metrics(candidate)

    assert reference_score.total == pytest.approx(0.0)
    assert drift_score.components["strike_centroid"] > 0.0
    assert drift_score.total > reference_score.total
    assert metrics.strike_centroid_xyz is not None
    assert metrics.strike_spread is not None


def test_boundary_response_matrix_solver_hits_weighted_target():
    response = [
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
    ]
    current = [0.0, 0.0, 0.0]
    target = [1.0, -2.0, 0.0]

    result = solve_boundary_response_matrix(
        response,
        current,
        target,
        labels=["island.q3", "chaos.edge", "strike.branch"],
    )

    assert result.success is True
    assert result.labels == ("island.q3", "chaos.edge", "strike.branch")
    assert result.controls[0] == pytest.approx(1.0)
    assert result.controls[1] == pytest.approx(-1.0)
    assert result.predicted == pytest.approx(target)
    assert result.diagnostics.rank == 2


def test_boundary_response_matrix_solver_respects_bounds_and_regularization():
    response = [
        [1.0, 0.9],
        [0.0, 0.1],
    ]
    result = solve_boundary_response_matrix(
        response,
        current=[0.0, 0.0],
        target=[2.0, 0.5],
        labels=["strike.heat_peak", "strike.spread"],
        weights={"strike.heat_peak": 4.0, "strike.spread": 1.0},
        bounds=(-0.5, 0.5),
        regularization=0.05,
    )

    assert result.success is True
    assert result.controls[0] <= 0.5 + 1.0e-12
    assert result.controls[1] <= 0.5 + 1.0e-12
    assert result.predicted[0] < 2.0
    assert result.residual[0] < 0.0


def test_boundary_response_matrix_solver_supports_fixed_control_bounds():
    result = solve_boundary_response_matrix(
        np.eye(2),
        current=[0.0, 0.0],
        target=[0.25, 0.75],
        control_labels=["locked", "free"],
        bounds={"locked": (0.25, 0.25), "free": (-1.0, 1.0)},
    )

    assert result.success is True
    np.testing.assert_allclose(result.controls, [0.25, 0.75], atol=1.0e-12)
    np.testing.assert_allclose(result.predicted, [0.25, 0.75], atol=1.0e-12)
    assert result.active_lower_bounds == ("locked",)
    assert result.active_upper_bounds == ("locked",)
    assert "fixed 1 bounded controls" in result.message

    all_fixed = solve_boundary_response_matrix(
        [[1.0]],
        current=[0.0],
        target=[1.0],
        control_labels=["locked"],
        bounds=([0.0], [0.0]),
    )
    assert all_fixed.success is True
    np.testing.assert_allclose(all_fixed.controls, [0.0])
    assert all_fixed.active_lower_bounds == ("locked",)
    assert all_fixed.active_upper_bounds == ("locked",)
    assert all_fixed.message == "all controls fixed by bounds"


def test_boundary_response_matrix_solver_accepts_label_mapped_targets():
    response = np.eye(3)
    current = [1.0, 2.0, 3.0]

    result = solve_boundary_response_matrix(
        response,
        current=current,
        target={"chaos.outer": 5.0},
        labels=["island.width", "chaos.outer", "strike.left"],
        control_labels=["mode_7_3", "mode_8_3", "trim_left"],
    )

    assert result.success is True
    assert result.control_labels == ("mode_7_3", "mode_8_3", "trim_left")
    assert result.controls_by_label == pytest.approx(
        {
            "mode_7_3": 0.0,
            "mode_8_3": 3.0,
            "trim_left": 0.0,
        }
    )
    assert result.target == pytest.approx([1.0, 5.0, 3.0])
    assert result.predicted == pytest.approx([1.0, 5.0, 3.0])
    assert result.weighted_residual_norm == pytest.approx(0.0)


def test_boundary_response_matrix_solver_uses_label_mapped_control_bounds_and_scale():
    result = solve_boundary_response_matrix(
        np.eye(2),
        current=[0.0, 0.0],
        target={"spectral.row": 1.0, "strike.row": 2.0},
        labels=["spectral.row", "strike.row"],
        control_labels=["spectral_mode", "trim_left"],
        bounds={"spectral_mode": (-0.25, 0.25), "trim_left": 3.0},
        regularization=0.1,
        control_scale={"spectral_mode": 2.0, "trim_left": 0.5},
    )

    assert result.success is True
    assert result.controls_by_label["spectral_mode"] <= 0.25 + 1.0e-12
    assert result.controls_by_label["trim_left"] < 2.0
    assert result.active_upper_bounds == ("spectral_mode",)
    assert result.weighted_control_norm > 0.0


def test_boundary_response_matrix_diagnostics_reports_column_correlation():
    diagnostics = boundary_response_matrix_diagnostics(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert diagnostics.rank == 2
    assert diagnostics.condition_number > 0.0
    assert diagnostics.column_correlation.shape == (3, 3)
    assert diagnostics.column_correlation[0, 1] == pytest.approx(1.0)
    assert diagnostics.column_correlation[0, 2] == pytest.approx(0.0)


def test_boundary_observables_stack_and_centered_response_system():
    current = stack_boundary_response_observables(
        [
            boundary_response_observables(["width"], [0.2], weights=[5.0], prefix="island"),
            boundary_response_observables(["edge"], [0.4], weights=[2.0], prefix="chaos"),
        ]
    )
    plus = [
        boundary_response_observables(current.labels, [0.3, 0.5], weights=current.weights),
        boundary_response_observables(current.labels, [0.1, 0.7], weights=current.weights),
    ]
    minus = [
        boundary_response_observables(current.labels, [0.1, 0.3], weights=current.weights),
        boundary_response_observables(current.labels, [0.3, 0.1], weights=current.weights),
    ]

    system = finite_difference_boundary_response_system(
        current,
        plus,
        minus,
        steps=[0.5, 2.0],
        control_labels=["coil_a", "coil_b"],
    )

    assert system.labels == ("island.width", "chaos.edge")
    assert system.control_labels == ("coil_a", "coil_b")
    np.testing.assert_allclose(system.response_matrix, [[0.2, -0.05], [0.2, 0.15]])
    assert system.diagnostics.rank == 2
    result = system.solve([0.4, 0.7], bounds=(-10.0, 10.0))
    assert result.success is True
    np.testing.assert_allclose(result.predicted, [0.4, 0.7], atol=1.0e-12)


def test_finite_difference_response_system_accepts_asymmetric_steps():
    current = boundary_response_observables(["heat"], [3.0])
    plus = [boundary_response_observables(["heat"], [4.5])]
    minus = [boundary_response_observables(["heat"], [2.0])]

    system = finite_difference_boundary_response_system(
        current,
        plus,
        minus,
        steps=[1.0],
        plus_steps=[0.75],
        minus_steps=[0.5],
        control_labels=["trim"],
    )

    np.testing.assert_allclose(system.response_matrix[:, 0], [2.0])


def test_stack_boundary_linear_response_systems_aligns_control_labels():
    island = finite_difference_boundary_response_system(
        boundary_response_observables(["island.width"], [0.0]),
        [boundary_response_observables(["island.width"], [2.0])],
        [boundary_response_observables(["island.width"], [-2.0])],
        control_labels=["spectral_mode"],
    )
    strike = finite_difference_boundary_response_system(
        boundary_response_observables(["strike.branch"], [1.0]),
        [
            boundary_response_observables(["strike.branch"], [1.5]),
            boundary_response_observables(["strike.branch"], [0.5]),
        ],
        [
            boundary_response_observables(["strike.branch"], [0.5]),
            boundary_response_observables(["strike.branch"], [1.5]),
        ],
        control_labels=["trim_a", "spectral_mode"],
    )

    stacked = stack_boundary_linear_response_systems([island, strike])

    assert stacked.labels == ("island.width", "strike.branch")
    assert stacked.control_labels == ("spectral_mode", "trim_a")
    np.testing.assert_allclose(stacked.response_matrix, [[2.0, 0.0], [-0.5, 0.5]])
    np.testing.assert_allclose(stacked.current, [0.0, 1.0])
    assert stacked.control_index == {"spectral_mode": 0, "trim_a": 1}
    assert stacked.row_index == {"island.width": 0, "strike.branch": 1}


def test_boundary_linear_response_system_rejects_duplicate_control_labels():
    with pytest.raises(ValueError, match="control_labels must be unique"):
        finite_difference_boundary_response_system(
            boundary_response_observables(["island.width"], [0.0]),
            [
                boundary_response_observables(["island.width"], [1.0]),
                boundary_response_observables(["island.width"], [2.0]),
            ],
            [
                boundary_response_observables(["island.width"], [-1.0]),
                boundary_response_observables(["island.width"], [-2.0]),
            ],
            control_labels=["trim", "trim"],
        )


def test_wall_heat_region_observables_accept_non_square_masks():
    heat = np.arange(15, dtype=float).reshape(3, 5)
    left = np.zeros_like(heat, dtype=bool)
    left[:, :2] = True
    outer = np.zeros_like(heat, dtype=bool)
    outer[2, :] = True

    rows = wall_heat_region_observables(
        heat,
        [left, outer],
        ["left", "outer"],
        normalize=True,
        weights=[3.0, 4.0],
    )

    assert rows.labels == ("strike.left", "strike.outer")
    assert rows.weights == pytest.approx([3.0, 4.0])
    total = float(np.sum(heat))
    assert rows.values[0] == pytest.approx(float(np.sum(heat[:, :2])) / total)
    assert rows.values[1] == pytest.approx(float(np.sum(heat[2, :])) / total)


def test_chaotic_layer_region_observables_measure_coverage():
    intervals = [
        SimpleNamespace(inner=0.20, outer=0.35),
        SimpleNamespace(inner=0.42, outer=0.60),
    ]

    rows = chaotic_layer_region_observables(
        intervals,
        [(0.10, 0.40), (0.40, 0.70)],
        ["inner_edge", "outer_edge"],
    )

    assert rows.labels == ("chaos.inner_edge", "chaos.outer_edge")
    assert rows.values[0] == pytest.approx(0.15 / 0.30)
    assert rows.values[1] == pytest.approx(0.18 / 0.30)


def test_resonant_chain_observables_use_largest_chain_per_mode():
    chains = [
        SimpleNamespace(m=7, n=3, radial_label=0.42, b_res=1.0e-4, half_width=0.01, coefficient=1.0 + 0.0j),
        SimpleNamespace(m=7, n=3, radial_label=0.43, b_res=4.0e-4, half_width=0.03, coefficient=1.0j),
    ]

    rows = resonant_chain_observables(
        chains,
        [(7, 3), (8, 3)],
        quantities=["half_width", "coefficient_real", "coefficient_imag", "phase_sin"],
    )

    assert rows.labels == (
        "island.m7.n3.half_width",
        "island.m7.n3.coefficient_real",
        "island.m7.n3.coefficient_imag",
        "island.m7.n3.phase_sin",
        "island.m8.n3.half_width",
        "island.m8.n3.coefficient_real",
        "island.m8.n3.coefficient_imag",
        "island.m8.n3.phase_sin",
    )
    assert rows.values[:4] == pytest.approx([0.03, 0.0, 1.0, 1.0])
    assert rows.values[4:] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def _dpk_payload(matrices, *, term=None, return_period=1.0, alive=None):
    matrices = np.asarray(matrices, dtype=float)
    k = np.arange(1, matrices.shape[0] + 1, dtype=np.int32)
    eig_abs = np.asarray([np.abs(np.linalg.eigvals(mat)) for mat in matrices], dtype=float)
    alive_arr = np.ones(k.size, dtype=np.int32) if alive is None else np.asarray(alive, dtype=np.int32)
    base = (
        k,
        np.zeros(k.size),
        np.zeros(k.size),
        k.astype(float) * float(return_period),
        matrices.reshape(k.size, 4),
        eig_abs,
        alive_arr,
    )
    if term is None:
        return base
    return base + (np.asarray([np.nan, np.nan, np.nan, np.nan]), np.asarray([term], dtype=np.int32))


def test_boundary_dpk_growth_metrics_classify_regular_rotation():
    angles = np.linspace(0.1, 0.5, 5)
    matrices = [
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        for angle in angles
    ]

    metrics = boundary_dpk_growth_metrics(_dpk_payload(matrices), return_period=1.0)
    rows = dpk_growth_observables(metrics)

    assert metrics.classification == "regular"
    assert metrics.eigenvalue_ftle < 1.0e-12
    assert metrics.spectral_regularity < 1.0e-12
    assert metrics.svd_regularity < 1.0e-12
    assert metrics.spectral_recurrence_min < 1.0e-12
    assert metrics.spectral_recurrence_k == 1
    assert metrics.spectral_recurrence_fraction == pytest.approx(1.0)
    assert metrics.svd_at_spectral_recurrence < 1.0e-12
    assert metrics.recurrent_surface_indicator == pytest.approx(1.0)
    assert rows.metadata["classification"] == "regular"
    assert rows.values[rows.labels.index("dpk.regular")] == pytest.approx(1.0)
    assert rows.values[rows.labels.index("dpk.recurrent_surface")] == pytest.approx(1.0)


def test_boundary_dpk_growth_metrics_classify_hyperbolic_growth():
    matrices = [
        [[np.exp(0.55 * k), 0.0], [0.0, np.exp(-0.55 * k)]]
        for k in range(1, 6)
    ]

    metrics = boundary_dpk_growth_metrics(_dpk_payload(matrices), return_period=1.0)

    assert metrics.classification == "strongly_chaotic"
    assert metrics.eigenvalue_ftle == pytest.approx(0.55)
    assert metrics.ftle == pytest.approx(0.55)
    assert metrics.growth_slope == pytest.approx(0.55)
    assert metrics.spectral_regularity > 0.3
    assert metrics.spectral_recurrence_min > 0.3
    assert metrics.spectral_recurrence_fraction == pytest.approx(0.0)
    assert metrics.recurrent_surface_indicator == pytest.approx(0.0)


def test_boundary_dpk_growth_metrics_classify_open_loss_term():
    matrices = np.repeat(np.eye(2)[None, :, :], 3, axis=0)

    metrics = boundary_dpk_growth_metrics(_dpk_payload(matrices, term=1), return_period=1.0)

    assert metrics.classification == "open_loss"
    rows = dpk_growth_observables(metrics)
    assert rows.values[rows.labels.index("dpk.open_loss")] == pytest.approx(1.0)


def test_boundary_dpk_growth_observables_are_finite_when_all_traces_are_dead():
    matrices = np.repeat(np.eye(2)[None, :, :], 3, axis=0)

    metrics = boundary_dpk_growth_metrics(
        _dpk_payload(matrices, term=2, alive=np.zeros(3, dtype=np.int32)),
        return_period=1.0,
    )
    rows = dpk_growth_observables(metrics)

    assert metrics.classification == "unknown"
    assert metrics.n_recorded == 0
    assert np.all(np.isfinite(rows.values))
    assert metrics.spectral_recurrence_min == pytest.approx(1.0)
    assert metrics.svd_at_spectral_recurrence == pytest.approx(1.0)


def test_dpk_growth_observables_select_continuous_rows_for_chaotic_to_regular_control():
    chaotic = _dpk_payload(
        [
            [[np.exp(0.5 * k), 0.0], [0.0, np.exp(-0.5 * k)]]
            for k in range(1, 5)
        ]
    )
    regular = _dpk_payload(np.repeat(np.eye(2)[None, :, :], 4, axis=0))
    quantities = ("spectral_regularity", "svd_regularity")

    chaotic_rows = dpk_growth_observables(chaotic, return_period=1.0, quantities=quantities)
    regular_rows = dpk_growth_observables(regular, return_period=1.0, quantities=quantities)

    assert chaotic_rows.labels == ("dpk.spectral_regularity", "dpk.svd_regularity")
    assert regular_rows.labels == chaotic_rows.labels
    assert chaotic_rows.metadata["classification"] == "strongly_chaotic"
    assert regular_rows.metadata["classification"] == "regular"
    assert np.all(regular_rows.values < chaotic_rows.values)


def test_boundary_dpk_growth_metrics_detect_nonnormal_shear():
    matrices = [
        [[1.0, 5.0 * k], [0.0, 1.0]]
        for k in range(1, 6)
    ]

    metrics = boundary_dpk_growth_metrics(_dpk_payload(matrices), return_period=1.0)

    assert metrics.classification == "strongly_chaotic"
    assert metrics.eigenvalue_ftle < 1.0e-12
    assert metrics.spectral_regularity < 1.0e-12
    assert metrics.spectral_recurrence_min < 1.0e-12
    assert metrics.recurrent_surface_indicator == pytest.approx(0.0)
    assert metrics.svd_at_spectral_recurrence > 1.0
    assert metrics.nonnormality > 1.0
    assert metrics.svd_regularity > 0.3


def test_boundary_dpk_growth_metrics_honors_recurrence_k_window():
    matrices = np.asarray(
        [
            [[np.exp(0.40), 0.0], [0.0, np.exp(-0.40)]],
            np.eye(2),
            [[np.exp(0.25), 0.0], [0.0, np.exp(-0.25)]],
        ],
        dtype=float,
    )

    metrics = boundary_dpk_growth_metrics(
        _dpk_payload(matrices),
        return_period=1.0,
        recurrence_min_k=2,
        recurrence_max_k=2,
    )

    assert metrics.spectral_recurrence_k == 2
    assert metrics.spectral_recurrence_min == pytest.approx(0.0)
    assert metrics.recurrent_surface_indicator == pytest.approx(1.0)
