import numpy as np
import pytest

from pyna.toroidal.control.boundary_plasma_response import (
    CallableBoundaryPlasmaResponseBackend,
    core_preservation_snapshot,
)
from pyna.toroidal.control.boundary_topology_control import (
    BoundaryTopologyControlProblem,
    format_boundary_topology_control_summary,
    linearize_boundary_topology_control,
    resolve_boundary_topology_control_target,
    solve_boundary_topology_control,
)
from pyna.toroidal.control.boundary_topology_design import boundary_response_observables


def test_resolve_boundary_topology_control_target_locks_core_and_preserve_rows():
    rows = boundary_response_observables(
        ["island.width", "core.axis.dR", "preserve.iota_edge", "heat.unrequested"],
        [0.2, 0.03, 0.78, 0.42],
    )

    target = resolve_boundary_topology_control_target(
        {"island.width": 0.6},
        rows,
        zero_prefixes=("core.",),
        preserve_initial_prefixes=("preserve.",),
    )

    assert target == pytest.approx(
        {
            "island.width": 0.6,
            "core.axis.dR": 0.0,
            "preserve.iota_edge": 0.78,
            "heat.unrequested": 0.42,
        }
    )


def test_linearize_boundary_topology_control_uses_plasma_backend_and_core_rows():
    reference = core_preservation_snapshot(axis=[3.0, 0.0])

    def response_backend(request):
        controls = request.controls
        return {
            "core": core_preservation_snapshot(
                axis=[3.0 + 0.1 * controls[0] - 0.05 * controls[1], 0.2 * controls[1]]
            )
        }

    def edge_rows(_snapshot, request):
        controls = request.controls
        return boundary_response_observables(["edge.response"], [1.0 + controls[0] + 2.0 * controls[1]])

    problem = BoundaryTopologyControlProblem(
        CallableBoundaryPlasmaResponseBackend(response_backend),
        initial_controls=[0.0, 0.0],
        control_labels=["spectral_mode", "strike_trim"],
        target={"edge.response": 1.5},
        observable_builders=[edge_rows],
        core_reference=reference,
        steps=[1.0e-4, 1.0e-4],
    )

    system = linearize_boundary_topology_control(problem)
    index = system.row_index

    np.testing.assert_allclose(system.response_matrix[index["edge.response"]], [1.0, 2.0], atol=1.0e-10)
    np.testing.assert_allclose(system.response_matrix[index["core.axis.dR"]], [0.1, -0.05], atol=1.0e-10)
    np.testing.assert_allclose(system.response_matrix[index["core.axis.dZ"]], [0.0, 0.2], atol=1.0e-10)
    assert system.control_labels == ("spectral_mode", "strike_trim")


def test_solve_boundary_topology_control_runs_complete_core_preserving_workflow():
    reference = core_preservation_snapshot(axis=[3.0, 0.0])
    matrix = np.array([[1.0, 0.5], [0.0, 2.0], [0.2, 0.3]], dtype=float)
    base = np.array([0.2, -0.1, 0.1], dtype=float)

    def response_backend(_request):
        return {"B0": "synthetic_healed_B0", "delta_B": "synthetic_deltaB", "core": reference}

    def design_rows(snapshot, request):
        assert snapshot.has_b0_delta_split is True
        return boundary_response_observables(
            ["island.width", "chaos.layer", "heat.outer"],
            base + matrix @ request.controls,
            weights=[2.0, 1.0, 1.5],
        )

    problem = BoundaryTopologyControlProblem(
        CallableBoundaryPlasmaResponseBackend(response_backend),
        initial_controls=[0.0, 0.0],
        control_labels=["spectral_7_3", "strike_trim"],
        target={"island.width": 0.95, "chaos.layer": 0.5, "heat.outer": 0.31},
        observable_builders=[design_rows],
        core_reference=reference,
        core_weights={"axis": 10.0},
        steps=[0.1, 0.1],
        n_iterations=2,
        bounds=(-2.0, 2.0),
    )

    result = solve_boundary_topology_control(problem)

    assert result.controls_by_label["spectral_7_3"] == pytest.approx(0.6, abs=1.0e-10)
    assert result.controls_by_label["strike_trim"] == pytest.approx(0.3, abs=1.0e-10)
    assert result.resolved_target["core.axis.dR"] == pytest.approx(0.0)
    assert result.resolved_target["core.axis.dZ"] == pytest.approx(0.0)
    assert result.validation.final_weighted_residual_norm < 1.0e-9
    assert result.validation.residual_reduction_fraction > 0.99
    assert result.validation.group_weighted_residual_norms["island"] < 1.0e-9
    assert result.final_system is not None

    text = format_boundary_topology_control_summary(result)
    assert "Boundary topology control report" in text
    assert "final controls:" in text
    assert "linear response diagnostics:" in text
