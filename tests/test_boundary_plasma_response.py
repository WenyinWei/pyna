from types import SimpleNamespace

import numpy as np
import pytest

from pyna.toroidal.control.boundary_plasma_response import (
    BoundaryPlasmaResponseBackend,
    BoundaryPlasmaResponseInput,
    BoundaryPlasmaResponseSnapshot,
    CallableBoundaryPlasmaResponseBackend,
    CorePreservationSnapshot,
    VacuumBoundaryPlasmaResponseBackend,
    boundary_plasma_response_input,
    boundary_plasma_response_snapshot,
    core_preservation_observables,
    core_preservation_snapshot,
    evaluate_boundary_plasma_response,
    finite_difference_plasma_response_system,
    plasma_response_observable_evaluator,
    plasma_response_observables,
)
from pyna.toroidal.control.boundary_topology_design import boundary_response_observables


def _core_surfaces(shift_R=0.0, shift_Z=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    radial = np.asarray([0.25, 0.55])
    R = 3.0 + radial[:, None] * np.cos(theta)[None, :] + shift_R
    Z = 0.72 * radial[:, None] * np.sin(theta)[None, :] + shift_Z
    return radial, R, Z


def test_core_preservation_observables_measure_core_deviation_rows():
    radial, R0, Z0 = _core_surfaces()
    _radial, R1, Z1 = _core_surfaces(shift_R=0.02, shift_Z=-0.01)
    reference = core_preservation_snapshot(
        axis=[3.0, 0.0],
        radial_labels=radial,
        surface_R=R0,
        surface_Z=Z0,
        q_profile=[1.20, 1.45],
        iota_profile=[0.83, 0.69],
        scalars={"volume": 12.0, "minor_radius": 0.55},
    )
    current = core_preservation_snapshot(
        axis=[3.01, -0.02],
        radial_labels=radial,
        surface_R=R1,
        surface_Z=Z1,
        q_profile=[1.22, 1.41],
        iota_profile=[0.82, 0.72],
        scalars={"volume": 12.4, "minor_radius": 0.56},
    )

    rows = core_preservation_observables(
        current,
        reference,
        weights={"axis": 10.0, "surface": 5.0, "q_profile": 3.0, "iota_profile": 2.0, "scalar.volume": 7.0},
    )

    index = {label: idx for idx, label in enumerate(rows.labels)}
    assert rows.metadata["kind"] == "core_preservation"
    assert rows.values[index["core.axis.dR"]] == pytest.approx(0.01)
    assert rows.values[index["core.axis.dZ"]] == pytest.approx(-0.02)
    assert rows.values[index["core.axis.displacement"]] == pytest.approx(np.hypot(0.01, 0.02))
    assert rows.values[index["core.surface.rms_displacement"]] == pytest.approx(np.hypot(0.02, 0.01))
    assert rows.values[index["core.surface.max_displacement"]] == pytest.approx(np.hypot(0.02, 0.01))
    assert rows.values[index["core.q_profile.rms_delta"]] == pytest.approx(np.sqrt(np.mean([0.02**2, 0.04**2])))
    assert rows.values[index["core.q_profile.max_abs_delta"]] == pytest.approx(0.04)
    assert rows.values[index["core.iota_profile.max_abs_delta"]] == pytest.approx(0.03)
    assert rows.values[index["core.scalar.volume.delta"]] == pytest.approx(0.4)
    assert rows.values[index["core.scalar.minor_radius.delta"]] == pytest.approx(0.01)
    assert rows.weights[index["core.axis.dR"]] == pytest.approx(10.0)
    assert rows.weights[index["core.surface.rms_displacement"]] == pytest.approx(5.0)
    assert rows.weights[index["core.q_profile.rms_delta"]] == pytest.approx(3.0)
    assert rows.weights[index["core.scalar.volume.delta"]] == pytest.approx(7.0)


def test_core_preservation_snapshot_accepts_mapping_and_object_inputs():
    mapping = {
        "magnetic_axis": [3.0, 0.0],
        "rho": [0.2, 0.4],
        "q": [1.1, 1.3],
        "metadata": {"source": "synthetic"},
    }
    obj = SimpleNamespace(axis=[3.0, 0.0], iota=[0.9, 0.7], radial_labels=[0.2, 0.4])

    from_mapping = core_preservation_snapshot(mapping)
    from_object = core_preservation_snapshot(obj)

    assert isinstance(from_mapping, CorePreservationSnapshot)
    assert from_mapping.metadata["source"] == "synthetic"
    np.testing.assert_allclose(from_mapping.q_profile, [1.1, 1.3])
    np.testing.assert_allclose(from_object.iota_profile, [0.9, 0.7])


def test_vacuum_boundary_plasma_response_backend_preserves_b0_delta_split():
    core = core_preservation_snapshot(axis=[3.0, 0.0])
    backend = VacuumBoundaryPlasmaResponseBackend(core_reference=core, metadata={"backend": "unit"})
    request = boundary_plasma_response_input(
        [0.2, -0.1],
        control_labels=["mode_5_2", "trim"],
        baseline_equilibrium={"eq": "baseline"},
        baseline_field="B0",
        vacuum_delta_field="deltaB_vac",
        metadata={"total_field": "B_total", "case": "synthetic"},
    )

    response = backend.evaluate(request)

    assert isinstance(request, BoundaryPlasmaResponseInput)
    assert isinstance(backend, BoundaryPlasmaResponseBackend)
    assert request.controls_by_label == pytest.approx({"mode_5_2": 0.2, "trim": -0.1})
    assert isinstance(response, BoundaryPlasmaResponseSnapshot)
    assert response.has_b0_delta_split is True
    assert response.background_field == "B0"
    assert response.delta_field == "deltaB_vac"
    assert response.total_field == "B_total"
    assert response.core is core
    assert response.metadata["response_model"] == "vacuum"
    assert response.metadata["backend"] == "unit"
    assert response.metadata["case"] == "synthetic"


def test_callable_boundary_plasma_response_backend_coerces_mapping_snapshot():
    def evaluator(request):
        assert request.controls_by_label["mode"] == pytest.approx(0.3)
        return {
            "B0": "healed_background",
            "delta_B": "self_consistent_delta",
            "equilibrium": {"iteration": 2},
            "core": {"axis": [3.02, -0.01]},
            "metadata": {"response_model": "linear_surrogate"},
        }

    response = evaluate_boundary_plasma_response(
        CallableBoundaryPlasmaResponseBackend(evaluator),
        [0.3],
        control_labels=["mode"],
    )

    assert response.has_b0_delta_split is True
    assert response.background_field == "healed_background"
    assert response.delta_field == "self_consistent_delta"
    assert response.equilibrium == {"iteration": 2}
    np.testing.assert_allclose(response.core.axis, [3.02, -0.01])
    assert response.metadata["response_model"] == "linear_surrogate"


def test_boundary_plasma_response_validation_rejects_bad_shapes():
    with pytest.raises(ValueError, match="controls length"):
        boundary_plasma_response_input([1.0, 2.0], control_labels=["only_one"])
    with pytest.raises(ValueError, match="unique"):
        boundary_plasma_response_input([1.0, 2.0], control_labels=["a", "a"])
    with pytest.raises(ValueError, match="surface_R and surface_Z"):
        core_preservation_snapshot(surface_R=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="matching shapes"):
        core_preservation_observables(
            core_preservation_snapshot(surface_R=np.zeros((2, 3)), surface_Z=np.zeros((2, 3))),
            core_preservation_snapshot(surface_R=np.zeros((2, 4)), surface_Z=np.zeros((2, 4))),
        )


def test_boundary_plasma_response_snapshot_coercion_from_object():
    obj = SimpleNamespace(B0="background", delta_B="perturbation", metadata={"kind": "object"})

    snapshot = boundary_plasma_response_snapshot(obj)

    assert snapshot.has_b0_delta_split is True
    assert snapshot.background_field == "background"
    assert snapshot.delta_field == "perturbation"
    assert snapshot.metadata["kind"] == "object"


def test_plasma_response_observables_stack_builder_rows_and_core_preservation():
    reference = core_preservation_snapshot(
        axis=[3.0, 0.0],
        radial_labels=[0.2, 0.5],
        q_profile=[1.2, 1.4],
    )

    def response_backend(request):
        controls = request.controls
        core = core_preservation_snapshot(
            axis=[3.0 + 0.1 * controls[0] - 0.05 * controls[1], 0.2 * controls[1]],
            radial_labels=[0.2, 0.5],
            q_profile=np.asarray(reference.q_profile) + np.array([controls[0], 2.0 * controls[1]]),
        )
        return {"B0": "healed_background", "delta_B": "self_consistent_delta", "core": core}

    seen_controls = []

    def edge_observables(snapshot, request):
        assert snapshot.has_b0_delta_split is True
        seen_controls.append(request.controls_by_label)
        return boundary_response_observables(
            ["edge.width"],
            [request.controls[0] + 2.0 * request.controls[1]],
            weights=[3.0],
        )

    rows = plasma_response_observables(
        CallableBoundaryPlasmaResponseBackend(response_backend),
        [0.2, -0.1],
        control_labels=["spectral_mode", "strike_trim"],
        observable_builders=[edge_observables],
        core_reference=reference,
        core_weights={"axis": 5.0, "q_profile": 7.0},
        metadata={"case": "synthetic"},
    )

    index = {label: idx for idx, label in enumerate(rows.labels)}
    assert rows.metadata["kind"] == "plasma_response_observables"
    assert rows.metadata["case"] == "synthetic"
    assert rows.values[index["edge.width"]] == pytest.approx(0.0)
    assert rows.values[index["core.axis.dR"]] == pytest.approx(0.025)
    assert rows.values[index["core.axis.dZ"]] == pytest.approx(-0.02)
    assert rows.values[index["core.q_profile.max_abs_delta"]] == pytest.approx(0.2)
    assert rows.weights[index["edge.width"]] == pytest.approx(3.0)
    assert rows.weights[index["core.axis.dR"]] == pytest.approx(5.0)
    assert rows.weights[index["core.q_profile.rms_delta"]] == pytest.approx(7.0)
    assert len(seen_controls) == 1
    assert seen_controls[0] == pytest.approx({"spectral_mode": 0.2, "strike_trim": -0.1})


def test_finite_difference_plasma_response_system_linearizes_observables():
    reference = core_preservation_snapshot(axis=[3.0, 0.0])

    def response_backend(request):
        controls = request.controls
        core = core_preservation_snapshot(
            axis=[3.0 + 0.1 * controls[0] - 0.05 * controls[1], 0.2 * controls[1]]
        )
        return {"core": core}

    backend = CallableBoundaryPlasmaResponseBackend(response_backend)

    def edge_observables(_snapshot, request):
        controls = request.controls
        return boundary_response_observables(
            ["edge.response"],
            [1.0 + controls[0] + 2.0 * controls[1]],
            weights=[4.0],
        )

    evaluate = plasma_response_observable_evaluator(
        backend,
        control_labels=["spectral_mode", "strike_trim"],
        observable_builders=[edge_observables],
        core_reference=reference,
    )
    current = evaluate([0.0, 0.0])
    system = finite_difference_plasma_response_system(
        backend,
        [0.0, 0.0],
        control_labels=["spectral_mode", "strike_trim"],
        steps=[1.0e-4, 1.0e-4],
        observable_builders=[edge_observables],
        core_reference=reference,
    )

    index = {label: idx for idx, label in enumerate(system.labels)}
    assert system.control_labels == ("spectral_mode", "strike_trim")
    assert system.labels == current.labels
    assert system.current[index["edge.response"]] == pytest.approx(1.0)
    np.testing.assert_allclose(system.response_matrix[index["edge.response"]], [1.0, 2.0], atol=1.0e-10)
    np.testing.assert_allclose(system.response_matrix[index["core.axis.dR"]], [0.1, -0.05], atol=1.0e-10)
    np.testing.assert_allclose(system.response_matrix[index["core.axis.dZ"]], [0.0, 0.2], atol=1.0e-10)
