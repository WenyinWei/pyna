import numpy as np
import pytest

from pyna.dynamics import (
    trace_cartesian_streamlines,
    trace_cartesian_trajectory,
    vector_field_cylind_cartesian_rhs,
)
from pyna.fields import VectorFieldCartesian, VectorFieldCylind


def test_trace_cartesian_trajectory_accepts_generic_callable():
    sol = trace_cartesian_trajectory(
        lambda x, s: np.array([1.0, -2.0, 0.5]),
        [0.0, 0.0, 0.0],
        s_span=(0.0, 2.0),
        n_steps=8,
    )

    assert sol.coordinate_names == ("x", "y", "z")
    assert sol.metadata["trace_backend"] == "pyna.dynamics.fixed_step_rk4_cartesian"
    np.testing.assert_allclose(sol.final, [2.0, -4.0, 1.0], atol=1.0e-12)


def test_vector_field_cartesian_traces_direct_cartesian_components():
    X = np.linspace(-0.2, 1.2, 5)
    Y = np.linspace(-0.2, 1.2, 5)
    Z = np.linspace(-0.2, 0.2, 3)
    shape = (X.size, Y.size, Z.size)
    field = VectorFieldCartesian(
        X,
        Y,
        Z,
        VX=np.ones(shape),
        VY=np.zeros(shape),
        VZ=np.zeros(shape),
        name="constant Cartesian field",
    )

    sol = trace_cartesian_trajectory(field, [0.0, 0.0, 0.0], s_span=(0.0, 0.75), n_steps=12)

    np.testing.assert_allclose(sol.final, [0.75, 0.0, 0.0], atol=1.0e-12)
    assert sol.metadata["system_type"] == "VectorFieldCartesian"


def test_vector_field_cylind_traces_toroidal_streamline_in_cartesian_space():
    R = np.linspace(0.6, 1.4, 9)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    shape = (R.size, Z.size, Phi.size)
    field = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=Phi,
        BR=np.zeros(shape),
        BZ=np.zeros(shape),
        BPhi=np.ones(shape),
    )

    sol = trace_cartesian_trajectory(
        field,
        [1.0, 0.0, 0.0],
        s_span=(0.0, 0.5 * np.pi),
        n_steps=256,
        normalize=True,
    )

    np.testing.assert_allclose(np.linalg.norm(sol.y[:, :2], axis=1), 1.0, atol=2.0e-6)
    np.testing.assert_allclose(sol.y[:, 2], 0.0, atol=2.0e-12)
    np.testing.assert_allclose(sol.final, [0.0, 1.0, 0.0], atol=2.0e-5)
    assert sol.metadata["system_type"] == "VectorFieldCylind"


def test_vector_field_cylind_cartesian_rhs_supports_batched_points():
    R = np.linspace(0.6, 1.4, 9)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    shape = (R.size, Z.size, Phi.size)
    field = VectorFieldCylind(R=R, Z=Z, Phi=Phi, BR=np.zeros(shape), BZ=np.zeros(shape), BPhi=np.ones(shape))
    rhs = vector_field_cylind_cartesian_rhs(field, normalize=True)

    values = rhs(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.0)

    np.testing.assert_allclose(values[0], [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(values[1], [-1.0, 0.0, 0.0], atol=1.0e-12)


def test_trace_cartesian_trajectory_stops_below_min_speed():
    field = VectorFieldCartesian(
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        VX=np.zeros((2, 2, 2)),
        VY=np.zeros((2, 2, 2)),
        VZ=np.zeros((2, 2, 2)),
    )

    sol = trace_cartesian_trajectory(field, [0.0, 0.0, 0.0], s_span=(0.0, 1.0), n_steps=4, min_speed=1.0e-9)

    assert sol.n_samples == 1
    assert sol.metadata["terminated_reason"] == "invalid_rhs_or_state"


def test_trace_cartesian_streamlines_returns_bidirectional_curves():
    curves = trace_cartesian_streamlines(
        lambda x: np.array([1.0, 0.0, 0.0]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        length=0.5,
        n_steps=4,
    )

    assert len(curves) == 2
    np.testing.assert_allclose(curves[0].initial, [-0.5, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(curves[0].final, [0.5, 0.0, 0.0], atol=1.0e-12)


def test_vector_field_cartesian_rejects_bad_grid_shape():
    with pytest.raises(ValueError, match="shape"):
        VectorFieldCartesian([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], np.zeros((2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))
