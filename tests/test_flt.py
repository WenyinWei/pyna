"""Tests for pyna.flt FieldLineTracer."""
import numpy as np
import pytest
from pyna.flt import FieldLineTracer, get_backend


def pure_toroidal_field(rzphi):
    """BR=0, BZ=0, Bphi proportional to 1/R — normalised unit tangent."""
    R = rzphi[0]
    # Bphi = 1/R; |B| = 1/R; unit tangent = (0, 0, 1/(R * 1/R)) = (0, 0, 1)
    # dR/dl = 0, dZ/dl = 0, dphi/dl = Bphi/(R*|B|) = (1/R)/(R*(1/R)) = 1/R
    B_phi = 1.0 / R
    B_mag = B_phi
    return np.array([0.0, 0.0, B_phi / (R * B_mag)])


def test_tracer_stays_near_start_R():
    tracer = FieldLineTracer(pure_toroidal_field, dt=0.1)
    start = np.array([2.0, 0.0, 0.0])
    traj = tracer.trace(start, t_max=20.0)
    assert traj.shape[1] == 3
    R_values = traj[:, 0]
    assert np.allclose(R_values, 2.0, atol=1e-6), \
        f"R varied from 2.0: min={R_values.min():.6f} max={R_values.max():.6f}"


def test_tracer_Z_constant():
    tracer = FieldLineTracer(pure_toroidal_field, dt=0.1)
    start = np.array([2.0, 0.5, 0.0])
    traj = tracer.trace(start, t_max=10.0)
    assert np.allclose(traj[:, 1], 0.5, atol=1e-6)


def test_trace_many_returns_list():
    tracer = FieldLineTracer(pure_toroidal_field, dt=0.1)
    starts = np.array([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    trajs = tracer.trace_many(starts, t_max=5.0)
    assert len(trajs) == 2


def test_get_backend_cpu():
    backend = get_backend('cpu')
    assert backend is not None


def test_get_backend_cuda_raises():
    with pytest.raises(NotImplementedError):
        get_backend('cuda')


def test_get_backend_opencl_raises():
    with pytest.raises(NotImplementedError):
        get_backend('opencl')
