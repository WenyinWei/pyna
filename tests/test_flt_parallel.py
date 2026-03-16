"""Tests for FieldLineTracer parallelism and CUDA backend."""
from __future__ import annotations

import sys
import pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyna.flt import FieldLineTracer, get_backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R0, a, B0, q0 = 1.0, 0.3, 1.0, 2.0

def solovev_field(rzphi: np.ndarray) -> np.ndarray:
    R, Z, phi = rzphi
    lam = B0 * a / (q0 * R0)
    dpsi_dR = (R * R - R0 * R0) * R / (R0 * R0 * a * a)
    dpsi_dZ = 2.0 * Z / (a * a)
    BR   = -lam / R * dpsi_dZ
    BZ   =  lam / R * dpsi_dR
    Bphi = B0 * R0 / R
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
    return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])


RZLIMIT = (R0 - 1.5*a, R0 + 1.5*a, -1.5*a, 1.5*a)
DT      = 0.1
T_MAX   = 5.0
N       = 10

thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
STARTS = np.column_stack([
    R0 + 0.05 * a * np.cos(thetas),
    0.05 * a * np.sin(thetas),
    np.zeros(N),
])


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------

class TestFieldLineTracer:

    def test_trace_single_shape(self):
        tracer = FieldLineTracer(solovev_field, dt=DT, RZlimit=RZLIMIT)
        traj = tracer.trace(STARTS[0], T_MAX)
        assert traj.ndim == 2
        assert traj.shape[1] == 3
        assert traj.shape[0] >= 2

    def test_trace_many_thread_pool(self):
        """ThreadPoolExecutor path: trace 10 lines, verify shape."""
        tracer = FieldLineTracer(solovev_field, dt=DT, RZlimit=RZLIMIT)
        trajs = tracer.trace_many(STARTS, T_MAX, n_workers=4)
        assert len(trajs) == N
        for t in trajs:
            assert t.ndim == 2
            assert t.shape[1] == 3

    def test_determinism_serial_vs_parallel(self):
        """Serial (n_workers=1) and parallel (n_workers=4) must give identical results."""
        tracer = FieldLineTracer(solovev_field, dt=DT, RZlimit=RZLIMIT)
        serial   = tracer.trace_many(STARTS, T_MAX, n_workers=1)
        parallel = tracer.trace_many(STARTS, T_MAX, n_workers=4)
        assert len(serial) == len(parallel)
        for s, p in zip(serial, parallel):
            np.testing.assert_array_equal(
                s, p,
                err_msg="Serial and parallel trajectories differ — non-determinism detected",
            )

    def test_rzlimit_enforced(self):
        """Trajectory should stop at domain boundary."""
        tight_limit = (R0 - 0.01, R0 + 0.01, -0.01, 0.01)
        tracer = FieldLineTracer(solovev_field, dt=DT, RZlimit=tight_limit)
        traj = tracer.trace(STARTS[0], T_MAX * 10)
        n_steps_unconstrained = int(T_MAX * 10 / DT) + 1
        assert traj.shape[0] < n_steps_unconstrained, "RZlimit was not enforced"

    def test_get_backend_cpu(self):
        backend = get_backend('cpu', field_func=solovev_field, dt=DT, RZlimit=RZLIMIT)
        assert isinstance(backend, FieldLineTracer)
        trajs = backend.trace_many(STARTS, T_MAX)
        assert len(trajs) == N

    def test_get_backend_opencl_raises(self):
        with pytest.raises(NotImplementedError, match="OpenCL"):
            get_backend('opencl')

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend('foobar')


# ---------------------------------------------------------------------------
# CUDA tests (skipped if CuPy unavailable)
# ---------------------------------------------------------------------------

try:
    import cupy  # noqa: F401
    _CUDA_OK = True
except ImportError:
    _CUDA_OK = False


@pytest.mark.skipif(not _CUDA_OK, reason="CuPy not available")
class TestFieldLineTracerCUDA:

    def test_cuda_trace_many_shape(self):
        tracer = get_backend('cuda', R0=R0, a=a, B0=B0, q0=q0, dt=DT, RZlimit=RZLIMIT)
        trajs = tracer.trace_many(STARTS, T_MAX)
        assert len(trajs) == N
        for t in trajs:
            assert t.ndim == 2
            assert t.shape[1] == 3

    def test_cuda_trace_many_start_preserved(self):
        """First point of each trajectory should equal the starting point."""
        tracer = get_backend('cuda', R0=R0, a=a, B0=B0, q0=q0, dt=DT, RZlimit=RZLIMIT)
        trajs = tracer.trace_many(STARTS, T_MAX)
        for i, traj in enumerate(trajs):
            np.testing.assert_allclose(
                traj[0], STARTS[i], atol=1e-12,
                err_msg=f"Starting point mismatch for trajectory {i}",
            )

    def test_cuda_helical_perturbation(self):
        """Non-zero epsilon_h should produce different trajectories."""
        kw = dict(R0=R0, a=a, B0=B0, q0=q0, dt=DT, RZlimit=RZLIMIT)
        t_clean = get_backend('cuda', epsilon_h=0.0, **kw).trace_many(STARTS, T_MAX)
        t_pert  = get_backend('cuda', epsilon_h=0.05, m_h=2.0, n_h=1.0, **kw).trace_many(STARTS, T_MAX)
        # At least one trajectory should differ
        any_diff = any(
            not np.allclose(c, p, atol=1e-6)
            for c, p in zip(t_clean, t_pert)
            if c.shape == p.shape
        )
        assert any_diff, "Helical perturbation had no effect on trajectories"
