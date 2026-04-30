"""CUDA-accelerated field line tracing via CuPy.

Uses a custom CUDA kernel for batched Euler integration of field lines
(one GPU thread per field line — all lines traced simultaneously).

The magnetic field is encoded directly in the kernel as a Solov'ev
equilibrium (with optional helical perturbation) for maximum throughput.

Requirements
------------
cupy (``pip install cupy-cuda12x``)

Notes
-----
The kernel uses a simple Euler integrator rather than RK4 because
device-function calls inside RawKernel require CUDA ``__device__`` helpers,
which would significantly increase kernel complexity.  For physics-accurate
work, use smaller *dt* values to compensate.  A proper RK4 version can be
added later via CuPy ``__device__`` helper strings.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# CUDA kernel — batched Euler field-line integration (Solov'ev equilibrium)
# ---------------------------------------------------------------------------

_TRACE_KERNEL_CODE = r"""
extern "C" __global__
void rk4_trace_kernel(
    const double* __restrict__ start_pts,  // (n_lines, 3): R, Z, phi
    double*       __restrict__ traj_out,   // (n_lines, max_steps, 3)
    int*          __restrict__ n_steps_out,// (n_lines,)
    int    n_lines,
    int    max_steps,
    double dt,
    double R_min, double R_max,
    double Z_min, double Z_max,
    // Solov'ev equilibrium parameters
    double R0, double a, double B0, double q0,
    // Optional helical perturbation
    double epsilon_h,
    double m_h,
    double n_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lines) return;

    double R   = start_pts[idx * 3 + 0];
    double Z   = start_pts[idx * 3 + 1];
    double phi = start_pts[idx * 3 + 2];

    // Helper: compute field derivatives at (R, Z, phi)
    // Solov'ev: psi = [(R^2 - R0^2)^2] / (4 R0^2 a^2) + Z^2/a^2
    // lambda = B0*a / (q0*R0)
    // BR   = -lam/R * dpsi/dZ
    // BZ   =  lam/R * dpsi/dR
    // Bphi = B0*R0/R
    #define FIELD(R_, Z_, phi_, BR_, BZ_, Bphi_)                         \
    {                                                                      \
        double lam_     = B0 * a / (q0 * R0);                             \
        double dpsi_dR_ = (R_*R_ - R0*R0) * R_ / (R0*R0 * a*a);          \
        double dpsi_dZ_ = 2.0 * Z_ / (a*a);                               \
        BR_   = -lam_ / R_ * dpsi_dZ_;                                    \
        BZ_   =  lam_ / R_ * dpsi_dR_;                                    \
        Bphi_ = B0 * R0 / R_;                                             \
        if (epsilon_h > 0.0) {                                             \
            double psi_n_  = ((R_*R_ - R0*R0)*(R_*R_ - R0*R0))           \
                             / (4.0*R0*R0*a*a) + Z_*Z_/(a*a);             \
            double theta_  = atan2(Z_, R_ - R0);                          \
            double env_    = psi_n_ * (1.0 - psi_n_);                     \
            double phase_  = m_h * theta_ - n_h * phi_;                   \
            BR_   += epsilon_h * B0 * env_ * cos(phase_);                 \
            BZ_   += epsilon_h * B0 * env_ * sin(phase_);                 \
        }                                                                  \
    }

    // Store initial position
    traj_out[idx * max_steps * 3 + 0] = R;
    traj_out[idx * max_steps * 3 + 1] = Z;
    traj_out[idx * max_steps * 3 + 2] = phi;

    int step;
    for (step = 1; step < max_steps; step++) {

        // ---- RK4 stage 1 ----
        double BR1, BZ1, Bphi1;
        FIELD(R, Z, phi, BR1, BZ1, Bphi1);
        double Bmag1 = sqrt(BR1*BR1 + BZ1*BZ1 + Bphi1*Bphi1) + 1e-30;
        double dR1   = BR1 / Bmag1 * dt;
        double dZ1   = BZ1 / Bmag1 * dt;
        double dphi1 = Bphi1 / (R * Bmag1) * dt;

        // ---- RK4 stage 2 ----
        double R2    = R   + 0.5*dR1;
        double Z2    = Z   + 0.5*dZ1;
        double phi2  = phi + 0.5*dphi1;
        double BR2, BZ2, Bphi2;
        FIELD(R2, Z2, phi2, BR2, BZ2, Bphi2);
        double Bmag2 = sqrt(BR2*BR2 + BZ2*BZ2 + Bphi2*Bphi2) + 1e-30;
        double dR2   = BR2 / Bmag2 * dt;
        double dZ2   = BZ2 / Bmag2 * dt;
        double dphi2 = Bphi2 / (R2 * Bmag2) * dt;

        // ---- RK4 stage 3 ----
        double R3    = R   + 0.5*dR2;
        double Z3    = Z   + 0.5*dZ2;
        double phi3  = phi + 0.5*dphi2;
        double BR3, BZ3, Bphi3;
        FIELD(R3, Z3, phi3, BR3, BZ3, Bphi3);
        double Bmag3 = sqrt(BR3*BR3 + BZ3*BZ3 + Bphi3*Bphi3) + 1e-30;
        double dR3   = BR3 / Bmag3 * dt;
        double dZ3   = BZ3 / Bmag3 * dt;
        double dphi3 = Bphi3 / (R3 * Bmag3) * dt;

        // ---- RK4 stage 4 ----
        double R4    = R   + dR3;
        double Z4    = Z   + dZ3;
        double phi4  = phi + dphi3;
        double BR4, BZ4, Bphi4;
        FIELD(R4, Z4, phi4, BR4, BZ4, Bphi4);
        double Bmag4 = sqrt(BR4*BR4 + BZ4*BZ4 + Bphi4*Bphi4) + 1e-30;
        double dR4   = BR4 / Bmag4 * dt;
        double dZ4   = BZ4 / Bmag4 * dt;
        double dphi4 = Bphi4 / (R4 * Bmag4) * dt;

        // ---- Combine ----
        R   += (dR1   + 2.0*dR2   + 2.0*dR3   + dR4)   / 6.0;
        Z   += (dZ1   + 2.0*dZ2   + 2.0*dZ3   + dZ4)   / 6.0;
        phi += (dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4) / 6.0;

        traj_out[(idx * max_steps + step) * 3 + 0] = R;
        traj_out[(idx * max_steps + step) * 3 + 1] = Z;
        traj_out[(idx * max_steps + step) * 3 + 2] = phi;

        // Domain check
        if (R < R_min || R > R_max || Z < Z_min || Z > Z_max) {
            step++;
            break;
        }
    }

    n_steps_out[idx] = step;
    #undef FIELD
}
"""


# ---------------------------------------------------------------------------
# Python class
# ---------------------------------------------------------------------------

class FieldLineTracerCUDA:
    """GPU-accelerated batched field-line tracer for Solov'ev equilibria.

    All field lines are traced simultaneously on the GPU; each CUDA thread
    handles one field line using a 4th-order Runge-Kutta integrator.

    The magnetic field is hard-coded in the kernel as a Solov'ev equilibrium
    (optionally with a helical perturbation).  For arbitrary numerical fields
    use the CPU :class:`~pyna.flt.FieldLineTracer` instead.

    Parameters
    ----------
    R0, a, B0, q0 : float
        Solov'ev equilibrium parameters.
    epsilon_h : float
        Helical perturbation amplitude (0 = pure axisymmetric tokamak).
    m_h, n_h : float
        Poloidal and toroidal mode numbers of the helical perturbation.
    dt : float
        Integration step size (arc length).
    RZlimit : tuple ``(Rmin, Rmax, Zmin, Zmax)`` or None
        Domain boundary.  Defaults to ``(R0-1.5a, R0+1.5a, -1.5a, 1.5a)``.
    """

    def __init__(
        self,
        R0: float,
        a: float,
        B0: float,
        q0: float,
        epsilon_h: float = 0.0,
        m_h: float = 0.0,
        n_h: float = 0.0,
        dt: float = 0.04,
        RZlimit=None,
    ) -> None:
        if not _CUPY_AVAILABLE:
            raise ImportError(
                "CuPy not installed.  Run:  pip install cupy-cuda12x"
            )
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.q0 = q0
        self.epsilon_h = epsilon_h
        self.m_h = m_h
        self.n_h = n_h
        self.dt = dt
        self.RZlimit = RZlimit or (R0 - 1.5 * a, R0 + 1.5 * a, -1.5 * a, 1.5 * a)
        # Compile kernel (JIT — happens once per process)
        self._kernel = cp.RawKernel(_TRACE_KERNEL_CODE, 'rk4_trace_kernel')

    def trace_many(self, start_pts, t_max: float) -> list:
        """Trace all field lines simultaneously on the GPU.

        Parameters
        ----------
        start_pts : array-like, shape ``(n_lines, 3)``
            Starting ``(R, Z, phi)`` for each field line.
        t_max : float
            Maximum integration arc length.

        Returns
        -------
        list of ndarray
            Each element is an ``(n_steps, 3)`` trajectory array.
        """
        start_pts = np.asarray(start_pts, dtype=np.float64)
        n_lines = len(start_pts)
        max_steps = int(t_max / self.dt) + 1

        # Allocate GPU arrays
        d_starts  = cp.asarray(start_pts)
        d_traj    = cp.zeros((n_lines, max_steps, 3), dtype=np.float64)
        d_nsteps  = cp.zeros(n_lines, dtype=np.int32)

        # Launch
        block_size = 128
        grid_size  = (n_lines + block_size - 1) // block_size
        Rmin, Rmax, Zmin, Zmax = self.RZlimit

        self._kernel(
            (grid_size,),
            (block_size,),
            (
                d_starts, d_traj, d_nsteps,
                n_lines, max_steps,
                self.dt,
                Rmin, Rmax, Zmin, Zmax,
                self.R0, self.a, self.B0, self.q0,
                self.epsilon_h,
                float(self.m_h),
                float(self.n_h),
            ),
        )
        cp.cuda.Stream.null.synchronize()

        traj_np   = cp.asnumpy(d_traj)
        nsteps_np = cp.asnumpy(d_nsteps)
        return [traj_np[i, :nsteps_np[i]] for i in range(n_lines)]
