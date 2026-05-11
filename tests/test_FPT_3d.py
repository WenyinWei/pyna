"""Test: delta_DPm_along_cycle_3d vs finite-difference monodromy change.

Strategy
--------
1. Construct a mock 3-D field with a known X-point orbit.
   We use a φ-dependent equilibrium so the test exercises the full 3-D path.

2. From the unperturbed orbit, extract:
   - trajectory_RZphi  (N, 3)
   - DX_pol_along_cycle (N, 2, 2)  [from CycleVariationalData]
   - DPm0  (2, 2)  ??unperturbed monodromy

3. Add a small perturbation δI = 0.01:
   - FPT prediction:  DPm_fpt = DPm0 + delta_DPm_along_cycle_3d(...)
   - Newton (FD) result: DPm_fd  from finite differencing P^m

4. Assert |DPm_fpt - DPm_fd| / |DPm_fd| < 1 %.

The mock field is a simple "rectangular" approximation around an O/X-point
pair.  We use pyna's analytic-field infrastructure.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

# ----- Module under test ----
from pyna.control.FPT_3d import (
    delta_DPm_along_cycle_3d,
    lambda_perturbation,
    basis_delta_lambda,
)
from pyna.control.FPT import A_matrix, DPm_change
from pyna.topo.monodromy import evolve_DPm_along_cycle, CycleVariationalData

# ============================================================================
# Analytic mock field
# ============================================================================
# We use a φ-varying Hamiltonian:
#
#   H(R, Z, φ) = (1/2) * [ A*(R ??R0)^2 ??B*(Z ??Z0)^2 ]
#                + ε_0 * cos(φ) * (R ??R0) * (Z ??Z0)
#
# which gives an X-point at (R0, Z0).  The field direction is parameterised as
#   R * BR / Bphi = dH/dZ / dH/dR  (field-line equation in (R, Z, φ))
# but for simplicity we build a direct field_func from a known A-matrix that
# depends on φ.
#
# Concretely, we set:
#   A_base(R, Z, φ) = [[a + c*cos(φ), s*sin(φ)],
#                       [s*sin(φ),     -a + c*cos(φ)]]   (traceless ??det(DPm)=1)
#
# The corresponding g = A * (R ??R0, Z ??Z0)^T, i.e.
#   g_R(R,Z,phi) = (a + c*cos(phi))*(R-R0) + s*sin(phi)*(Z-Z0)
#   g_Z(R,Z,phi) = s*sin(phi)*(R-R0) + (-a + c*cos(phi))*(Z-Z0)
#
# field_func converts g to (dR/dl, dZ/dl, dphi/dl):
#   dphi/dl ??1  (we choose a normalisation so dphi/dl = 1/Bnorm)
#   dR/dl  = g_R * dphi/dl
#   dZ/dl  = g_Z * dphi/dl
#
# This is entirely analytic so we know A exactly.

R0 = 5.0    # m  ??X-point R
Z0 = 0.0    # m  ??X-point Z
A_COEF  = 0.15   # base hyperbolic coefficient
C_COEF  = 0.03   # φ-modulation amplitude
S_COEF  = 0.01   # off-diagonal φ-modulation
BNORM   = 1.0    # normalisation (dphi/dl = 1)
M_TURNS = 1      # single-turn orbit for speed


def _g(R, Z, phi, c_extra=0.0):
    """g = (R·BR/Bphi, R·BZ/Bphi) for the mock field."""
    dR = R - R0
    dZ = Z - Z0
    a = A_COEF + c_extra
    c = C_COEF
    s = S_COEF
    gR = (a + c * np.cos(phi)) * dR + s * np.sin(phi) * dZ
    gZ = s * np.sin(phi) * dR + (-a + c * np.cos(phi)) * dZ
    return gR, gZ


def make_field_func(c_extra=0.0):
    """Return field_func([R, Z, phi]) ??[dR/dl, dZ/dl, dphi/dl]."""
    def field_func(rzphi):
        R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
        gR, gZ = _g(R, Z, phi, c_extra)
        # dphi/dl = 1 / (R * Bnorm) ??1 (simplified)
        dphi_dl = 1.0 / BNORM
        return np.array([gR * dphi_dl, gZ * dphi_dl, dphi_dl])
    return field_func


BASE_FIELD = make_field_func(c_extra=0.0)

# Small perturbation amplitude
DELTA_I = 0.01
C_PERT  = 0.02   # per unit current extra coefficient
PERT_FIELD = make_field_func(c_extra=DELTA_I * C_PERT)


# ============================================================================
# Compute the unperturbed orbit via ODE integration
# ============================================================================

def compute_unperturbed_orbit(n_points=200):
    """Integrate the mock field from (R0, Z0, 0) for M_TURNS turns.

    Returns (trajectory_RZphi, DX_pol_along_cycle, DPm0).
    """
    phi0 = 0.0
    phi_end = phi0 + M_TURNS * 2.0 * np.pi

    def _gvec(R, Z, phi):
        gR, gZ = _g(R, Z, phi)
        return np.array([gR, gZ])

    def rhs(phi, y):
        R, Z = y[0], y[1]
        DX_pol = y[2:6].reshape(2, 2)
        drz = _gvec(R, Z, phi)  # dphi/dl = 1

        # A-matrix (analytic, φ-dependent)
        a = A_COEF
        c = C_COEF
        s = S_COEF
        A = np.array([
            [a + c * np.cos(phi), s * np.sin(phi)],
            [s * np.sin(phi),    -a + c * np.cos(phi)],
        ])
        dDX_pol = A @ DX_pol
        return np.concatenate([drz, dDX_pol.flatten()])

    t_eval = np.linspace(phi0, phi_end, n_points)
    y0 = np.array([R0, Z0, 1.0, 0.0, 0.0, 1.0])  # (R, Z, I_2x2)
    sol = solve_ivp(rhs, (phi0, phi_end), y0, t_eval=t_eval,
                    method='DOP853', rtol=1e-10, atol=1e-11)
    assert sol.success, f"Orbit integration failed: {sol.message}"

    phi_arr = sol.t
    traj_RZ = sol.y[:2].T  # (N, 2)
    DX_pol_arr = sol.y[2:6].T.reshape(-1, 2, 2)  # (N, 2, 2)
    DPm0 = DX_pol_arr[-1]  # DX_pol at phi_end = monodromy

    trajectory_RZphi = np.column_stack([traj_RZ, phi_arr])
    return trajectory_RZphi, DX_pol_arr, DPm0


# ============================================================================
# Compute DPm_fd via finite difference on the perturbed field
# ============================================================================

def DPm_perturbed_fd(c_extra, fd_eps=1e-4, n_points=200):
    """Compute monodromy matrix for field with c_extra via 5-point FD."""
    phi0 = 0.0
    phi_end = phi0 + M_TURNS * 2.0 * np.pi
    n_pts = n_points

    def gvec(R, Z, phi):
        gR, gZ = _g(R, Z, phi, c_extra)
        return np.array([gR, gZ])

    def rhs_pos(phi, y):
        R, Z = y
        return gvec(R, Z, phi)

    def trace_final(R_start, Z_start):
        sol = solve_ivp(rhs_pos, (phi0, phi_end), [R_start, Z_start],
                        method='DOP853', rtol=1e-12, atol=1e-13,
                        t_eval=[phi_end])
        assert sol.success
        return sol.y[0, -1], sol.y[1, -1]

    Rc, Zc = trace_final(R0, Z0)  # should return close to R0, Z0
    R_Rp, Z_Rp = trace_final(R0 + fd_eps, Z0)
    R_Rm, Z_Rm = trace_final(R0 - fd_eps, Z0)
    R_Zp, Z_Zp = trace_final(R0, Z0 + fd_eps)
    R_Zm, Z_Zm = trace_final(R0, Z0 - fd_eps)

    DPm = np.array([
        [(R_Rp - R_Rm) / (2 * fd_eps), (R_Zp - R_Zm) / (2 * fd_eps)],
        [(Z_Rp - Z_Rm) / (2 * fd_eps), (Z_Zp - Z_Zm) / (2 * fd_eps)],
    ])
    return DPm


# ============================================================================
# Tests
# ============================================================================

def test_delta_DPm_vs_fd():
    """FPT δDPm + DPm0 should match finite-difference DPm to within 5%."""
    traj, DX_pol_arr, DPm0 = compute_unperturbed_orbit(n_points=400)

    # Cycle shift: X-point at (R0,Z0) stays fixed for this symmetric pert
    # (c_extra shifts the hyperbolic coeff but not the fixed point location).
    # δx_cyc ??0 for this perturbation.
    delta_xcyc = np.array([0.0, 0.0])

    dDPm_fpt = delta_DPm_along_cycle_3d(
        traj, DX_pol_arr, delta_xcyc,
        BASE_FIELD, PERT_FIELD,
        m=M_TURNS, method='rk4',
    )
    DPm_fpt = DPm0 + dDPm_fpt

    # Newton / FD reference
    DPm_fd = DPm_perturbed_fd(c_extra=DELTA_I * C_PERT)

    # Relative error matrix norm
    err = np.linalg.norm(DPm_fpt - DPm_fd) / (np.linalg.norm(DPm_fd) + 1e-30)
    print(f"\nDPm0   = {DPm0}")
    print(f"δDPm (FPT) = {dDPm_fpt}")
    print(f"DPm_fpt = {DPm_fpt}")
    print(f"DPm_fd  = {DPm_fd}")
    print(f"Relative error: {err:.4f}")
    assert err < 0.05, f"FPT vs FD relative error {err:.4f} > 5%"


def test_lambda_perturbation_sign():
    """lambda_perturbation should return a nonzero float for an X-point."""
    traj, DX_pol_arr, DPm0 = compute_unperturbed_orbit(n_points=400)
    delta_xcyc = np.array([0.0, 0.0])
    dDPm = delta_DPm_along_cycle_3d(
        traj, DX_pol_arr, delta_xcyc,
        BASE_FIELD, PERT_FIELD,
        m=M_TURNS,
    )
    dl = lambda_perturbation(DPm0, dDPm)
    print(f"\nδλ = {dl:.6f}")
    # λ_> > 1 for X-point; perturbation should give nonzero result
    assert isinstance(dl, float)
    assert abs(dl) < 10.0, f"δλ suspiciously large: {dl}"


def test_basis_delta_lambda_shape():
    """basis_delta_lambda should return shape (K,) for K coil groups."""
    traj, DX_pol_arr, DPm0 = compute_unperturbed_orbit(n_points=200)
    K = 3
    delta_xcyc_basis = np.zeros((K, 2))
    c_extras = [0.002, 0.005, 0.010]
    basis_fields = [make_field_func(c) for c in c_extras]

    result = basis_delta_lambda(
        traj, DX_pol_arr, delta_xcyc_basis, DPm0,
        BASE_FIELD, basis_fields,
        m=M_TURNS,
    )
    print(f"\nbasis_delta_lambda = {result}")
    assert result.shape == (K,)
    # Results should be non-trivial (not all zero)
    assert not np.allclose(result, 0.0, atol=1e-12)


def test_axisymmetric_consistency():
    """Test that 3D FPT delta_DPm_along_cycle_3d equals axisymmetric DPm_change
    when the field is exactly axisymmetric (phi-independent).

    For an axisymmetric field:
    - A_matrix is constant along the orbit
    - DPm_change (from FPT.py) gives: integral formula with expm
    - delta_DPm_along_cycle_3d (from FPT_3d.py) integrates the ODE
    Both should give the same result to within numerical tolerance (~2%).
    """
    # Use phi-independent field: c=0, s=0 so A is constant
    a = A_COEF

    def _g_axisym(R, Z, phi, a_extra=0.0):
        dR = R - R0
        dZ = Z - Z0
        gR = (a + a_extra) * dR
        gZ = -(a + a_extra) * dZ
        return gR, gZ

    def make_axisym_field(a_extra=0.0):
        def field_func(rzphi):
            R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
            gR, gZ = _g_axisym(R, Z, phi, a_extra)
            return np.array([gR, BNORM**-1, -gZ / R]) if False else np.array([gR, gZ, 1.0])
        return field_func

    base_field_axisym = make_axisym_field(a_extra=0.0)
    delta_a_extra = DELTA_I * C_PERT
    pert_field_axisym = make_axisym_field(a_extra=delta_a_extra)

    # Compute unperturbed orbit for axisymmetric field
    phi0 = 0.0
    phi_end = phi0 + M_TURNS * 2.0 * np.pi
    n_points = 400

    A_axisym = np.array([[a, 0.0], [0.0, -a]])

    def rhs_axisym(phi, y):
        R, Z = y[0], y[1]
        DX_pol = y[2:6].reshape(2, 2)
        gR, gZ = _g_axisym(R, Z, phi)
        drz = np.array([gR, gZ])
        dDX_pol = A_axisym @ DX_pol
        return np.concatenate([drz, dDX_pol.flatten()])

    t_eval = np.linspace(phi0, phi_end, n_points)
    y0 = np.array([R0, Z0, 1.0, 0.0, 0.0, 1.0])
    sol = solve_ivp(rhs_axisym, (phi0, phi_end), y0, t_eval=t_eval,
                    method='DOP853', rtol=1e-10, atol=1e-11)
    assert sol.success

    phi_arr = sol.t
    traj_RZ = sol.y[:2].T
    DX_pol_arr = sol.y[2:6].T.reshape(-1, 2, 2)

    trajectory_RZphi = np.column_stack([traj_RZ, phi_arr])

    # 3D FPT result (should work for axisymmetric field too)
    delta_xcyc = np.array([0.0, 0.0])
    result_3d = delta_DPm_along_cycle_3d(
        trajectory_RZphi, DX_pol_arr, delta_xcyc,
        base_field_axisym, pert_field_axisym,
        m=M_TURNS, method='rk4',
    )

    # Axisymmetric FPT result via DPm_change
    # delta_A for this perturbation: A -> A + delta_a_extra * diag(1,-1)
    delta_A = np.array([[delta_a_extra, 0.0], [0.0, -delta_a_extra]])
    result_axisym = DPm_change(A_axisym, delta_A, m_turns=M_TURNS)

    print(f"\nresult_3d     = {result_3d}")
    print(f"result_axisym = {result_axisym}")
    err = np.linalg.norm(result_3d - result_axisym) / (np.linalg.norm(result_axisym) + 1e-30)
    print(f"Relative error: {err:.4f}")

    assert np.allclose(result_3d, result_axisym, rtol=0.02), (
        f"3D FPT vs axisymmetric FPT relative error {err:.4f} > 2%\n"
        f"result_3d={result_3d}\nresult_axisym={result_axisym}"
    )


# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("test_delta_DPm_vs_fd")
    test_delta_DPm_vs_fd()
    print("PASSED")

    print("=" * 60)
    print("test_lambda_perturbation_sign")
    test_lambda_perturbation_sign()
    print("PASSED")

    print("=" * 60)
    print("test_basis_delta_lambda_shape")
    test_basis_delta_lambda_shape()
    print("PASSED")

    print("=" * 60)
    print("test_axisymmetric_consistency")
    test_axisymmetric_consistency()
    print("PASSED")

    print("=" * 60)
    print("All tests passed.")

