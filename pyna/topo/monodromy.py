"""Monodromy matrix and variational equation analysis along field-line orbits.

The variational equations describe how a small displacement δX evolves:
    dJ/dφ = A(r,z,φ) · J,   J(φ0) = I

where A is the 2×2 Jacobian of the field direction:
    A_ij = ∂(R·B_pol_i / Bφ) / ∂x_j

The monodromy matrix M = J(φ0 + 2πn) gives the linearized n-turn Poincaré map.
Eigenvalues of M: λ₁·λ₂ = 1 (area-preserving),
    |λ| = 1 → elliptic (O-point), λ > 1 → hyperbolic (X-point).

DPm evolution (Lie algebra / commutator equation):
    dDPm/dφ = A·DPm - DPm·A
Used for computing how monodromy changes with perturbation.

Orbit shift under perturbation δB:
    dXcyc/dφ = A·Xcyc + δb_pol
    δb_pol_R = R·δBR/Bφ - R·BR·δBφ/Bφ²
    δb_pol_Z = R·δBZ/Bφ - R·BZ·δBφ/Bφ²

Reference: W7-X Jacobian analysis notebook (Julia) — ported to Python.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Callable, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MonodromyAnalysis:
    """Full monodromy analysis result along a periodic orbit.

    Attributes
    ----------
    phi_arr : ndarray
        Toroidal angle array along orbit.
    trajectory : ndarray, shape (N, 2)
        (R, Z) trajectory.
    J_arr : ndarray, shape (N, 2, 2)
        Jacobian matrix J(φ) at each φ.
    DPm_arr : ndarray, shape (N, 2, 2)
        DPm matrix (commutator evolution).
    Jac : ndarray, shape (2, 2)
        M = J(φ_end).
    """
    phi_arr: np.ndarray
    trajectory: np.ndarray
    J_arr: np.ndarray
    DPm_arr: np.ndarray
    Jac: np.ndarray

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the monodromy matrix."""
        return np.linalg.eigvals(self.Jac)

    @property
    def stability_index(self) -> float:
        """Tr(M)/2 for a 2×2 symplectic map."""
        return float(np.trace(self.Jac) / 2.0)

    @property
    def greene_residue(self) -> float:
        """Greene's residue R = (2 - Tr(M))/4. R<0: hyperbolic, 0<R<1: elliptic."""
        return float((2.0 - np.trace(self.Jac)) / 4.0)

    def J_at(self, phi: float) -> np.ndarray:
        """Interpolate Jacobian matrix J at arbitrary φ."""
        n = len(self.phi_arr)
        J_flat = self.J_arr.reshape(n, 4)
        out = np.zeros(4)
        for k in range(4):
            out[k] = float(interp1d(self.phi_arr, J_flat[:, k], kind='cubic')(phi))
        return out.reshape(2, 2)

    def DPm_at(self, phi: float) -> np.ndarray:
        """Interpolate DPm matrix at arbitrary φ."""
        n = len(self.phi_arr)
        DPm_flat = self.DPm_arr.reshape(n, 4)
        out = np.zeros(4)
        for k in range(4):
            out[k] = float(interp1d(self.phi_arr, DPm_flat[:, k], kind='cubic')(phi))
        return out.reshape(2, 2)


# ---------------------------------------------------------------------------
# Build A matrix function
# ---------------------------------------------------------------------------

def build_A_matrix_func(field_func: Callable, eps: float = 1e-4) -> Callable:
    """Build A(r,z,phi) = Jacobian of (R·BR/Bφ, R·BZ/Bφ) w.r.t. (R, Z).

    Uses forward finite differences on field_func.

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi) → (dR/dl, dZ/dl, dphi/dl)``.
    eps : float
        Finite-difference step size.

    Returns
    -------
    callable
        ``A_func(r, z, phi) → ndarray shape (2, 2)``.
    """
    def _g(rzphi):
        """φ-parameterized field direction g = (R·BR/Bφ, R·BZ/Bφ)."""
        f = np.asarray(field_func(rzphi), dtype=float)
        dphi_dl = f[2]
        if abs(dphi_dl) < 1e-30:
            return np.zeros(2)
        return np.array([f[0] / dphi_dl, f[1] / dphi_dl])

    def A_func(r: float, z: float, phi: float) -> np.ndarray:
        rzphi = np.array([r, z, phi])
        g0 = _g(rzphi)

        rzphi_R = np.array([r + eps, z, phi])
        gR = _g(rzphi_R)

        rzphi_Z = np.array([r, z + eps, phi])
        gZ = _g(rzphi_Z)

        A = np.array([
            [(gR[0] - g0[0]) / eps, (gZ[0] - g0[0]) / eps],
            [(gR[1] - g0[1]) / eps, (gZ[1] - g0[1]) / eps],
        ])
        return A

    return A_func


def build_delta_b_pol_func(
    field_func: Callable,
    delta_field_func: Callable,
) -> Callable:
    """Build δb_pol(r, z, phi) = perturbation forcing in the orbit equation.

    δb_pol_R = R·δBR/Bφ - R·BR·δBφ/Bφ²
    δb_pol_Z = R·δBZ/Bφ - R·BZ·δBφ/Bφ²

    For a field_func returning (dR/dl, dZ/dl, dphi/dl), and similarly for
    delta_field_func, we have:
        BR/|B| = f[0], BZ/|B| = f[1], Bphi/(R|B|) = f[2]
    so Bphi = R * |B| * f[2], BR = |B| * f[0], etc.

    The ratio R·BR/Bφ = f[0] / f[2] (cancels |B|).
    For perturbation δB, same ratios but cross terms:
        R·δBR/Bφ - R·BR·δBφ/Bφ² = δf[0]/f[2] - f[0]*δf[2]/f[2]²

    Parameters
    ----------
    field_func : callable
        Unperturbed field.
    delta_field_func : callable
        Perturbation field (same signature).

    Returns
    -------
    callable
        ``delta_b_pol(r, z, phi) → ndarray shape (2,)``.
    """
    def delta_b_pol(r: float, z: float, phi: float) -> np.ndarray:
        rzphi = np.array([r, z, phi])
        f = np.asarray(field_func(rzphi), dtype=float)
        df = np.asarray(delta_field_func(rzphi), dtype=float)
        dphi_dl = f[2]
        if abs(dphi_dl) < 1e-30:
            return np.zeros(2)
        dbR = df[0] / dphi_dl - f[0] * df[2] / dphi_dl ** 2
        dbZ = df[1] / dphi_dl - f[1] * df[2] / dphi_dl ** 2
        return np.array([dbR, dbZ])

    return delta_b_pol


# ---------------------------------------------------------------------------
# Compute monodromy
# ---------------------------------------------------------------------------

def compute_Jac(
    field_func: Callable,
    orbit,
    n_turns: Optional[int] = None,
    dt_output: float = 0.1,
    rtol: float = 1e-8,
    atol: float = 1e-9,
) -> MonodromyAnalysis:
    """Compute monodromy matrix and full Jacobian evolution along an orbit.

    Integrates simultaneously (φ-parameterized):
    1. The orbit trajectory (R(φ), Z(φ))
    2. The variational equation dJ/dφ = A(r,z,φ)·J
    3. The DPm commutator equation dDPm/dφ = A·DPm - DPm·A

    State vector layout (10 components):
        y[0:2]  = (R, Z)
        y[2:6]  = J flattened (row-major: J00, J01, J10, J11)
        y[6:10] = DPm flattened

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi) → (dR/dl, dZ/dl, dphi/dl)``.
    orbit : PeriodicOrbit
        The periodic orbit to analyze.
    n_turns : int or None
        Number of turns. If None, uses orbit.period_n.
    dt_output : float
        Output spacing in φ.
    rtol, atol : float
        ODE solver tolerances.

    Returns
    -------
    MonodromyAnalysis
    """
    from pyna.topo.cycle import PeriodicOrbit  # avoid circular at import time

    if n_turns is None:
        n_turns = orbit.period_n

    R0, Z0, phi0 = float(orbit.rzphi0[0]), float(orbit.rzphi0[1]), float(orbit.rzphi0[2])
    phi_end = phi0 + n_turns * 2.0 * np.pi

    A_func = build_A_matrix_func(field_func)

    def _g(r, z, phi):
        rzphi = np.array([r, z, phi])
        f = np.asarray(field_func(rzphi), dtype=float)
        dphi_dl = f[2]
        if abs(dphi_dl) < 1e-30:
            return np.zeros(2)
        return np.array([f[0] / dphi_dl, f[1] / dphi_dl])

    def rhs(phi, y):
        r, z = y[0], y[1]
        J = y[2:6].reshape(2, 2)
        DPm = y[6:10].reshape(2, 2)

        # orbit velocity
        drz = _g(r, z, phi)

        A = A_func(r, z, phi)
        dJ = A @ J
        dDPm = A @ DPm - DPm @ A

        return np.concatenate([drz, dJ.flatten(), dDPm.flatten()])

    # Initial condition: orbit start, J=I, DPm=I (initial DPm = J(phi0) = I)
    y0 = np.zeros(10)
    y0[0], y0[1] = R0, Z0
    y0[2:6] = np.eye(2).flatten()   # J(phi0) = I
    y0[6:10] = np.eye(2).flatten()  # DPm(phi0) = M(phi0) = I

    n_out = max(int((phi_end - phi0) / dt_output), 50)
    t_eval = np.linspace(phi0, phi_end, n_out)

    sol = solve_ivp(
        rhs,
        (phi0, phi_end),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=dt_output * 2,
    )

    if not sol.success:
        raise RuntimeError(f"Monodromy integration failed: {sol.message}")

    phi_arr = sol.t
    traj = sol.y[:2].T                   # (N, 2)
    J_arr = sol.y[2:6].T.reshape(-1, 2, 2)   # (N, 2, 2)
    DPm_arr = sol.y[6:10].T.reshape(-1, 2, 2)  # (N, 2, 2)
    Jac = J_arr[-1]

    return MonodromyAnalysis(
        phi_arr=phi_arr,
        trajectory=traj,
        J_arr=J_arr,
        DPm_arr=DPm_arr,
        Jac=Jac,
    )


# ---------------------------------------------------------------------------
# Orbit shift under perturbation
# ---------------------------------------------------------------------------

def orbit_shift_under_perturbation(
    field_func: Callable,
    delta_field_func: Callable,
    orbit,
    monodromy_analysis: MonodromyAnalysis,
) -> np.ndarray:
    """Compute the orbit position shift under a perturbation δB.

    Solves the inhomogeneous variational equation:
        dXcyc/dφ = A·Xcyc + δb_pol(r(φ), z(φ), φ)

    with periodic boundary condition. The initial condition is:
        Xcyc(φ0) = (M - I)^{-1} · (-∫ J·J^{-1}·δb dφ)
    which is found by requiring Xcyc(φ0 + 2πn) = Xcyc(φ0).

    In practice: first integrate with Xcyc(φ0)=0 to find the particular
    solution Xpart(φ_end), then set Xcyc(φ0) = (M - I)^{-1} · (-Xpart(φ_end)).

    Parameters
    ----------
    field_func : callable
        Unperturbed field.
    delta_field_func : callable
        Perturbation field δB, same signature as field_func.
    orbit : PeriodicOrbit
    monodromy_analysis : MonodromyAnalysis

    Returns
    -------
    ndarray, shape (N, 2)
        Orbit displacement (δR(φ), δZ(φ)) along the orbit.
    """
    A_func = build_A_matrix_func(field_func)
    db_pol_func = build_delta_b_pol_func(field_func, delta_field_func)

    phi_arr = monodromy_analysis.phi_arr
    phi0 = phi_arr[0]
    phi_end = phi_arr[-1]
    R0, Z0 = float(orbit.rzphi0[0]), float(orbit.rzphi0[1])

    # Precompute trajectory interpolants
    traj = monodromy_analysis.trajectory
    r_interp = interp1d(phi_arr, traj[:, 0], kind='cubic')
    z_interp = interp1d(phi_arr, traj[:, 1], kind='cubic')

    def rhs_particular(phi, Xcyc):
        r = float(r_interp(phi))
        z = float(z_interp(phi))
        A = A_func(r, z, phi)
        db = db_pol_func(r, z, phi)
        return A @ Xcyc + db

    # Step 1: integrate with Xcyc(phi0) = 0 → particular solution
    sol_part = solve_ivp(
        rhs_particular,
        (phi0, phi_end),
        [0.0, 0.0],
        method="DOP853",
        t_eval=phi_arr,
        rtol=1e-8,
        atol=1e-9,
    )

    M = monodromy_analysis.Jac
    Xpart_end = sol_part.y[:, -1]

    # Step 2: periodic BC: X(phi_end) = X(phi0)
    # M * X0 + Xpart_end = X0  → (M - I) * X0 = -Xpart_end
    try:
        X0 = np.linalg.solve(M - np.eye(2), -Xpart_end)
    except np.linalg.LinAlgError:
        X0 = np.zeros(2)

    # Step 3: integrate again with correct IC
    sol_full = solve_ivp(
        rhs_particular,
        (phi0, phi_end),
        X0,
        method="DOP853",
        t_eval=phi_arr,
        rtol=1e-8,
        atol=1e-9,
    )

    return sol_full.y.T  # (N, 2)


# ---------------------------------------------------------------------------
# Monodromy change under perturbation
# ---------------------------------------------------------------------------

def monodromy_change_under_perturbation(
    orbit,
    monodromy_analysis: MonodromyAnalysis,
    orbit_shift: np.ndarray,
    delta_A_func: Callable,
) -> np.ndarray:
    """Compute how the monodromy matrix changes under perturbation δB.

    δM = ∫_φ0^{φ_end} J(φ_end)·J^{-1}(φ)·δA_eff(φ)·J(φ) dφ

    where δA_eff(φ) = δA(r(φ), z(φ), φ) accounts for the change in the
    A matrix due to the perturbation (both direct δA and shift of orbit).

    The integral is evaluated numerically using the trapezoidal rule.

    Parameters
    ----------
    orbit : PeriodicOrbit
    monodromy_analysis : MonodromyAnalysis
    orbit_shift : ndarray, shape (N, 2)
        Orbit displacement from orbit_shift_under_perturbation.
    delta_A_func : callable
        ``delta_A_func(r, z, phi) → ndarray (2, 2)`` — the change in A
        due to the perturbation δB (computed via build_A_matrix_func on δB).

    Returns
    -------
    ndarray, shape (2, 2)
        δM — the change in the monodromy matrix.
    """
    phi_arr = monodromy_analysis.phi_arr
    J_arr = monodromy_analysis.J_arr
    traj = monodromy_analysis.trajectory
    M = monodromy_analysis.Jac

    integrand = np.zeros((len(phi_arr), 2, 2))

    for i, phi in enumerate(phi_arr):
        r, z = traj[i, 0], traj[i, 1]
        dr, dz = orbit_shift[i, 0], orbit_shift[i, 1]
        J_phi = J_arr[i]

        # J(φ_end) · J^{-1}(φ) = M · J^{-1}(φ)
        try:
            J_inv = np.linalg.inv(J_phi)
        except np.linalg.LinAlgError:
            continue

        dA = delta_A_func(r, z, phi)

        # Contribution to δM integrand
        integrand[i] = M @ J_inv @ dA @ J_phi

    # Trapezoidal integration
    dphi = np.diff(phi_arr)
    dM = np.zeros((2, 2))
    for i in range(len(phi_arr) - 1):
        dM += 0.5 * dphi[i] * (integrand[i] + integrand[i + 1])

    return dM


# ---------------------------------------------------------------------------
# Second-order orbit variation
# ---------------------------------------------------------------------------

def second_order_orbit_variation(
    field_func: Callable,
    delta_field_func: Callable,
    orbit,
    monodromy_analysis: MonodromyAnalysis,
    first_order_shift: np.ndarray,
) -> np.ndarray:
    r"""Compute the second-order orbit position variation δ²X(φ).

    The second-order variational equation is

        d(δ²Xᵢ)/dφ = Σⱼ A_{ij} δ²Xⱼ
                    + Σⱼ,ₖ H_{ijk} δXⱼ δXₖ
                    + Σⱼ δA_{ij} δXⱼ

    where:
        - A = ∂f/∂X is the Jacobian of the unperturbed field direction,
        - H_{ijk} = ∂²f_i/∂X_j∂X_k is the Hessian of f,
        - δA_{ij} = ∂(δf_i)/∂X_j is the Jacobian of the perturbation δf,
        - δX is the first-order orbit shift (from
          :func:`orbit_shift_under_perturbation`).

    The periodic boundary condition is handled in the same way as in
    :func:`orbit_shift_under_perturbation`: integrate first with zero
    initial condition to find the particular solution, then use
    (M − I) δ²X₀ = −δ²X_particular(φ_end).

    Parameters
    ----------
    field_func : callable
        Unperturbed field function.
    delta_field_func : callable
        First-order perturbation field δB (same signature as field_func).
    orbit : PeriodicOrbit
        The unperturbed periodic orbit.
    monodromy_analysis : MonodromyAnalysis
        Monodromy analysis of the unperturbed orbit (from
        :func:`compute_Jac`).
    first_order_shift : ndarray, shape (N, 2)
        First-order orbit position shift δX(φ), as returned by
        :func:`orbit_shift_under_perturbation`.

    Returns
    -------
    ndarray, shape (N, 2)
        Second-order orbit displacement δ²X(φ) = (δ²R(φ), δ²Z(φ)).

    Notes
    -----
    This function computes the *second-order* correction to the orbit
    position under the perturbation, following the expansion

        X = X₀ + ε δX + ε² δ²X / 2 + O(ε³).

    The dependence on ε² makes this term important when the first-order
    shift δX is large (e.g. near a separatrix or for strong perturbations).
    """
    from scipy.interpolate import interp1d as _interp1d

    A_func = build_A_matrix_func(field_func)
    # Build Jacobian of delta_field_func (δA)
    from pyna.topo.variational import _fd_jacobian as _fd_jac

    def delta_A_func(r: float, z: float, phi: float) -> np.ndarray:
        """Finite-difference Jacobian of δf w.r.t. (R, Z)."""
        def df(r_, z_, phi_):
            rzphi = np.array([r_, z_, phi_])
            f = np.asarray(field_func(rzphi), dtype=float)
            df_ = np.asarray(delta_field_func(rzphi), dtype=float)
            dphi_dl = f[2]
            if abs(dphi_dl) < 1e-30:
                return np.zeros(2)
            dbR = df_[0] / dphi_dl - f[0] * df_[2] / dphi_dl ** 2
            dbZ = df_[1] / dphi_dl - f[1] * df_[2] / dphi_dl ** 2
            return np.array([dbR, dbZ])
        return _fd_jac(df, np.array([r, z]), phi, 1e-6)

    phi_arr = monodromy_analysis.phi_arr
    phi0 = phi_arr[0]
    phi_end = phi_arr[-1]

    traj = monodromy_analysis.trajectory
    r_interp = _interp1d(phi_arr, traj[:, 0], kind='cubic')
    z_interp = _interp1d(phi_arr, traj[:, 1], kind='cubic')

    # Interpolate δX(φ) — the first-order shift
    dX_interp_0 = _interp1d(phi_arr, first_order_shift[:, 0], kind='cubic')
    dX_interp_1 = _interp1d(phi_arr, first_order_shift[:, 1], kind='cubic')

    # Compute Hessian of f along the orbit (expensive; evaluated on-the-fly)
    from pyna.topo.variational import _fd_hessian as _fd_hes

    def _hessian_f(r: float, z: float, phi: float) -> np.ndarray:
        def f2(r_, z_, phi_):
            rzphi = np.array([r_, z_, phi_])
            fv = np.asarray(field_func(rzphi), dtype=float)
            dphi_dl = fv[2]
            if abs(dphi_dl) < 1e-30:
                return np.zeros(2)
            return np.array([fv[0] / dphi_dl, fv[1] / dphi_dl])
        return _fd_hes(f2, np.array([r, z]), phi, 1e-5)

    def rhs_second_order(phi: float, d2X: np.ndarray) -> np.ndarray:
        r = float(r_interp(phi))
        z = float(z_interp(phi))
        dX = np.array([float(dX_interp_0(phi)), float(dX_interp_1(phi))])

        A = A_func(r, z, phi)
        H = _hessian_f(r, z, phi)          # shape (2, 2, 2)
        dA = delta_A_func(r, z, phi)        # shape (2, 2)

        # A @ δ²X
        lhs1 = A @ d2X
        # Σⱼ,ₖ H_{ijk} δXⱼ δXₖ   → shape (2,)
        lhs2 = np.einsum('ijk,j,k->i', H, dX, dX)
        # Σⱼ δA_{ij} δXⱼ  → shape (2,)
        lhs3 = dA @ dX

        return lhs1 + lhs2 + lhs3

    # Step 1: particular solution with δ²X(φ0) = 0
    from scipy.integrate import solve_ivp as _solve_ivp
    sol_part = _solve_ivp(
        rhs_second_order, (phi0, phi_end), [0.0, 0.0],
        method="DOP853", t_eval=phi_arr, rtol=1e-8, atol=1e-9,
    )
    d2X_part_end = sol_part.y[:, -1]

    # Step 2: periodic BC:  (M − I) d²X₀ = −d²X_part_end
    M = monodromy_analysis.Jac
    try:
        d2X0 = np.linalg.solve(M - np.eye(2), -d2X_part_end)
    except np.linalg.LinAlgError:
        d2X0 = np.zeros(2)

    # Step 3: full solution with correct IC
    sol_full = _solve_ivp(
        rhs_second_order, (phi0, phi_end), d2X0,
        method="DOP853", t_eval=phi_arr, rtol=1e-8, atol=1e-9,
    )
    return sol_full.y.T  # shape (N, 2)
