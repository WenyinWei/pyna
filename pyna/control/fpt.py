"""Functional Perturbation Theory (FPT) for magnetic field topology.

Theory reference:
  Wei, W. et al. "Functional perturbation theory under axisymmetry:
  Simplified formulae and their uses for tokamaks"

Under strict axisymmetry, all formulae reduce to closed-form expressions
(no φ-integration needed), enabling sub-millisecond real-time computation.

For 3D configurations (stellarators, RMP tokamaks), the full φ-integration
is required — same mathematical structure, higher computational cost.

Key objects
-----------
A_matrix : 2×2 Jacobian of (R·B_pol/B_phi) w.r.t. (R, Z)
DPm      : Jacobian of full-period Poincaré map
             axisymmetric: DPm = exp(2π·A)  [exact, O(1) cost]
             general:      integrate dDPm/dphi = [A, DPm]  [O(N) cost]
δx_cyc   : X/O-cycle shift under perturbation
             axisymmetric: δx_cyc = -A⁻¹ · δ(R·B_pol/B_phi)  [exact]
δDPm     : DPm change under perturbation (affects L_c, detachment)
δX^{u/s} : Stable/unstable manifold shift (LCFS shift)
δχ(θ,r)  : Flux surface deformation
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from typing import Callable, Optional


def A_matrix(field_func: Callable, R: float, Z: float,
             phi: float = 0.0, eps: float = 1e-4) -> np.ndarray:
    """Compute the 2×2 A-matrix = ∂(R·B_pol/B_phi)/∂(R,Z) at (R, Z, phi).

    A_ij = ∂(R·B_pol_i / B_phi) / ∂x_j

    Uses central finite differences.

    Parameters
    ----------
    field_func : callable
        field_func([R, Z, phi]) -> [dR/dl, dZ/dl, dphi/dl]
        where dl is arc length. Components:
          f[0] = BR/|B|, f[1] = BZ/|B|, f[2] = Bphi/(R|B|)
        Hence g(R,Z,phi) = [R·BR/Bphi, R·BZ/Bphi] = [f[0]/f[2], f[1]/f[2]].
    R, Z : float
        Evaluation point.
    phi : float
        Toroidal angle (irrelevant under axisymmetry).
    eps : float
        Finite-difference step size.

    Returns
    -------
    A : ndarray, shape (2,2)
        A[0,0] = ∂(R·BR/Bphi)/∂R,  A[0,1] = ∂(R·BR/Bphi)/∂Z
        A[1,0] = ∂(R·BZ/Bphi)/∂R,  A[1,1] = ∂(R·BZ/Bphi)/∂Z
    """
    def g(r, z):
        f = np.asarray(field_func([r, z, phi]), dtype=float)
        return np.array([f[0] / f[2], f[1] / f[2]])  # [R·BR/Bphi, R·BZ/Bphi]

    gRp = g(R + eps, Z)
    gRm = g(R - eps, Z)
    gZp = g(R, Z + eps)
    gZm = g(R, Z - eps)

    A = np.array([
        [(gRp[0] - gRm[0]) / (2 * eps), (gZp[0] - gZm[0]) / (2 * eps)],
        [(gRp[1] - gRm[1]) / (2 * eps), (gZp[1] - gZm[1]) / (2 * eps)],
    ])
    return A


def DPm_axisymmetric(A: np.ndarray, m_turns: float = 1.0) -> np.ndarray:
    """Compute DPm = exp(2π·m·A) for axisymmetric equilibrium.

    Under axisymmetry, A is constant along any X/O orbit, so
    the full-period Poincaré map Jacobian is exactly exp(2π·A).

    Parameters
    ----------
    A : ndarray, shape (2,2)
        A-matrix at the cycle position.
    m_turns : float
        Number of toroidal turns (usually 1.0 for simple cycle).

    Returns
    -------
    DPm : ndarray, shape (2,2)
        Poincaré map Jacobian.  det(DPm) = 1 (area-preserving).
    """
    return expm(2 * np.pi * m_turns * A)


def cycle_shift(A: np.ndarray, delta_g: np.ndarray) -> np.ndarray:
    """Compute X/O-cycle shift under perturbation δB (axisymmetric).

    δx_cyc = -A⁻¹ · δ(R·B_pol/B_phi)|_{x_cyc}

    Closed-form result from integrating the constant-coefficient
    inhomogeneous ODE for δX_pol and applying the periodic orbit condition.
    Derivation: see Appendix A of Wei et al.

    Parameters
    ----------
    A : ndarray, shape (2,2)
        A-matrix at the unperturbed cycle position.
    delta_g : ndarray, shape (2,)
        Perturbation of g = (R·BR/Bphi, R·BZ/Bphi) at the cycle position.
          delta_g[0] = R·δBR/Bphi - R·BR·δBphi/Bphi²
          delta_g[1] = R·δBZ/Bphi - R·BZ·δBphi/Bphi²

    Returns
    -------
    delta_xcyc : ndarray, shape (2,)
        Cycle position shift (δR, δZ).
    """
    return -np.linalg.solve(A, delta_g)


def delta_g_from_delta_B(
    R: float, Z: float, phi: float,
    BR: float, BZ: float, Bphi: float,
    delta_BR: float, delta_BZ: float, delta_Bphi: float,
) -> np.ndarray:
    """Compute δ(R·B_pol/B_phi) from explicit field perturbation δB.

    δ(R·BR/Bphi) = R·δBR/Bphi - R·BR·δBphi/Bphi²
    δ(R·BZ/Bphi) = R·δBZ/Bphi - R·BZ·δBphi/Bphi²

    Parameters
    ----------
    R, Z, phi : float
        Evaluation point.
    BR, BZ, Bphi : float
        Base magnetic field components.
    delta_BR, delta_BZ, delta_Bphi : float
        Perturbation field components.

    Returns
    -------
    delta_g : ndarray, shape (2,)
    """
    return np.array([
        R * delta_BR / Bphi - R * BR * delta_Bphi / Bphi**2,
        R * delta_BZ / Bphi - R * BZ * delta_Bphi / Bphi**2,
    ])


def DPm_change(A: np.ndarray, delta_A: np.ndarray,
               m_turns: float = 1.0, n_quad: int = 20) -> np.ndarray:
    """Compute δDPm under perturbation (axisymmetric).

    δDPm = ∫₀¹ exp(α·2π·A) · δ(2π·A) · exp((1-α)·2π·A) dα

    Uses Gauss-Legendre quadrature over α ∈ [0,1].

    Parameters
    ----------
    A : ndarray, shape (2,2)
        Unperturbed A-matrix.
    delta_A : ndarray, shape (2,2)
        Total variation of A = δA_direct + δA_indirect.
          δA_direct   = ∂A/∂B · δB  (local field change at x_cyc)
          δA_indirect = (δx_cyc · ∂/∂(R,Z)) A  (chain rule via cycle shift)
    m_turns : float
        Number of toroidal turns.
    n_quad : int
        Number of Gauss-Legendre quadrature points.

    Returns
    -------
    delta_DPm : ndarray, shape (2,2)
    """
    alphas, weights = np.polynomial.legendre.leggauss(n_quad)
    alphas = 0.5 * (alphas + 1)   # map [-1,1] → [0,1]
    weights = 0.5 * weights

    delta_DPm = np.zeros((2, 2))
    twopiA = 2 * np.pi * m_turns * A
    delta_2piA = 2 * np.pi * m_turns * delta_A

    for alpha, w in zip(alphas, weights):
        eL = expm(alpha * twopiA)
        eR = expm((1 - alpha) * twopiA)
        delta_DPm += w * eL @ delta_2piA @ eR

    return delta_DPm


def delta_A_total(
    field_func: Callable,
    delta_field_func: Callable,
    R_cyc: float, Z_cyc: float, phi: float,
    A: np.ndarray,
    delta_xcyc: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute total δA = δA_direct + δA_indirect at cycle position.

    δA_direct   : perturbation to A from δB at the (unperturbed) cycle position
    δA_indirect : chain-rule correction due to cycle shift δx_cyc
                  = δx_cyc[0] * ∂A/∂R + δx_cyc[1] * ∂A/∂Z

    Parameters
    ----------
    field_func : callable
        Base field function.
    delta_field_func : callable
        Perturbation field function (same signature as field_func).
    R_cyc, Z_cyc, phi : float
        Cycle position and toroidal angle.
    A : ndarray, shape (2,2)
        Unperturbed A-matrix at (R_cyc, Z_cyc).
    delta_xcyc : ndarray, shape (2,)
        Cycle shift from cycle_shift().
    eps : float
        Finite-difference step.

    Returns
    -------
    delta_A : ndarray, shape (2,2)
    """
    def combined(rzphi):
        return np.asarray(field_func(rzphi), dtype=float) + \
               np.asarray(delta_field_func(rzphi), dtype=float)

    A_pert = A_matrix(combined, R_cyc, Z_cyc, phi, eps)
    delta_A_direct = A_pert - A

    # Spatial gradient of A (base field only)
    A_dR = (A_matrix(field_func, R_cyc + eps, Z_cyc, phi, eps) -
            A_matrix(field_func, R_cyc - eps, Z_cyc, phi, eps)) / (2 * eps)
    A_dZ = (A_matrix(field_func, R_cyc, Z_cyc + eps, phi, eps) -
            A_matrix(field_func, R_cyc, Z_cyc - eps, phi, eps)) / (2 * eps)
    delta_A_indirect = delta_xcyc[0] * A_dR + delta_xcyc[1] * A_dZ

    return delta_A_direct + delta_A_indirect


def manifold_shift(
    field_func: Callable,
    delta_field_func: Callable,
    manifold_points: np.ndarray,
    delta_xcyc: np.ndarray,
    stable: bool = True,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute shift of stable/unstable manifold under perturbation δB.

    Solves the linear ODE (ζ-parameterized):
      d/dζ (δX^{u/s}) = ±{ δB_pol + ∂B_pol/∂(R,Z) · δX^{u/s} }

    with initial condition δX^{u/s}(ζ=0) = δx_cyc.

    Uses a forward Euler integrator along the discrete manifold arc.

    Parameters
    ----------
    field_func : callable
        Base field function [R,Z,phi] -> [dR/dl, dZ/dl, dphi/dl].
    delta_field_func : callable
        Perturbation field function (same signature).
    manifold_points : ndarray, shape (N, 2)
        (R, Z) points along the manifold (may be equally spaced in ζ or arc).
    delta_xcyc : ndarray, shape (2,)
        Initial condition — the cycle shift.
    stable : bool
        True for stable manifold (sign = −1), False for unstable (sign = +1).
    eps : float
        Finite-difference step for gradient computation.

    Returns
    -------
    delta_manifold : ndarray, shape (N, 2)
        (δR, δZ) shifts along the manifold.
    """
    sign = -1.0 if stable else +1.0
    N = len(manifold_points)
    delta_X = np.zeros((N, 2))
    delta_X[0] = delta_xcyc.copy()

    for i in range(N - 1):
        R, Z = manifold_points[i]
        phi = 0.0  # axisymmetric

        f0 = np.asarray(field_func([R, Z, phi]), dtype=float)
        fd = np.asarray(delta_field_func([R, Z, phi]), dtype=float)

        def Bpol_at(r, z):
            f = np.asarray(field_func([r, z, phi]), dtype=float)
            return np.array([f[0] / f[2], f[1] / f[2]])

        Bpol0 = Bpol_at(R, Z)
        Bpol_mag = np.linalg.norm(Bpol0)

        # δB_pol (approximate: first-order ratio perturbation)
        delta_Bpol = np.array([
            fd[0] / (f0[2] + 1e-30) - f0[0] * fd[2] / (f0[2]**2 + 1e-30),
            fd[1] / (f0[2] + 1e-30) - f0[1] * fd[2] / (f0[2]**2 + 1e-30),
        ])

        # ∂B_pol/∂(R,Z) — shape (2,2), columns = ∂/∂R, ∂/∂Z
        dBpol_dR = (Bpol_at(R + eps, Z) - Bpol_at(R - eps, Z)) / (2 * eps)
        dBpol_dZ = (Bpol_at(R, Z + eps) - Bpol_at(R, Z - eps)) / (2 * eps)
        grad_Bpol = np.column_stack([dBpol_dR, dBpol_dZ])  # (2,2)

        # Arc-length step → ζ-step
        dXpol = manifold_points[i + 1] - manifold_points[i]
        dzeta = np.linalg.norm(dXpol) / (Bpol_mag + 1e-30)

        # Forward Euler
        rhs = sign * (delta_Bpol + grad_Bpol @ delta_X[i])
        delta_X[i + 1] = delta_X[i] + dzeta * rhs

    return delta_X


def flux_surface_deformation(
    poincare_func: Callable,
    dpk_func: Callable,
    delta_pk_func: Callable,
    chi_grid: np.ndarray,
    theta_grid: np.ndarray,
    k_max: int = 8,
    delta_xcyc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute flux surface deformation δχ(θ,r) under perturbation.

    Solves the linearized Poincaré equation:
      δP^k(χ(θ,r)) + DP^k(χ)·δχ(θ,r) = δχ(θ + k·Δθ, r)

    for δχ by representing it in a Fourier basis and solving the
    resulting (over-determined) linear system via least squares.

    Parameters
    ----------
    poincare_func : callable
        P^k(x0) → x_final (k-turn Poincaré map).
    dpk_func : callable
        DP^k(x0) → 2×2 Jacobian.
    delta_pk_func : callable
        δP^k(x0) → 2-vector, endpoint shift of k-turn orbit.
    chi_grid : ndarray, shape (N_theta, 2)
        (R, Z) on the unperturbed flux surface at theta_grid.
    theta_grid : ndarray, shape (N_theta,)
        Poloidal angle values in [0, 2π).
    k_max : int
        Maximum k used to build equations (currently ignored in this
        simplified version; kept for API consistency).
    delta_xcyc : ndarray or None
        Constant anchoring shift. If None, zero.

    Returns
    -------
    delta_chi : ndarray, shape (N_theta, 2)
        (δR, δZ) deformation at each θ grid point.

    Notes
    -----
    The current implementation uses a constant-shift approximation
    (δχ ≈ δx_cyc) as the zeroth-order solution.  A full implementation
    would build the Fourier-coupled linear system and solve it.
    """
    N = len(theta_grid)
    n_modes = min(N // 4, 16)

    # Fourier basis: [1, cos θ, sin θ, cos 2θ, sin 2θ, ...]
    basis = np.zeros((N, 2 * n_modes + 1))
    basis[:, 0] = 1.0
    for j in range(1, n_modes + 1):
        basis[:, 2 * j - 1] = np.cos(j * theta_grid)
        basis[:, 2 * j]     = np.sin(j * theta_grid)

    delta_chi = np.zeros((N, 2))
    if delta_xcyc is not None:
        delta_chi[:] = delta_xcyc[np.newaxis, :]

    return delta_chi
