"""Magnetic island chain control via external coil perturbations.

Algorithms:
1. island_suppression_current: find coil currents that cancel ψ_mn at target surface
2. phase_control_current: find currents to rotate island phase by desired angle
3. multi_mode_control: optimize currents to suppress target while monitoring side effects

The "press-down-gourd" (按下葫芦起了瓢) problem:
When suppressing mode (m1,n1), mode (m2,n2) may be amplified.
Multi-mode control handles this by solving a constrained optimization.

Physics background
------------------
The external coil system produces an additional perturbation δb_mn. The total
resonant driving term at the q=m/n surface becomes:

    b_mn_total = b_mn_natural + δb_mn(I_coil)

Suppression: find I_coil such that |b_mn_total| → 0.
Phase control: find I_coil such that arg(b_mn_total) = desired_phase.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core computation: resonant Fourier amplitude on a flux surface
# ---------------------------------------------------------------------------

def compute_resonant_amplitude(
    field_func_perturbation,
    S_res: float,
    m: int,
    n: int,
    equilibrium,
    n_theta: int = 64,
    n_phi: int = 64,
) -> complex:
    """Compute the (m,n) Fourier component of a perturbation field at a resonant surface.

    Integrates the normal (radial) component of the perturbation field along
    the q=m/n flux surface and extracts the (m,n) Fourier coefficient.

    The resonant amplitude is:

        b̃_mn = (1 / (2π)²) ∫₀²π ∫₀²π B_r(θ,φ) exp(-i(mθ - nφ)) dθ dφ

    evaluated on the flux surface at r = r_res = sqrt(S_res) * r0.

    Parameters
    ----------
    field_func_perturbation : callable
        Function f(R, Z, phi) → (BR_pert, BZ_pert, BPhi_pert) giving the
        perturbation field at a point.  For a CoilSet, use a wrapper that
        calls Biot_Savart_field.
    S_res : float
        Normalised flux coordinate of the resonant surface (ψ_norm ∈ [0,1]).
    m, n : int
        Poloidal and toroidal mode numbers.
    equilibrium : SimpleStellarator
        The equilibrium object (provides R0, r0).
    n_theta, n_phi : int
        Number of integration points in each angle.

    Returns
    -------
    complex
        The complex amplitude b̃_mn = |b̃_mn| · exp(i·phase_mn).
    """
    R0 = equilibrium.R0
    r0 = equilibrium.r0
    r_res = np.sqrt(max(S_res, 1e-6)) * r0

    theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    total = 0.0 + 0.0j
    dth = 2 * np.pi / n_theta
    dph = 2 * np.pi / n_phi

    for phi in phi_arr:
        for theta in theta_arr:
            R = R0 + r_res * np.cos(theta)
            Z = r_res * np.sin(theta)
            try:
                br, bz, bp = field_func_perturbation(R, Z, phi)
            except Exception:
                br = bz = bp = 0.0
            # Radial (normal to flux surface) component
            b_rad = br * np.cos(theta) + bz * np.sin(theta)
            total += b_rad * np.exp(-1j * (m * theta - n * phi)) * dth * dph

    return total / (2 * np.pi) ** 2


def _make_coil_field_func(coil_set):
    """Wrap a CoilSet into a (R, Z, phi) -> (BR, BZ, BPhi) scalar callable."""
    from pyna.toroidal.coils.coil_system import Biot_Savart_field

    def field_func(R, Z, phi):
        br_tot = 0.0
        bz_tot = 0.0
        bp_tot = 0.0
        for pts, current in coil_set.coils:
            R_arr = np.array([[R]])
            Z_arr = np.array([[Z]])
            phi_arr = np.array([[phi]])
            br, bz, bp = Biot_Savart_field(pts, current, R_arr, Z_arr, phi_arr)
            br_tot += float(br[0, 0])
            bz_tot += float(bz[0, 0])
            bp_tot += float(bp[0, 0])
        return br_tot, bz_tot, bp_tot

    return field_func


def _natural_field_func(stellarator):
    """Wrap stellarator.field_func into (R, Z, phi) -> (BR, BZ, BPhi)."""
    def field_func(R, Z, phi):
        rzphi = np.array([R, Z, phi])
        unit_vec = stellarator.field_func(rzphi)
        theta = np.arctan2(Z, R - stellarator.R0)
        psi = stellarator.psi_ax(R, Z)
        q = float(stellarator.q_of_psi(psi))
        r_minor = np.sqrt((R - stellarator.R0) ** 2 + Z ** 2)
        B_phi = stellarator.B0 * stellarator.R0 / R
        B_pol = B_phi * r_minor / (R * max(abs(q), 1e-3))
        if r_minor > 1e-10:
            BR0 = -B_pol * np.sin(theta)
            BZ0 = B_pol * np.cos(theta)
        else:
            BR0 = BZ0 = 0.0
        delta_BR = (
            stellarator.epsilon_h
            * stellarator.B0
            * np.sqrt(max(psi, 1e-12))
            * np.cos(stellarator.m_h * theta - stellarator.n_h * phi)
        )
        _ = unit_vec
        return BR0 + delta_BR, BZ0, 0.0

    return field_func


def _natural_perturbation_func(stellarator):
    """Extract only the helical perturbation part of the stellarator field."""
    def field_func(R, Z, phi):
        theta = np.arctan2(Z, R - stellarator.R0)
        psi = stellarator.psi_ax(R, Z)
        b_rad = (
            stellarator.epsilon_h
            * stellarator.B0
            * np.sqrt(max(psi, 1e-12))
            * np.cos(stellarator.m_h * theta - stellarator.n_h * phi)
        )
        return b_rad * np.cos(theta), b_rad * np.sin(theta), 0.0

    return field_func


# ---------------------------------------------------------------------------
# Island suppression
# ---------------------------------------------------------------------------

def island_suppression_current(
    stellarator,
    control_coils,
    target_m: int,
    target_n: int,
    monitor_modes: Optional[List[Tuple[int, int]]] = None,
    I_max: float = 1e4,
    n_theta: int = 32,
    n_phi: int = 32,
) -> Tuple[np.ndarray, dict]:
    """Find external coil currents to suppress the (target_m, target_n) island."""
    psi_list = stellarator.resonant_psi(target_m, target_n)
    if not psi_list:
        raise ValueError(f"No resonant surface for q={target_m}/{target_n}")
    S_res = psi_list[0]

    N_coils = len(control_coils)
    nat_func = _natural_perturbation_func(stellarator)
    b_nat = compute_resonant_amplitude(
        nat_func, S_res, target_m, target_n, stellarator, n_theta, n_phi
    )

    R_mat = np.zeros(N_coils, dtype=complex)
    saved_coils = [(pts.copy(), float(I)) for pts, I in control_coils.coils]

    for k in range(N_coils):
        for j in range(N_coils):
            control_coils.coils[j] = (control_coils.coils[j][0], 1.0 if j == k else 0.0)
        coil_func = _make_coil_field_func(control_coils)
        R_mat[k] = compute_resonant_amplitude(
            coil_func, S_res, target_m, target_n, stellarator, n_theta, n_phi
        )

    control_coils.coils = saved_coils

    def objective(I_vec):
        residual = b_nat + R_mat @ I_vec
        return residual.real**2 + residual.imag**2

    def gradient(I_vec):
        residual = b_nat + R_mat @ I_vec
        return 2.0 * (R_mat.real * residual.real + R_mat.imag * residual.imag)

    bounds = Bounds(lb=-I_max, ub=I_max)
    result = minimize(
        objective,
        np.zeros(N_coils),
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    I_opt = result.x
    b_after = b_nat + R_mat @ I_opt

    report = {
        "target_amplitude_before": abs(b_nat),
        "target_amplitude_after": abs(b_after),
        "suppression_ratio": abs(b_after) / (abs(b_nat) + 1e-30),
        "b_nat_complex": b_nat,
        "b_after_complex": b_after,
        "response_vector": R_mat,
        "optimization_success": result.success,
        "optimization_message": result.message,
        "monitor_amplitudes_before": {},
        "monitor_amplitudes_after": {},
    }

    if monitor_modes:
        for (mm, mn) in monitor_modes:
            psi_list_m = stellarator.resonant_psi(mm, mn)
            if not psi_list_m:
                continue
            S_res_m = psi_list_m[0]

            b_nat_m = compute_resonant_amplitude(
                nat_func, S_res_m, mm, mn, stellarator, n_theta, n_phi
            )
            R_mat_m = np.zeros(N_coils, dtype=complex)
            for k in range(N_coils):
                for j in range(N_coils):
                    control_coils.coils[j] = (control_coils.coils[j][0], 1.0 if j == k else 0.0)
                coil_func_m = _make_coil_field_func(control_coils)
                R_mat_m[k] = compute_resonant_amplitude(
                    coil_func_m, S_res_m, mm, mn, stellarator, n_theta, n_phi
                )
            control_coils.coils = [(pts.copy(), float(I)) for pts, I in saved_coils]

            b_after_m = b_nat_m + R_mat_m @ I_opt
            report["monitor_amplitudes_before"][(mm, mn)] = abs(b_nat_m)
            report["monitor_amplitudes_after"][(mm, mn)] = abs(b_after_m)

    control_coils.set_currents(I_opt)
    return I_opt, report


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

def phase_control_current(
    stellarator,
    control_coils,
    target_m: int,
    target_n: int,
    desired_phase_shift: float,
    I_max: float = 1e4,
    n_theta: int = 32,
    n_phi: int = 32,
) -> np.ndarray:
    """Find currents to shift the island chain phase by desired_phase_shift."""
    psi_list = stellarator.resonant_psi(target_m, target_n)
    if not psi_list:
        raise ValueError(f"No resonant surface for q={target_m}/{target_n}")
    S_res = psi_list[0]
    N_coils = len(control_coils)

    nat_func = _natural_perturbation_func(stellarator)
    b_nat = compute_resonant_amplitude(
        nat_func, S_res, target_m, target_n, stellarator, n_theta, n_phi
    )

    saved_coils = [(pts.copy(), float(I)) for pts, I in control_coils.coils]
    R_mat = np.zeros(N_coils, dtype=complex)
    for k in range(N_coils):
        for j in range(N_coils):
            control_coils.coils[j] = (control_coils.coils[j][0], 1.0 if j == k else 0.0)
        coil_func = _make_coil_field_func(control_coils)
        R_mat[k] = compute_resonant_amplitude(
            coil_func, S_res, target_m, target_n, stellarator, n_theta, n_phi
        )
    control_coils.coils = saved_coils

    phase_nat = np.angle(b_nat)
    b_target = abs(b_nat) * np.exp(1j * (phase_nat + desired_phase_shift))

    def objective(I_vec):
        residual = (b_nat + R_mat @ I_vec) - b_target
        return residual.real**2 + residual.imag**2

    def gradient(I_vec):
        residual = (b_nat + R_mat @ I_vec) - b_target
        return 2.0 * (R_mat.real * residual.real + R_mat.imag * residual.imag)

    bounds = Bounds(lb=-I_max, ub=I_max)
    result = minimize(
        objective,
        np.zeros(N_coils),
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    I_opt = result.x
    control_coils.set_currents(I_opt)
    return I_opt


# ---------------------------------------------------------------------------
# Multi-mode control (press-down-gourd problem)
# ---------------------------------------------------------------------------

def multi_mode_control(
    stellarator,
    control_coils,
    target_modes: List[Tuple[int, int]],
    weights: Optional[List[float]] = None,
    I_max: float = 1e4,
    n_theta: int = 32,
    n_phi: int = 32,
) -> Tuple[np.ndarray, dict]:
    """Optimize currents to suppress multiple modes simultaneously."""
    if weights is None:
        weights = [1.0] * len(target_modes)
    weights = np.array(weights, dtype=float)

    N_coils = len(control_coils)
    nat_func = _natural_perturbation_func(stellarator)
    saved_coils = [(pts.copy(), float(I)) for pts, I in control_coils.coils]

    b_nat_all = {}
    R_mat_all = {}

    for (mm, mn) in target_modes:
        psi_list = stellarator.resonant_psi(mm, mn)
        if not psi_list:
            b_nat_all[(mm, mn)] = 0.0 + 0.0j
            R_mat_all[(mm, mn)] = np.zeros(N_coils, dtype=complex)
            continue
        S_res = psi_list[0]
        b_nat_all[(mm, mn)] = compute_resonant_amplitude(
            nat_func, S_res, mm, mn, stellarator, n_theta, n_phi
        )
        R_vec = np.zeros(N_coils, dtype=complex)
        for k in range(N_coils):
            for j in range(N_coils):
                control_coils.coils[j] = (control_coils.coils[j][0], 1.0 if j == k else 0.0)
            coil_func = _make_coil_field_func(control_coils)
            R_vec[k] = compute_resonant_amplitude(
                coil_func, S_res, mm, mn, stellarator, n_theta, n_phi
            )
        control_coils.coils = [(pts.copy(), float(I)) for pts, I in saved_coils]
        R_mat_all[(mm, mn)] = R_vec

    def objective(I_vec):
        total = 0.0
        for i, (mm, mn) in enumerate(target_modes):
            b = b_nat_all[(mm, mn)] + R_mat_all[(mm, mn)] @ I_vec
            total += weights[i] * (b.real**2 + b.imag**2)
        return total

    def gradient(I_vec):
        grad = np.zeros(N_coils)
        for i, (mm, mn) in enumerate(target_modes):
            b = b_nat_all[(mm, mn)] + R_mat_all[(mm, mn)] @ I_vec
            R = R_mat_all[(mm, mn)]
            grad += 2.0 * weights[i] * (R.real * b.real + R.imag * b.imag)
        return grad

    bounds = Bounds(lb=-I_max, ub=I_max)
    result = minimize(
        objective,
        np.zeros(N_coils),
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-15},
    )
    I_opt = result.x

    report = {
        "amplitudes_before": {k: abs(v) for k, v in b_nat_all.items()},
        "amplitudes_after": {},
        "suppression_ratios": {},
        "optimization_success": result.success,
    }
    for mm, mn in target_modes:
        b_after = b_nat_all[(mm, mn)] + R_mat_all[(mm, mn)] @ I_opt
        report["amplitudes_after"][(mm, mn)] = abs(b_after)
        report["suppression_ratios"][(mm, mn)] = abs(b_after) / (abs(b_nat_all[(mm, mn)]) + 1e-30)

    control_coils.set_currents(I_opt)
    return I_opt, report


__all__ = [
    "compute_resonant_amplitude",
    "_make_coil_field_func",
    "_natural_field_func",
    "_natural_perturbation_func",
    "island_suppression_current",
    "phase_control_current",
    "multi_mode_control",
]
