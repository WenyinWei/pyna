"""Response matrix ∂(topology_state)/∂(control_inputs).

R_ij = ∂(observable_i) / ∂(control_j)

For axisymmetric tokamaks:
  ∂(x_cyc)/∂(I_coil_k)    = -A⁻¹ · ∂g(x_cyc)/∂(I_coil_k)
  ∂(DPm_eigval)/∂(I_coil_k) = from δDPm formula

For 3D/stellarator: same structure but needs φ-integration.

The response matrix enables the linear control problem:
  δ_state ≈ R · δ_controls
hence:
  min  Σ_i w_i |state_i + R_ij δI_j − target_i|²
  s.t. constraints
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional

from pyna.control.fpt import (
    A_matrix,
    cycle_shift,
    delta_A_total,
    DPm_change,
)
from pyna.control.topology_state import TopologyState


def build_full_response_matrix(
    base_field_func: Callable,
    coil_field_funcs: List[Callable],
    state: TopologyState,
    wall=None,
    field_func_key: str = 'default',
    observables: Optional[List[str]] = None,
    eps_current: float = 1.0,
):
    """Full response matrix combining all observable categories.

    Combines:
    - X/O-point shifts and DPm eigenvalue changes (FPT closed-form)
    - Plasma-wall gap responses (FPT manifold shift, if wall is supplied)
    - q-profile (placeholder zeros, to be filled by pyna-qprofile-response)

    Parameters
    ----------
    base_field_func : callable
        Base equilibrium field function.
    coil_field_funcs : list of callable
        Per-unit-current coil field functions.
    state : TopologyState
        Current topology state with at least one X-point.
    wall : WallGeometry or None
        First wall geometry.  If None, gap rows are zero placeholders.
    field_func_key : str
        Hashable key identifying the equilibrium (for caching manifold growth).
    observables : list of str or None
        Reserved for future filtering.
    eps_current : float
        Current perturbation for per-unit normalisation.

    Returns
    -------
    R_full : ndarray, shape (n_obs_full, n_coils)
    labels_full : list of str
    """
    from pyna.control.gap_response import gap_response_matrix_fpt

    R_topo, labels_topo = build_response_matrix(
        base_field_func, coil_field_funcs, state,
        observables=observables, eps_current=eps_current,
    )

    if wall is not None and len(state.xpoints) > 0:
        R_gap, labels_gap = gap_response_matrix_fpt(
            base_field_func, coil_field_funcs, wall,
            state.xpoints[0], field_func_key,
        )
        R_full = np.vstack([R_topo, R_gap])
        labels_full = labels_topo + [f'gap.{n}' for n in labels_gap]
    else:
        R_full = R_topo
        labels_full = labels_topo

    return R_full, labels_full


def build_response_matrix(
    base_field_func: Callable,
    coil_field_funcs: List[Callable],
    state: TopologyState,
    observables: Optional[List[str]] = None,
    eps_current: float = 1.0,
):
    """Build response matrix R[n_obs, n_coils] using FPT formulae.

    For each coil k, computes δ_state when I_k increases by eps_current,
    using closed-form FPT for axisymmetric observables (X/O-point shifts
    and DPm changes) and zero placeholders for observables that require
    additional geometric data (gaps, q-profile).

    Parameters
    ----------
    base_field_func : callable
        Base equilibrium field function: [R,Z,phi] → [dR/dl, dZ/dl, dphi/dl].
    coil_field_funcs : list of callable
        Coil perturbation field functions (per unit current).
        coil_field_funcs[k]([R,Z,phi]) = δB direction vector for 1 A on coil k.
    state : TopologyState
        Current topology state (must be pre-computed).
    observables : list of str or None
        Reserved for future observable filtering; currently unused.
    eps_current : float
        Current perturbation amplitude used to define per-unit response.
        (The result is divided by eps_current so R is per-ampere.)

    Returns
    -------
    R_mat : ndarray, shape (n_obs, n_coils)
    obs_labels : list of str, length n_obs
    """
    n_coils = len(coil_field_funcs)
    _, obs_labels = state.to_vector()
    n_obs = len(obs_labels)

    R_mat = np.zeros((n_obs, n_coils))

    for k, delta_field in enumerate(coil_field_funcs):
        delta_vec = _compute_delta_state_axisymmetric(
            base_field_func, delta_field, state, scale=eps_current
        )
        R_mat[:, k] = delta_vec / eps_current

    return R_mat, obs_labels


def _compute_delta_state_axisymmetric(
    field_func: Callable,
    delta_field_func: Callable,
    state: TopologyState,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute δstate using closed-form FPT (axisymmetric).

    Parameters
    ----------
    field_func : callable
        Base field.
    delta_field_func : callable
        Perturbation field (per unit amplitude).
    state : TopologyState
    scale : float
        Scaling factor applied to the perturbation field.

    Returns
    -------
    delta_vec : ndarray, shape (n_obs,)
    """
    delta_vec: list = []
    phi = state.phi_ref

    # ── X-points ───────────────────────────────────────────────────────────
    for xp in state.xpoints:
        R, Z = xp.R, xp.Z

        f0 = np.asarray(field_func([R, Z, phi]), dtype=float)
        fd = np.asarray(delta_field_func([R, Z, phi]), dtype=float) * scale

        # g = [R·BR/Bphi, R·BZ/Bphi] = [f[0]/f[2], f[1]/f[2]]
        g0 = np.array([f0[0] / f0[2], f0[1] / f0[2]])
        denom = f0[2] + fd[2]
        g1 = np.array([(f0[0] + fd[0]) / denom, (f0[1] + fd[1]) / denom])
        delta_g = g1 - g0

        # Cycle shift: δx_cyc = -A⁻¹ · δg
        dxcyc = cycle_shift(xp.A_matrix, delta_g)
        delta_vec.extend([dxcyc[0], dxcyc[1]])

        # δDPm and resulting eigenvalue changes
        scaled_delta_field = lambda rzphi, _fd=delta_field_func: \
            np.asarray(_fd(rzphi), dtype=float) * scale
        dA = delta_A_total(
            field_func, scaled_delta_field,
            R, Z, phi, xp.A_matrix, dxcyc,
        )
        dDPm = DPm_change(xp.A_matrix, dA)
        new_eigs = np.linalg.eigvals(xp.DPm + dDPm)
        deigs = np.abs(new_eigs) - np.abs(xp.DPm_eigenvalues)
        delta_vec.extend(deigs.real.tolist())

    # ── O-points ───────────────────────────────────────────────────────────
    for op in state.opoints:
        R, Z = op.R, op.Z

        f0 = np.asarray(field_func([R, Z, phi]), dtype=float)
        fd = np.asarray(delta_field_func([R, Z, phi]), dtype=float) * scale

        g0 = np.array([f0[0] / f0[2], f0[1] / f0[2]])
        denom = f0[2] + fd[2]
        g1 = np.array([(f0[0] + fd[0]) / denom, (f0[1] + fd[1]) / denom])
        delta_g = g1 - g0

        dxcyc = cycle_shift(op.A_matrix, delta_g)
        # iota change: placeholder (requires full DPm eigenvector tracking)
        delta_vec.extend([dxcyc[0], dxcyc[1], 0.0])

    # ── Plasma-wall gaps ────────────────────────────────────────────────────
    for _ in state.gap_gi:
        delta_vec.append(0.0)   # requires wall geometry (not in field_func)

    # ── q-profile ───────────────────────────────────────────────────────────
    if state.q_samples is not None:
        delta_vec.extend([0.0] * len(state.q_samples))

    return np.array(delta_vec)
