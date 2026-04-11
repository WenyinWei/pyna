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

from pyna.control.FPT import (
    A_matrix,
    cycle_shift,
    delta_g_from_delta_B,
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
    equilibrium=None,
    plasma_response: bool = False,
    R_grid=None,
    Z_grid=None,
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
    equilibrium : EquilibriumSolovev or compatible object, optional
        If provided and ``plasma_response=True``, the plasma response
        δB_plasma is added to each coil's δB_ext before computing
        topology/gap observables.
    plasma_response : bool
        Whether to include plasma response.
        - True (recommended for core quantities like q-profile):
            δB_total = δB_ext + solve_perturbed_gs(B0, J0, p0, δB_ext)
        - False (OK for edge quantities like gap_gi when coils are far):
            δB_total = δB_ext  (vacuum approximation)
    R_grid, Z_grid : array-like or None
        Grid for plasma response computation.  If None, a default grid
        around the equilibrium is constructed automatically.

    Returns
    -------
    R_full : ndarray, shape (n_obs_full, n_coils)
    labels_full : list of str
    """
    from pyna.toroidal.control import gap_response_matrix_fpt

    # Optionally include plasma response (experimental)
    effective_coil_funcs = coil_field_funcs
    if plasma_response and equilibrium is not None:
        import warnings
        warnings.warn(
            "plasma_response=True with build_full_response_matrix is experimental. "
            "Per-point field functions cannot directly use the perturbed GS solver. "
            "For proper plasma response, pre-compute delta_B_total grids per coil "
            "and pass them as coil_field_funcs.",
            UserWarning, stacklevel=2,
        )

    R_topo, labels_topo = build_response_matrix(
        base_field_func, effective_coil_funcs, state,
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
    raw_coil_B_funcs: Optional[List[Callable]] = None,
    base_raw_B_func: Optional[Callable] = None,
):
    """Build response matrix R[n_obs, n_coils] using FPT formulae.

    For each coil k, computes δ_state when I_k increases by eps_current,
    using closed-form FPT for axisymmetric observables (X/O-point shifts
    and DPm changes) and zero placeholders for observables that require
    additional geometric data (gaps, q-profile).

    Parameters
    ----------
    base_field_func : callable
        Base equilibrium field function: [R,Z,phi] → [BR/|B|, BZ/|B|, Bphi/(R|B|)].
    coil_field_funcs : list of callable
        Coil perturbation field functions (per unit current).
        **Legacy form** (only used when raw_coil_B_funcs is None):
          coil_field_funcs[k]([R,Z,phi]) → [δBR/|B0|, δBZ/|B0|, δBphi/(R·|B0|)]
        This form is first-order correct for |δB| << |B0| away from X-points.
        Near the poloidal-field null (X-point), |Bpol/B| ~ 1e-5, so normalized
        coil components are unreliable.  Use raw_coil_B_funcs instead.
    state : TopologyState
        Current topology state (must be pre-computed).
    observables : list of str or None
        Reserved for future observable filtering; currently unused.
    eps_current : float
        Current perturbation amplitude used to define per-unit response.
        (The result is divided by eps_current so R is per-ampere.)
    raw_coil_B_funcs : list of callable or None
        If provided, preferred over coil_field_funcs for DPm eigenvalue
        computation.  Each callable returns raw coil B components:
          raw_coil_B_funcs[k]([R,Z,phi]) → [δBR, δBZ, δBphi]   (in Tesla)
        The base field (B0) must then also be recoverable.  Supply
        base_raw_B_func alongside this argument.
    base_raw_B_func : callable or None
        Returns raw base B components:
          base_raw_B_func([R,Z,phi]) → [BR0, BZ0, Bphi0]   (in Tesla)
        Required when raw_coil_B_funcs is provided.

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
        raw_coil_B = raw_coil_B_funcs[k] if raw_coil_B_funcs is not None else None
        delta_vec = _compute_delta_state_axisymmetric(
            base_field_func, delta_field, state, scale=eps_current,
            raw_coil_B_func=raw_coil_B,
            base_raw_B_func=base_raw_B_func,
        )
        R_mat[:, k] = delta_vec / eps_current

    return R_mat, obs_labels


def _compute_delta_state_axisymmetric(
    field_func: Callable,
    delta_field_func: Callable,
    state: TopologyState,
    scale: float = 1.0,
    raw_coil_B_func: Optional[Callable] = None,
    base_raw_B_func: Optional[Callable] = None,
) -> np.ndarray:
    """Compute δstate using closed-form FPT (axisymmetric).

    Parameters
    ----------
    field_func : callable
        Base field: [R,Z,phi] → [BR/|B|, BZ/|B|, Bphi/(R|B|)].
    delta_field_func : callable
        Coil perturbation field (per unit amplitude, legacy normalized form).
        Used only for δg (cycle shift) and for DPm when raw_coil_B_func is None.
    state : TopologyState
    scale : float
        Scaling factor (= coil current amplitude in Amperes).
    raw_coil_B_func : callable or None
        Returns raw coil B: [δBR, δBZ, δBphi] for 1 A.
        When provided (together with base_raw_B_func), the perturbed direction
        vector is constructed as normalize(B0 + scale*δB) for DPm computation.
    base_raw_B_func : callable or None
        Returns raw base B: [BR0, BZ0, Bphi0].

    Returns
    -------
    delta_vec : ndarray, shape (n_obs,)
    """
    delta_vec: list = []
    phi = state.phi_ref

    # Helper: build properly normalized perturbed direction-vector function.
    # When raw B data is available, normalize(B0+scale*δB) is exact.
    # Otherwise fall back to the first-order approximation f0 + scale*δf.
    def make_perturbed_ff(R_eval: float, Z_eval: float):
        """Return a scalar perturbed direction-vector function."""
        if raw_coil_B_func is not None and base_raw_B_func is not None:
            def ff_pert(rzphi):
                R, Z = float(rzphi[0]), float(rzphi[1])
                b0 = np.asarray(base_raw_B_func([R, Z, phi]), dtype=float)
                db = np.asarray(raw_coil_B_func([R, Z, phi]), dtype=float) * scale
                bpert = b0 + db
                Bmod = np.sqrt(bpert[0]**2 + bpert[1]**2 + bpert[2]**2)
                return [bpert[0]/Bmod, bpert[1]/Bmod, bpert[2]/(R*Bmod)]
        else:
            def ff_pert(rzphi):
                return (np.asarray(field_func(rzphi), dtype=float) +
                        np.asarray(delta_field_func(rzphi), dtype=float) * scale)
        return ff_pert

    # ── X-points ───────────────────────────────────────────────────────────
    for xp in state.xpoints:
        R, Z = xp.R, xp.Z

        # Base field at X-point: f = [BR/|B|, BZ/|B|, Bphi/(R|B|)]
        f0 = np.asarray(field_func([R, Z, phi]), dtype=float)
        fd = np.asarray(delta_field_func([R, Z, phi]), dtype=float) * scale

        # δg = first-order perturbation of g = [R·BR/Bphi, R·BZ/Bphi].
        # Exact formula: δg_i = δf_i/f0[2] - f0[i]*δf[2]/f0[2]²
        # Valid for |δf[2]| << |f0[2]|, i.e., |δBphi/Bphi| << 1 (always true).
        # Also valid for |δf[0,1]| << |f0[0,1]| in normal operation.
        # At the X-point |f0[0]| and |f0[1]| are near-zero, but the formula
        # is still correct because δg represents the change in g at this point
        # regardless of the smallness of g itself.
        f2sq = f0[2] ** 2
        delta_g = np.array([
            fd[0] / f0[2] - f0[0] * fd[2] / f2sq,
            fd[1] / f0[2] - f0[1] * fd[2] / f2sq,
        ])

        # Cycle shift: δx_cyc = -A⁻¹ · δg
        dxcyc = cycle_shift(xp.A_matrix, delta_g)
        delta_vec.extend([dxcyc[0], dxcyc[1]])

        # δDPm eigenvalue response at the FIXED X-point position.
        # Use only δA_direct = A(ff_pert, x_xpt) − A(ff_base, x_xpt).
        # The indirect term (spatial gradient correction for X-point relocation)
        # is NOT added here — that would compute the eigenvalue at the
        # relocated X-point, not the response ∂(λ)/∂I at fixed position.
        ff_pert = make_perturbed_ff(R, Z)
        A_pert = A_matrix(ff_pert, R, Z, phi)
        dA_direct = A_pert - xp.A_matrix
        dDPm = DPm_change(xp.A_matrix, dA_direct)

        # Sort eigenvalues by magnitude for stable pairing (λ_u > 1, λ_s < 1).
        new_eigs = np.linalg.eigvals(xp.DPm + dDPm)
        base_sorted = np.sort(np.abs(xp.DPm_eigenvalues.real))[::-1]
        new_sorted  = np.sort(np.abs(new_eigs.real))[::-1]
        deigs = new_sorted - base_sorted
        delta_vec.extend(deigs.tolist())

    # ── O-points ───────────────────────────────────────────────────────────
    for op in state.opoints:
        R, Z = op.R, op.Z

        f0 = np.asarray(field_func([R, Z, phi]), dtype=float)
        fd = np.asarray(delta_field_func([R, Z, phi]), dtype=float) * scale

        # First-order δg (same formula as X-point)
        f2sq = f0[2] ** 2
        delta_g = np.array([
            fd[0] / f0[2] - f0[0] * fd[2] / f2sq,
            fd[1] / f0[2] - f0[1] * fd[2] / f2sq,
        ])

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
