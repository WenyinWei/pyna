"""Multi-objective magnetic topology controller.

Solves the weighted least-squares problem:

  min  Σ_i w_i · |state_i + R_ij · δI_j − target_i|²  +  λ · ||δI||²
  s.t. I_min_j ≤ δI_j ≤ I_max_j          (coil current limits)

Uses scipy.optimize.minimize with the SLSQP method.

Design principle — "avoid whack-a-mole"
  • All objectives are optimised simultaneously, not sequentially.
  • Weights encode the physicist's priorities.
  • Hard bound constraints enforce safety limits.
  • L2 regularisation on ΔI penalises large coil changes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pyna.control.topology_state import TopologyState


@dataclass
class ControlWeights:
    """Weights for multi-objective topology optimisation.

    Higher weight → higher priority.
    Safety-critical observables (gap_gi) should carry the highest weight.
    """
    gap_gi: float = 100.0               # plasma-wall gap  — safety critical
    xpoint_position: float = 10.0       # X-point (R, Z) position
    DPm_eigenvalue: float = 5.0         # DPm eigenvalue magnitudes (L_c tuning)
    DPm_eigenvector: float = 2.0        # DPm eigenvector direction
    Bpol_lcfs: float = 2.0              # |B_pol| on LCFS
    q_profile: float = 1.0              # q-profile samples
    opoint_position: float = 3.0        # magnetic-axis position
    iota_core: float = 0.5              # rotation transform
    delta_I_regularization: float = 0.1 # penalise large coil excursions


@dataclass
class ControlConstraints:
    """Hard constraints for the optimisation."""
    I_max: Optional[np.ndarray] = None              # max |δI_k|  (per coil)
    I_min: Optional[np.ndarray] = None              # min δI_k    (signed)
    gap_min: Dict[str, float] = field(default_factory=dict)  # min gap [m]
    xpoint_R_bounds: Optional[Tuple[float, float]] = None
    xpoint_Z_bounds: Optional[Tuple[float, float]] = None


class TopologyController:
    """Multi-objective magnetic topology controller.

    Uses the linear response matrix R (∂state/∂controls) to find the
    optimal coil-current changes δI that move the topology toward a
    target state while respecting coil limits.

    For nonlinear problems iterate:
      compute state → build R → solve → apply δI → repeat.

    Example
    -------
    >>> ctrl = TopologyController(n_coils=6)
    >>> weights = ControlWeights(gap_gi=100.0, DPm_eigenvalue=10.0)
    >>> delta_I, result = ctrl.solve(
    ...     current_state, target_state,
    ...     response_matrix, obs_labels,
    ...     weights=weights,
    ... )
    """

    def __init__(self, n_coils: int):
        self.n_coils = n_coils

    # ------------------------------------------------------------------
    def solve(
        self,
        current_state: TopologyState,
        target_state: TopologyState,
        response_matrix: np.ndarray,
        obs_labels: List[str],
        weights: Optional[ControlWeights] = None,
        constraints: Optional[ControlConstraints] = None,
    ) -> Tuple[np.ndarray, object]:
        """Solve multi-objective optimisation for optimal δI.

        Parameters
        ----------
        current_state : TopologyState
        target_state  : TopologyState
        response_matrix : ndarray, shape (n_obs, n_coils)
        obs_labels : list of str
        weights : ControlWeights or None
        constraints : ControlConstraints or None

        Returns
        -------
        delta_I : ndarray, shape (n_coils,)
        result  : scipy OptimizeResult
        """
        if weights is None:
            weights = ControlWeights()
        if constraints is None:
            constraints = ControlConstraints()

        s_vec, _ = current_state.to_vector()
        t_vec, _ = target_state.to_vector()
        R = response_matrix

        w_vec = self._build_weight_vector(obs_labels, weights)
        W = np.diag(w_vec)
        lam = weights.delta_I_regularization

        def objective(dI):
            residual = s_vec + R @ dI - t_vec
            return float(residual @ W @ residual + lam * dI @ dI)

        def jac(dI):
            residual = s_vec + R @ dI - t_vec
            return 2.0 * R.T @ W @ residual + 2.0 * lam * dI

        # Bounds
        n = self.n_coils
        I_max = constraints.I_max if constraints.I_max is not None else np.full(n, 1e6)
        I_min = constraints.I_min if constraints.I_min is not None else -I_max
        bounds = Bounds(I_min, I_max)

        # Warm start: pseudoinverse of R applied to (target - current)
        x0, *_ = np.linalg.lstsq(R, t_vec - s_vec, rcond=None)
        x0 = np.clip(x0, I_min, I_max)

        result = minimize(
            objective, x0, jac=jac, bounds=bounds,
            method='SLSQP',
            options={'ftol': 1e-10, 'maxiter': 500},
        )
        return result.x, result

    # ------------------------------------------------------------------
    def predict_response(
        self,
        current_state: TopologyState,
        delta_I: np.ndarray,
        response_matrix: np.ndarray,
        obs_labels: List[str],
    ) -> Dict[str, float]:
        """Predict how each observable changes under δI.

        Returns
        -------
        dict {label: predicted_change}
        """
        delta_state = response_matrix @ delta_I
        return {label: float(dv) for label, dv in zip(obs_labels, delta_state)}

    # ------------------------------------------------------------------
    def _build_weight_vector(
        self, labels: List[str], weights: ControlWeights
    ) -> np.ndarray:
        w = []
        for label in labels:
            if 'gap.' in label:
                w.append(weights.gap_gi)
            elif 'DPm_eig' in label:
                w.append(weights.DPm_eigenvalue)
            elif '.iota' in label:
                w.append(weights.iota_core)
            elif 'xp' in label and ('.R' in label or '.Z' in label):
                w.append(weights.xpoint_position)
            elif 'op' in label and ('.R' in label or '.Z' in label):
                w.append(weights.opoint_position)
            elif 'q.' in label:
                w.append(weights.q_profile)
            else:
                w.append(1.0)
        return np.array(w)
