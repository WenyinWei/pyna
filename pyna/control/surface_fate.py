"""Flux surface fate classification under perturbation.

Classifies whether a flux surface (characterised by ψ or ι) will:
  - Remain intact        (KAM torus survives)
  - Deform but stay closed
  - Break into an island chain  (resonance → residue crosses 0 or 1)
  - Become chaotic       (Chirikov overlap)

Criteria used:
  - Greene's residue  R = (2 − Tr(DPm)) / 4
  - Relative DPm perturbation magnitude vs ε_KAM / ε_chaos thresholds
  - Higher-order tensors D²Pm, D³Pm (future extension)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional, Tuple

from pyna.control.topology_state import SurfaceFate


# ──────────────────────────────────────────────────────────────────────────────
# Greene's residue
# ──────────────────────────────────────────────────────────────────────────────

def greene_residue(DPm: np.ndarray) -> float:
    """Greene's residue R = (2 − Tr(DPm)) / 4.

    Interpretation
    --------------
    R < 0      : hyperbolic (X-point-like), field lines diverge
    0 < R < 1  : elliptic  (O-point-like), field lines rotate
    R > 1      : inverse hyperbolic
    R = 0 or 1 : parabolic (marginal / transition point)

    Parameters
    ----------
    DPm : ndarray, shape (2,2)
        Poincaré map Jacobian (one full period).

    Returns
    -------
    float
    """
    return float((2.0 - np.trace(DPm)) / 4.0)


# ──────────────────────────────────────────────────────────────────────────────
# Single-surface classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_surface_fate(
    iota: float,
    delta_iota: float,
    DPm: np.ndarray,
    delta_DPm: np.ndarray,
    epsilon_KAM: float = 0.05,
    epsilon_chaos: float = 0.30,
) -> SurfaceFate:
    """Classify how a flux surface responds to a field perturbation.

    Parameters
    ----------
    iota : float
        Unperturbed rotation transform ι = 1/q.
    delta_iota : float
        Change in rotation transform under perturbation.
    DPm : ndarray, shape (2,2)
        Unperturbed Poincaré map Jacobian.
    delta_DPm : ndarray, shape (2,2)
        Change in DPm under perturbation.
    epsilon_KAM : float
        Relative DPm-change threshold below which the KAM torus survives.
    epsilon_chaos : float
        Relative DPm-change threshold above which chaos is expected.

    Returns
    -------
    SurfaceFate
    """
    R0 = greene_residue(DPm)
    R1 = greene_residue(DPm + delta_DPm)

    rel_pert = np.linalg.norm(delta_DPm) / (np.linalg.norm(DPm) + 1e-30)

    if rel_pert < epsilon_KAM:
        return SurfaceFate.INTACT
    elif rel_pert >= epsilon_chaos:
        return SurfaceFate.CHAOTIC
    else:
        # Residue crosses 0 (hyperbolic→elliptic or vice versa)
        # or crosses 1 (inverse hyperbolic transition)
        crosses_zero = (R0 > 0.0 and R1 < 0.0) or (R0 < 0.0 and R1 > 0.0)
        crosses_one  = (R0 < 1.0 and R1 > 1.0) or (R0 > 1.0 and R1 < 1.0)
        if crosses_zero or crosses_one:
            return SurfaceFate.ISLAND
        return SurfaceFate.DEFORMED


# ──────────────────────────────────────────────────────────────────────────────
# Radial scan
# ──────────────────────────────────────────────────────────────────────────────

def scan_surface_fates(
    field_func: Callable,
    delta_field_func: Callable,
    psi_values: np.ndarray,
    q_values: np.ndarray,
    field_period: int = 1,
    epsilon_KAM: float = 0.05,
    epsilon_chaos: float = 0.30,
) -> Dict[float, Tuple[SurfaceFate, float, float]]:
    """Scan flux surfaces from core to edge and classify each.

    For each (psi, q) pair this function would compute the A-matrix and DPm
    at the corresponding O-point (requiring the actual flux surface geometry).
    The present implementation returns UNKNOWN for all surfaces as a stub —
    a full implementation needs the O-point (R,Z) positions for each ψ.

    Parameters
    ----------
    field_func : callable
        Base magnetic field function.
    delta_field_func : callable
        Perturbation field function.
    psi_values : ndarray
        Normalised poloidal flux values ψ_N ∈ [0, 1].
    q_values : ndarray
        Safety factor q at each ψ.
    field_period : int
        Toroidal field period (1 for tokamak, N_fp for stellarator).
    epsilon_KAM : float
        KAM survival threshold.
    epsilon_chaos : float
        Chaos threshold.

    Returns
    -------
    dict {psi_norm: (SurfaceFate, greene_residue_before, delta_residue)}
    """
    results: Dict[float, Tuple[SurfaceFate, float, float]] = {}
    for psi in psi_values:
        results[float(psi)] = (SurfaceFate.UNKNOWN, 0.0, 0.0)
    return results
