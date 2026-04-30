"""pyna.topo.regularity -- Spectral regularity diagnostics for dynamical systems.

In a **regular** region (composed of stacked invariant tori), the
eigenvalues of the variational / monodromy matrices evolve smoothly and
approach 1 at the full period.  In a **chaotic** region they fluctuate
erratically and diverge exponentially.

This module provides:

    spectral_regularity(DPk_sequence)
        Regularity index from a sequence of intermediate monodromy matrices.

    spectral_regularity_single(eigenvalues)
        Quick estimate from the eigenvalues of the full-period matrix.

    classify_orbit(eigenvalue_evolution)
        Classify an orbit as regular, resonant, weakly chaotic, or strongly
        chaotic based on the eigenvalue modulus evolution.

    hessian_regularity(D2Pm)
        Regularity diagnostic using the second-order variational tensor
        (Hessian of the Poincaré map), available when ``tangent_map(order=3)``
        has been called.

Theory
------
For a discrete map P with period m, the iterate P^k has monodromy DP^k.
In a regular region (invariant torus), the eigenvalues λ_i(DP^k)
satisfy |λ_i| ≈ 1 for all k, and converge smoothly to exactly 1 as
k → m (the full-period identity on the torus).

For a continuous flow φ^t with orbit period T, the same principle holds
for the variational matrix DX_t at intermediate times t ∈ (0, T).

The **regularity index** R quantifies the deviation from this ideal:

    R = (1/m) Σ_{k=1}^{m} max_i |log|λ_i(DP^k)||

R ≈ 0 for regular orbits, R > 0 for chaotic.

The regularity index is closely related to the finite-time Lyapunov
exponent (FTLE) but provides a *per-period* normalised diagnostic that
allows comparison across different resonance orders.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def spectral_regularity(DPk_sequence: Sequence[np.ndarray]) -> float:
    r"""Regularity index from a sequence [DP^1, DP^2, …, DP^m].

    Parameters
    ----------
    DPk_sequence : list of ndarray, shape (d, d)
        Intermediate monodromy matrices at iterates k = 1, …, m.

    Returns
    -------
    float
        Regularity index ≥ 0.  Near 0 → regular, larger → chaotic.

    Notes
    -----
    .. math::

        R = \frac{1}{m} \sum_{k=1}^{m} \max_i \bigl|\ln|\lambda_i(DP^k)|\bigr|
    """
    m = len(DPk_sequence)
    if m == 0:
        return 0.0
    total = 0.0
    for DPk in DPk_sequence:
        eigs = np.linalg.eigvals(DPk)
        total += spectral_regularity_single(eigs)
    return total / m


def spectral_regularity_single(eigenvalues: np.ndarray) -> float:
    """Quick regularity estimate from eigenvalues of one monodromy matrix.

    Returns ``max_i |log|λ_i||``.  For a perfectly regular orbit at the
    full period all |λ| = 1, giving 0.
    """
    mods = np.abs(eigenvalues)
    mods = np.where(mods < 1e-30, 1e-30, mods)  # protect log(0)
    return float(np.max(np.abs(np.log(mods))))


def classify_orbit(
    eigenvalue_evolution: np.ndarray,
    *,
    regular_threshold: float = 0.01,
    weak_chaos_threshold: float = 0.3,
) -> str:
    """Classify an orbit from its eigenvalue modulus evolution.

    Parameters
    ----------
    eigenvalue_evolution : ndarray, shape (m, d) or (m,)
        |λ_i(DP^k)| for k = 1, …, m.  If 2-D, the maximum over eigenvalue
        index *i* is taken.  If 1-D, interpreted as the dominant eigenvalue
        modulus at each iterate.
    regular_threshold : float
        Maximum mean |log|λ|| for an orbit to be classified as *regular*.
    weak_chaos_threshold : float
        Boundary between *weakly_chaotic* and *strongly_chaotic*.

    Returns
    -------
    str
        One of ``'regular'``, ``'resonant'``, ``'weakly_chaotic'``,
        ``'strongly_chaotic'``.
    """
    arr = np.asarray(eigenvalue_evolution, dtype=float)
    if arr.ndim == 2:
        arr = np.max(arr, axis=1)  # dominant eigenvalue at each iterate

    if len(arr) == 0:
        return "regular"

    log_mods = np.abs(np.log(np.clip(arr, 1e-30, None)))
    mean_log = float(np.mean(log_mods))

    # Check for resonance: |λ| ≈ 1 everywhere except near final iterate
    final_log = log_mods[-1] if len(log_mods) > 0 else 0.0
    intermediate_mean = float(np.mean(log_mods[:-1])) if len(log_mods) > 1 else 0.0

    if mean_log < regular_threshold:
        # Is this a resonance (λ exactly 1 at period)?
        if final_log < 1e-8 and intermediate_mean < regular_threshold:
            return "resonant"
        return "regular"
    elif mean_log < weak_chaos_threshold:
        return "weakly_chaotic"
    else:
        return "strongly_chaotic"


def hessian_regularity(D2Pm: np.ndarray) -> float:
    """Regularity diagnostic from the Hessian (second-order variational tensor).

    The Hessian D²P^m encodes how the linear monodromy varies across
    neighbouring orbits.  In a regular region D²P^m is bounded and varies
    smoothly; in a chaotic region it grows without bound.

    Parameters
    ----------
    D2Pm : ndarray, shape (d, d, d)
        Second-order variational tensor at the full period.  This is the
        *Q* tensor returned by ``tangent_map(order=3)`` reshaped to 3-D.

    Returns
    -------
    float
        Frobenius norm of D²P^m, normalised by the dimension.
        Larger values indicate stronger local chaos.
    """
    T = np.asarray(D2Pm, dtype=float)
    d = T.shape[0] if T.ndim >= 1 else 1
    return float(np.linalg.norm(T.ravel()) / max(d, 1))


# ---------------------------------------------------------------------------
# Convenience: eigenvalue_evolution_from_sequence
# ---------------------------------------------------------------------------

def eigenvalue_evolution_from_sequence(
    DPk_sequence: Sequence[np.ndarray],
) -> np.ndarray:
    """Extract eigenvalue modulus evolution from a monodromy sequence.

    Parameters
    ----------
    DPk_sequence : list of ndarray, shape (d, d)

    Returns
    -------
    ndarray, shape (m, d)
        |λ_i(DP^k)| for k = 1, …, m.
    """
    if not DPk_sequence:
        return np.empty((0, 0))
    eig_mods = []
    for DPk in DPk_sequence:
        eigs = np.linalg.eigvals(DPk)
        eig_mods.append(np.abs(eigs))
    return np.array(eig_mods)
