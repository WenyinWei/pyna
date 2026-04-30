"""pyna.topo.toroidal_island — Island re-exports + rational surface utilities.

Island/IslandChain/ChainRole now live in ``pyna.topo.toroidal``.
This module keeps the public utility functions for backward compatibility.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

from pyna.topo.toroidal import Island, IslandChain, ChainRole

__all__ = [
    "Island", "IslandChain", "ChainRole",
    "locate_rational_surface",
    "locate_all_rational_surfaces",
    "island_halfwidth",
    "all_rational_q",
]


# ---------------------------------------------------------------------------
# Public API: rational surface / island width utilities
# ---------------------------------------------------------------------------

def locate_rational_surface(
    S: np.ndarray,
    q_profile: np.ndarray,
    m: int,
    n: int,
    s: float = 0.01,
) -> List[float]:
    """Find the S locations where q(S) = m/n."""
    w = ~np.isnan(q_profile)
    q_safe = np.where(w, q_profile, 0.0)
    spl = UnivariateSpline(S, q_safe - m / n, w=w.astype(float), s=s)
    return list(spl.roots())


def locate_all_rational_surfaces(
    S: np.ndarray,
    q_profile: np.ndarray,
    m_max: int = 12,
    n_max: int = 3,
    s: float = 0.01,
) -> Dict[int, Dict[int, List[float]]]:
    """Find all rational surfaces q = m/n for |m| ≤ m_max, 1 ≤ n ≤ n_max."""
    result: Dict[int, Dict[int, List[float]]] = {}
    for m in range(-m_max, m_max + 1):
        result[m] = {}
        for n in range(1, n_max + 1):
            result[m][n] = locate_rational_surface(S, q_profile, m, n, s=s)
    return result


def island_halfwidth(
    m: int,
    n: int,
    S_res: float,
    S: np.ndarray,
    q_profile: np.ndarray,
    tilde_b_mn: np.ndarray,
    tilde_b_mn_index: Optional[tuple] = None,
) -> float:
    r"""Estimate the half-width of a magnetic island at a rational surface."""
    if tilde_b_mn_index is not None:
        m_idx, n_idx = tilde_b_mn_index
        b_profile = tilde_b_mn[:, m_idx, n_idx]
    else:
        b_profile = tilde_b_mn

    b_res = float(2.0 * np.abs(interp1d(S, b_profile)(S_res)))

    w = ~np.isnan(q_profile)
    q_safe = np.where(w, q_profile, 0.0)
    q_spl = UnivariateSpline(S, q_safe, w=w.astype(float), s=0.01)
    q_res = float(q_spl(S_res))
    dqds_res = float(q_spl.derivative()(S_res))

    denominator = abs(dqds_res * m)
    if denominator == 0.0:
        return float("nan")

    return float(np.sqrt(4.0 * q_res**2 * b_res / denominator))


def all_rational_q(
    m_max: int,
    n_max: int,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
) -> List[List[List[int]]]:
    """Enumerate all unique q = m/n rational values (m, n > 0)."""
    mn_list = [[m, n] for m in range(1, m_max + 1) for n in range(1, n_max + 1)]
    result: List[List[List[int]]] = []

    while mn_list:
        mn1 = mn_list.pop(0)
        q_val = mn1[0] / mn1[1]

        if q_min is not None and q_val < q_min:
            continue
        if q_max is not None and q_val > q_max:
            continue

        group = [mn1]
        to_remove = []
        for mn2 in mn_list:
            if mn2[0] / mn2[1] == q_val:
                group.append(mn2.copy())
                to_remove.append(mn2)
        for mn3 in to_remove:
            mn_list.remove(mn3)

        result.append(group)

    return result
