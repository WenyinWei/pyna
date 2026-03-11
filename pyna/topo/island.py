"""Island width estimation and rational surface location.

Functions for finding q = m/n rational surfaces on a flux-surface
coordinate S and estimating the half-width of magnetic islands driven
by resonant magnetic perturbations (RMPs).

References
----------
* Chirikov (1979): standard map / island overlap criterion.
* White (2014): *Theory of Tokamak Plasmas*, Ch. 4.
* MHDpy ``resonant/rationalq.py`` (original implementation,
  Wenyin Wei, EAST/Tsinghua).
"""
from __future__ import annotations

from math import gcd
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locate_rational_surface(
    S: np.ndarray,
    q_profile: np.ndarray,
    m: int,
    n: int,
    s: float = 0.01,
) -> List[float]:
    """Find the S locations where the safety factor q(S) = m/n.

    Uses a smoothing spline fit to ``q_profile`` on ``S`` and then
    calls :py:meth:`scipy.interpolate.UnivariateSpline.roots` on
    ``q_spline - m/n``.

    Parameters
    ----------
    S:
        1-D array of flux-surface labels (effective minor radius, 0–1).
    q_profile:
        1-D array of safety-factor values at each ``S``.  May contain
        NaN values; those points are excluded from the spline fit.
    m:
        Poloidal mode number (integer, may be negative or zero).
    n:
        Toroidal mode number (positive integer).
    s:
        Spline smoothing factor passed to :class:`UnivariateSpline`.

    Returns
    -------
    list of float
        S values where q = m/n, in ascending order.  Empty list if
        no such crossing exists within the domain.
    """
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
    """Find all rational surfaces q = m/n for |m| ≤ m_max, 1 ≤ n ≤ n_max.

    Parameters
    ----------
    S:
        1-D array of flux-surface labels.
    q_profile:
        1-D array of safety-factor values.
    m_max:
        Maximum absolute value of the poloidal mode number.
    n_max:
        Maximum toroidal mode number.
    s:
        Spline smoothing factor.

    Returns
    -------
    dict[int, dict[int, list[float]]]
        ``result[m][n]`` is a list of S locations where q = m/n.
        Negative m values are included (as in the original MHDpy code).
    """
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
    r"""Estimate the half-width of a magnetic island at a rational surface.

    The island half-width in the S coordinate is given by

    .. math::

        w = \sqrt{ \frac{4 q^2 |\tilde{b}_{mn}|}{|m \, \mathrm{d}q/\mathrm{d}S|} }

    where all quantities are evaluated at the rational surface
    S = S_res.

    Parameters
    ----------
    m:
        Poloidal mode number.
    n:
        Toroidal mode number.
    S_res:
        S location of the rational surface (q = m/n).
    S:
        1-D array of flux-surface labels.
    q_profile:
        1-D array of safety-factor values on ``S``.
    tilde_b_mn:
        Array of normalised perturbation amplitudes.  Indexing
        depends on ``tilde_b_mn_index``:

        * If ``tilde_b_mn_index`` is ``None``, ``tilde_b_mn`` is
          treated as a 1-D array over S and used directly.
        * Otherwise, ``tilde_b_mn_index = (m_idx, n_idx)`` selects
          a slice from a multi-dimensional array.
    tilde_b_mn_index:
        Optional tuple ``(m_idx, n_idx)`` for slicing ``tilde_b_mn``.

    Returns
    -------
    float
        Island half-width in the S coordinate.
    """
    # Extract the 1-D profile of |tilde_b_mn| over S
    if tilde_b_mn_index is not None:
        m_idx, n_idx = tilde_b_mn_index
        b_profile = tilde_b_mn[:, m_idx, n_idx]
    else:
        b_profile = tilde_b_mn

    b_res = float(2.0 * np.abs(interp1d(S, b_profile)(S_res)))

    # Build q spline, evaluate q and dq/dS at S_res
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
    """Enumerate all unique q = m/n rational values (m, n > 0).

    Rational values that are equal (i.e. share the same reduced
    fraction) are grouped together.  Optionally filtered to the
    window [q_min, q_max].

    Parameters
    ----------
    m_max:
        Maximum poloidal mode number.
    n_max:
        Maximum toroidal mode number.
    q_min:
        Lower bound on q (inclusive).  ``None`` means no lower bound.
    q_max:
        Upper bound on q (inclusive).  ``None`` means no upper bound.

    Returns
    -------
    list of list of [m, n]
        Each element is a group of ``[m, n]`` pairs that all
        represent the same rational q value.  Within a group the
        first element has the smallest m and n (reduced fraction).
    """
    mn_list = [[m, n] for m in range(1, m_max + 1) for n in range(1, n_max + 1)]
    result: List[List[List[int]]] = []

    while mn_list:
        mn1 = mn_list.pop(0)
        q_val = mn1[0] / mn1[1]

        # Apply optional range filter early
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
