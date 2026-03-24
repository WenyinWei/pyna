"""Island width estimation, rational surface location, and island topology.

This module provides:

* Functions for finding q = m/n rational surfaces on a flux-surface
  coordinate S and estimating the half-width of magnetic islands driven
  by resonant magnetic perturbations (RMPs).
* ``Island`` — data class representing a single magnetic island (one O-point
  of an island chain) together with its neighbouring X-points, estimated
  half-width, and position in the island-around-island hierarchy.
* ``IslandChain`` — data class representing a full chain of islands
  (m O-points + m X-points for q = m/n) with connectivity tracking and
  support for disconnected sub-chains (common near the plasma boundary).

References
----------
* Chirikov (1979): standard map / island overlap criterion.
* White (2014): *Theory of Tokamak Plasmas*, Ch. 4.
* MHDpy ``resonant/rationalq.py`` (original implementation,
  Wenyin Wei, EAST/Tsinghua).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


# ---------------------------------------------------------------------------
# Island topology data classes
# ---------------------------------------------------------------------------

@dataclass
class Island:
    """A single magnetic island (one O-point region of an island chain).

    An island is centred on an O-point (elliptic fixed point) and bounded
    by separatrices that pass through the neighbouring X-points (hyperbolic
    fixed points).

    Attributes
    ----------
    period_n : int
        Period of the Poincaré map (number of toroidal turns to close).
        Equal to the numerator m for a q = m/n island.
        Also accessible as ``period_m`` (preferred name).
    O_point : ndarray, shape (2,)
        Coordinates [R, Z] of the elliptic fixed point (island centre).
    X_points : list of ndarray, shape (2,) each
        Coordinates of the neighbouring hyperbolic fixed points.
        May be empty when X-points have not yet been located.
    halfwidth : float
        Estimated island half-width (in the S flux-coordinate or in metres,
        depending on the context).  ``nan`` when not yet estimated.
    level : int
        Hierarchy level in the island-around-island (Birkhoff) structure.
        Level 1 = primary island chain; level 2 = island inside a level-1
        island; etc.
    parent : Island or None
        The parent ``Island`` in the hierarchy (None for level-1 islands).
    label : str or None
        Optional human-readable identifier (e.g. ``"3/1"``).

    Examples
    --------
    >>> isl = Island(period_n=3, O_point=np.array([3.07, 0.0]),
    ...              X_points=[np.array([3.12, 0.0])], halfwidth=0.05)
    >>> isl.level
    1
    """
    period_n: int
    O_point: np.ndarray
    X_points: List[np.ndarray] = field(default_factory=list)
    halfwidth: float = float("nan")
    level: int = 1
    parent: Optional[Island] = None
    label: Optional[str] = None

    @property
    def period_m(self) -> int:
        """Number of toroidal turns per orbit (= m in q=m/n). Preferred over period_n."""
        return self.period_n

    def __post_init__(self):
        self.O_point = np.asarray(self.O_point, dtype=float)
        self.X_points = [np.asarray(x, dtype=float) for x in self.X_points]

    def explore_sub_islands(
        self,
        field_func,
        n_turns_range: range = None,
        r_scan_factor: float = 0.3,
        n_scan: int = 100,
        tol: float = 1e-10,
        verbose: bool = False,
    ) -> "List[Island]":
        """Rough exploration of sub-islands (islands-around-islands) near this island.

        Searches for periodic orbits of shorter periods nested inside or around
        this island.  This implements a coarse version of the
        *island-around-island* (Birkhoff) hierarchy.

        The search scans rings of radii ranging from a small fraction of the
        island half-width up to ~2× the half-width, looking for fixed points
        of Poincaré maps of various periods.

        Parameters
        ----------
        field_func : callable
            ``field_func(r, z, phi) → [dR/dφ, dZ/dφ]`` — 2-D field
            direction function.
        n_turns_range : range or None
            Range of orbit periods (number of Poincaré map iterations) to
            search.  If ``None``, searches periods 1 through
            ``max(2, self.period_n - 1)``.
        r_scan_factor : float
            Scan ring radius as a multiple of ``self.halfwidth``.  If
            ``halfwidth`` is ``nan``, a default of 0.05 m is used.  Rings are
            placed at ``[0.3, 0.7, 1.2, 1.8] × r_scan_factor × halfwidth``.
            Default 0.3.
        n_scan : int
            Number of test points on each ring.  Default 100.
        tol : float
            Newton refinement tolerance.  Default 1e-10.
        verbose : bool
            Print progress messages.

        Returns
        -------
        list of Island
            Sub-islands found.  Each has ``level = self.level + 1`` and
            ``parent = self``.  Returns an empty list if none are found.
        """
        from pyna.topo.fixed_points import (
            scan_fixed_point_seeds,
            refine_fixed_point,
            classify_fixed_point,
        )

        hw = self.halfwidth if np.isfinite(self.halfwidth) and self.halfwidth > 0 else 0.05
        base_r = hw * r_scan_factor

        if n_turns_range is None:
            max_period = max(2, self.period_n - 1) if self.period_n > 1 else 3
            n_turns_range = range(1, max_period + 1)

        scan_radii = [0.3 * base_r, 0.7 * base_r, 1.2 * base_r, 1.8 * base_r]

        found_fps: List[np.ndarray] = []
        sub_islands: List[Island] = []

        for n_turns in n_turns_range:
            all_seeds: List[np.ndarray] = []
            for r in scan_radii:
                if r < 1e-8:
                    continue
                seeds = scan_fixed_point_seeds(
                    field_func,
                    float(self.O_point[0]),
                    float(self.O_point[1]),
                    r,
                    n_turns,
                    n_scan=n_scan,
                )
                all_seeds.extend(seeds[:4])  # take top 4 candidates per ring

            refined: List[np.ndarray] = []
            dedup_tol = hw * 0.05
            for seed in all_seeds:
                fp = refine_fixed_point(seed, field_func, n_turns, tol=tol)
                if fp is None:
                    continue
                # Deduplicate
                if all(np.linalg.norm(fp - q) > dedup_tol for q in refined + found_fps):
                    refined.append(fp)

            for fp in refined:
                fp_type, _, _ = classify_fixed_point(fp, field_func, n_turns)
                if fp_type == "O":
                    isl = Island(
                        period_n=n_turns,
                        O_point=fp,
                        X_points=[],
                        halfwidth=float("nan"),
                        level=self.level + 1,
                        parent=self,
                        label=f"sub[{n_turns}]@{self.label or '?'}",
                    )
                    sub_islands.append(isl)
                    found_fps.append(fp)
                    if verbose:
                        print(f"  Sub-island period={n_turns} at R={fp[0]:.4f} Z={fp[1]:.4f}")

        return sub_islands


@dataclass
class IslandChain:
    """A chain of magnetic islands sharing the same q = m/n rational surface.

    For a resonance q = m/n there are m O-points and m X-points arranged
    symmetrically around the rational surface.  Near the plasma boundary,
    a high-mode-number chain may be broken into several *disconnected
    sub-chains* (e.g. by a strong stochastic layer).

    Attributes
    ----------
    m : int
        Poloidal mode number (numerator of q = m/n).
    n : int
        Toroidal mode number (denominator of q = m/n).
    islands : list of Island
        All ``Island`` instances in this chain (one per O-point found).
    connected : bool
        ``True`` when all islands form a single connected component (the
        typical inner-region case).  ``False`` when the chain has been
        broken into separate, non-communicating sub-chains.
    subchains : list of IslandChain
        If ``connected=False``, this list holds the individual disconnected
        sub-chains.  Empty when ``connected=True``.
    level : int
        Hierarchy level (mirrors ``Island.level``).

    Properties
    ----------
    period_n : int
        Equal to ``m`` (the number of toroidal turns to close the orbit).
    n_islands : int
        Number of ``Island`` objects currently stored.
    q_rational : float
        Rational safety factor m/n of this chain.

    Examples
    --------
    Build a chain from two O-points and two X-points:

    >>> isl0 = Island(period_n=2, O_point=np.array([3.1, 0.05]))
    >>> isl1 = Island(period_n=2, O_point=np.array([3.1, -0.05]))
    >>> chain = IslandChain(m=2, n=1, islands=[isl0, isl1])
    >>> chain.n_islands
    2
    >>> chain.q_rational
    2.0
    """
    m: int
    n: int
    islands: List[Island] = field(default_factory=list)
    connected: bool = True
    subchains: List[IslandChain] = field(default_factory=list)
    level: int = 1

    @property
    def period_n(self) -> int:
        """Number of toroidal turns per orbit (= m). Backward-compat alias for period_m."""
        return self.m

    @property
    def period_m(self) -> int:
        """Number of toroidal turns per orbit (= m in q=m/n). Preferred name."""
        return self.m

    @property
    def n_islands(self) -> int:
        """Number of ``Island`` objects currently stored."""
        return len(self.islands)

    @property
    def q_rational(self) -> float:
        """Rational safety factor m/n of this chain."""
        return self.m / self.n

    @classmethod
    def from_fixed_points(
        cls,
        O_points: List[np.ndarray],
        X_points: List[np.ndarray],
        m: int,
        n: int,
        halfwidths: Optional[List[float]] = None,
        level: int = 1,
        proximity_tol: float = 1.0,
    ) -> IslandChain:
        """Construct an ``IslandChain`` from lists of O-point and X-point arrays.

        Each O-point is matched with its nearest X-point(s) within
        ``proximity_tol`` metres.

        Parameters
        ----------
        O_points : list of ndarray, shape (2,)
            Elliptic fixed points [R, Z].
        X_points : list of ndarray, shape (2,)
            Hyperbolic fixed points [R, Z].
        m, n : int
            Mode numbers of the island chain (q = m/n).
        halfwidths : list of float or None
            Island half-widths corresponding to each O-point.
            If ``None``, all half-widths are set to ``nan``.
        level : int
            Hierarchy level for all created ``Island`` objects.
        proximity_tol : float
            Maximum distance (m) from an O-point to a candidate X-point for
            that X-point to be associated with the O-point.

        Returns
        -------
        IslandChain
        """
        if halfwidths is None:
            halfwidths = [float("nan")] * len(O_points)
        if len(halfwidths) != len(O_points):
            raise ValueError(
                f"len(halfwidths)={len(halfwidths)} must equal "
                f"len(O_points)={len(O_points)}"
            )

        period_n = m
        islands: List[Island] = []
        label_prefix = f"{m}/{n}"

        # Convert all X-points to arrays once to avoid repeated conversions
        X_points_arr = [np.asarray(xp, dtype=float) for xp in X_points]

        for idx, (op, hw) in enumerate(zip(O_points, halfwidths)):
            op = np.asarray(op, dtype=float)
            # Associate the nearest X-points with this O-point
            nearby_X = [
                xp for xp in X_points_arr
                if np.linalg.norm(xp - op) < proximity_tol
            ]
            isl = Island(
                period_n=period_n,
                O_point=op,
                X_points=nearby_X,
                halfwidth=float(hw),
                level=level,
                label=f"{label_prefix}[{idx}]",
            )
            islands.append(isl)

        return cls(m=m, n=n, islands=islands, connected=True, level=level)

    def split_into_subchains(
        self,
        connectivity_groups: List[List[int]],
    ) -> None:
        """Mark this chain as disconnected and populate ``subchains``.

        Parameters
        ----------
        connectivity_groups : list of list of int
            Each inner list contains the *indices* (into ``self.islands``) of
            the islands that form one connected sub-chain.

        Raises
        ------
        ValueError
            If any index is out of range or indices overlap.
        """
        all_indices = [i for grp in connectivity_groups for i in grp]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("connectivity_groups contains overlapping indices.")
        if any(i < 0 or i >= len(self.islands) for i in all_indices):
            raise ValueError("Index out of range in connectivity_groups.")

        self.connected = False
        self.subchains = []
        for grp in connectivity_groups:
            sub_islands = [self.islands[i] for i in grp]
            sub = IslandChain(
                m=self.m,
                n=self.n,
                islands=sub_islands,
                connected=True,
                level=self.level,
            )
            self.subchains.append(sub)

    def scan_xo_rings_parallel(
        self,
        field_func,
        r_scan: float,
        n_scan: int = 200,
        n_workers: Optional[int] = None,
        tol: float = 1e-12,
        dedup_tol: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        """Locate X/O rings for all (possibly disconnected) sub-chains in parallel.

        For a high-mode-number chain that has been split into disconnected
        sub-chains via :meth:`split_into_subchains`, this method scans for
        period-``m`` fixed points around each island O-point *in parallel*,
        populating the ``X_points`` list of each ``Island`` in-place.

        For *connected* chains, the X-points of the seed island are extended
        toroidally to locate the full ring; disconnected sub-chains are handled
        independently.

        Parameters
        ----------
        field_func : callable
            ``field_func(r, z, phi) → [dR/dφ, dZ/dφ]``.
        r_scan : float
            Ring-scan radius (m) used for seed generation.
        n_scan : int
            Points on the scan ring.
        n_workers : int or None
            Thread-pool size.  ``None`` → ``os.cpu_count()``.
        tol : float
            Newton refinement tolerance.
        dedup_tol : float
            Deduplication threshold (m).
        verbose : bool
            Print progress messages.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor
        from pyna.topo.fixed_points import (
            scan_fixed_point_seeds,
            refine_fixed_point,
            classify_fixed_point,
        )

        workers = n_workers or (os.cpu_count() or 4)

        # Collect all islands that need X-point scanning
        if self.connected:
            islands_to_scan = self.islands
        else:
            # For disconnected chains, scan one island per sub-chain
            islands_to_scan = [sub.islands[0] for sub in self.subchains if sub.islands]

        def _scan_one(isl: Island) -> List[np.ndarray]:
            seeds = scan_fixed_point_seeds(
                field_func,
                float(isl.O_point[0]),
                float(isl.O_point[1]),
                r_scan,
                self.m,
                n_scan=n_scan,
            )
            x_pts: List[np.ndarray] = []
            found_fps: List[np.ndarray] = []
            for seed in seeds:
                fp = refine_fixed_point(seed, field_func, self.m, tol=tol)
                if fp is None:
                    continue
                fp_type, _, _ = classify_fixed_point(fp, field_func, self.m)
                if fp_type == "X" and all(
                    np.linalg.norm(fp - q) > dedup_tol for q in found_fps
                ):
                    x_pts.append(fp)
                    found_fps.append(fp)
            return x_pts

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_scan_one, islands_to_scan))

        # Assign results
        if self.connected:
            for isl, x_pts in zip(islands_to_scan, results):
                isl.X_points = x_pts
                if verbose:
                    print(f"  Island O=({isl.O_point[0]:.4f}, {isl.O_point[1]:.4f}) "
                          f"→ {len(x_pts)} X-points")
        else:
            for sub, x_pts in zip(self.subchains, results):
                if sub.islands:
                    sub.islands[0].X_points = x_pts
                    if verbose:
                        op = sub.islands[0].O_point
                        print(f"  Sub-chain O=({op[0]:.4f}, {op[1]:.4f}) "
                              f"→ {len(x_pts)} X-points")


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
