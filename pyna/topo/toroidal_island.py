"""Island width estimation, rational surface location, and island topology.

This module provides:

* Functions for finding q = m/n rational surfaces on a flux-surface
  coordinate S and estimating the half-width of magnetic islands driven
  by resonant magnetic perturbations (RMPs).
* ``Island`` — new design: dataclass holding ``O_point: FixedPoint``,
  ``X_points: List[FixedPoint]``, with step()/step_back() ring navigation.
* ``IslandChain`` — new design: holds ``m``, ``n``, and list of Islands.

References
----------
* Chirikov (1979): standard map / island overlap criterion.
* White (2014): *Theory of Tokamak Plasmas*, Ch. 4.
* MHDpy ``resonant/rationalq.py`` (original implementation,
  Wenyin Wei, EAST/Tsinghua).
"""
from __future__ import annotations

import enum
import cmath
from dataclasses import dataclass, field
from math import gcd
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

from pyna.topo.invariant import InvariantSet
from pyna.topo.toroidal_invariants import FixedPoint, PeriodicOrbit
from pyna.topo.invariant_torus import InvariantTorus, _ToriMixin

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Semantic role enum (kept for backward compatibility)
# ---------------------------------------------------------------------------

class ChainRole(enum.Enum):
    """Semantic role of an :class:`IslandChain` in the island hierarchy."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NESTED = "nested"


# ---------------------------------------------------------------------------
# New Island design
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Island(_ToriMixin, InvariantSet):
    """A single magnetic island centred on one elliptic periodic orbit.

    An Island is a first-class invariant structure of a discrete map.

    Parameters
    ----------
    O_orbit : PeriodicOrbit
        Elliptic periodic orbit at the island centre.
    X_orbits : list of PeriodicOrbit
        Hyperbolic periodic orbit(s) bounding the island.
    child_chains : list of IslandChain
        Sub-chains nested inside this island.
    parent_chain : IslandChain or None
        The IslandChain that contains this island.
    label : str or None
        Human-readable tag.
    """
    O_orbit: PeriodicOrbit = field(default_factory=lambda: PeriodicOrbit())
    X_orbits: List[PeriodicOrbit] = field(default_factory=list)
    child_chains: List["IslandChain"] = field(default_factory=list)
    parent_chain: Optional["IslandChain"] = None
    label: Optional[str] = None
    # Back-references set by TubeChain after construction
    tube: Optional[Any] = field(default=None, repr=False)
    tube_chain: Optional[Any] = field(default=None, repr=False)
    resonance_index: Optional[int] = field(default=None, repr=False)
    section: Optional[Any] = field(default=None, repr=False)
    _next: Optional["Island"] = field(default=None, init=False, repr=False)
    _prev: Optional["Island"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        _ToriMixin.__init__(self)

    # ── Convenience: first orbit point ────────────────────────────────────────

    @property
    def O_point(self) -> FixedPoint:
        """First point of the elliptic orbit (convenience accessor)."""
        return self.O_orbit[0]

    @property
    def X_points(self) -> List[FixedPoint]:
        """First point from each hyperbolic orbit (convenience accessor)."""
        return [orb[0] for orb in self.X_orbits if len(orb) > 0]

    # ── Ring navigation ───────────────────────────────────────────────────────

    def step(self) -> "Island":
        """Return the next Island in the period-m ring."""
        if self._next is None:
            raise RuntimeError("Island not linked; create via Tube.section_cut()")
        return self._next

    def step_back(self) -> "Island":
        """Return the previous Island in the period-m ring."""
        if self._prev is None:
            raise RuntimeError("Island not linked; create via Tube.section_cut()")
        return self._prev

    def _set_next(self, island: "Island") -> None:
        self._next = island

    def _set_prev(self, island: "Island") -> None:
        self._prev = island

    def _set_last(self, island: "Island") -> None:
        self._prev = island

    # ── Tori mixin ────────────────────────────────────────────────────────────

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        mono = self.O_orbit.monodromy
        if mono is None:
            return (0.0,)
        eigs = mono.eigenvalues
        for eig in eigs:
            if abs(eig.imag) > 1e-10:
                angle = abs(cmath.phase(eig)) / (2 * np.pi)
                return (angle,)
        return (0.0,)

    # ── Hierarchy ─────────────────────────────────────────────────────────────

    def add_child_chain(self, chain: "IslandChain"):
        chain.parent_island = self
        self.child_chains.append(chain)

    def root_island(self) -> "Island":
        """Walk parent_chain.parent_island upward, return root Island."""
        isl = self
        while isl.parent_chain is not None and hasattr(isl.parent_chain, 'parent_island') and isl.parent_chain.parent_island is not None:
            isl = isl.parent_chain.parent_island
        return isl

    def axis_fixed_point(self) -> "FixedPoint":
        """Root Island's O_point is the magnetic axis point on the section."""
        return self.root_island().O_point

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def R(self) -> float:
        return self.O_point.R

    @property
    def Z(self) -> float:
        return self.O_point.Z

    @property
    def phi(self) -> float:
        return self.O_point.phi

    # ── InvariantSet interface ─────────────────────────────────────────────

    def section_cut(self, section) -> list:
        """Return [self] — an Island is already a map-level object."""
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Island',
            'label': self.label,
            'R': self.O_point.R,
            'Z': self.O_point.Z,
            'phi': self.O_point.phi,
            'kind': self.O_point.kind,
            'n_X_orbits': len(self.X_orbits),
            'n_child_chains': len(self.child_chains),
        }


# ---------------------------------------------------------------------------
# New IslandChain design
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class IslandChain(InvariantSet):
    """A chain of magnetic islands sharing the same q = m/n rational surface.

    Parameters
    ----------
    m : int
        Poloidal mode number (numerator of q = m/n).
    n : int
        Toroidal mode number (denominator of q = m/n).
    islands : list of Island
        All Island instances in this chain.
    parent_tube : any
        The Tube (or TubeChain) that produced this chain.
    label : str or None
        Human-readable tag.
    """
    m: int
    n: int
    islands: List[Island] = field(default_factory=list)
    parent_tube: Optional[Any] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Allow parent_island for hierarchy (set by Island.add_child_chain)
    parent_island: Optional[Island] = field(default=None, repr=False)

    def add_island(self, island: Island):
        island.parent_chain = self
        self.islands.append(island)

    @property
    def n_islands(self) -> int:
        """Number of Island objects currently stored."""
        return len(self.islands)

    @property
    def expected_n_islands(self) -> int:
        """Expected number of islands represented in this chain.

        For one independent Tube, the section cut contains ``m/gcd(m,n)``
        islands. For a full TubeChain with multiple independent Tubes, this is
        multiplied by the number of included Tubes.
        """
        islands_per_tube = self.m // gcd(self.m, self.n)
        n_tubes_included = int(self.metadata.get('n_tubes_included', 1))
        return islands_per_tube * n_tubes_included

    @property
    def n_independent_orbits(self) -> int:
        """Number of independent field-line trajectories = gcd(m, n)."""
        return gcd(self.m, self.n) if self.n > 0 else 1

    @property
    def is_connected(self) -> bool:
        """True when all islands belong to a single connected orbit (gcd(m,n)==1)."""
        return self.n_independent_orbits == 1

    @property
    def orbit_groups(self) -> List[List[Island]]:
        """Group islands by which independent orbit they belong to."""
        n_orbs = self.n_independent_orbits
        groups: List[List[Island]] = [[] for _ in range(n_orbs)]
        for idx, isl in enumerate(self.islands):
            groups[idx % n_orbs].append(isl)
        return groups

    @property
    def q_rational(self) -> float:
        """Rational safety factor m/n."""
        return self.m / self.n

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent_island
        while p is not None:
            d += 1
            p = p.parent_chain.parent_island if (p.parent_chain is not None and hasattr(p.parent_chain, 'parent_island')) else None
        return d

    # ── O/X point helpers ─────────────────────────────────────────────────────

    @property
    def O_points(self) -> List[FixedPoint]:
        """All O-type FixedPoints from islands in this chain."""
        return [isl.O_point for isl in self.islands]

    @property
    def X_points(self) -> List[FixedPoint]:
        """All X-type FixedPoints from islands in this chain (flattened)."""
        return [fp for isl in self.islands for fp in isl.X_points]

    def section_xpoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """X-points at section phi.

        First tries an exact phi-match against stored FixedPoint.phi values.
        If nothing is found (e.g. the chain was only refined at one section),
        falls back to trajectory intersection via
        ``PeriodicOrbit.section_at(phi)`` on every X_orbit, which correctly
        returns all periodic-orbit crossings at any requested toroidal angle.
        """
        pts = [fp for fp in self.X_points if abs(fp.phi - phi) < tol]
        if pts:
            return pts
        # Trajectory fallback: query every X_orbit's 3-D trajectory
        result = []
        for isl in self.islands:
            for xorb in isl.X_orbits:
                result.extend(xorb.section_at(phi, kind='X'))
        return result

    def section_opoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """O-points at section phi.

        Same fallback strategy as :meth:`section_xpoints`.
        """
        pts = [fp for fp in self.O_points if abs(fp.phi - phi) < tol]
        if pts:
            return pts
        # Trajectory fallback
        result = []
        for isl in self.islands:
            result.extend(isl.O_orbit.section_at(phi, kind='O'))
        return result

    def summary(self) -> str:
        return (
            f"IslandChain(m={self.m}, n={self.n}, islands={self.n_islands}/"
            f"{self.expected_n_islands}, connected={self.is_connected})"
        )

    def split_into_subchains(self, connectivity_groups: List[List[int]]) -> None:
        """Mark chain as having disconnected sub-groups (no-op in new design).

        In the new design, connectivity is encoded in orbit_groups.
        Kept for backward compatibility only.
        """
        pass

    def warn_if_incomplete(self, prefix: str = "") -> None:
        """Emit a warning if the chain is incomplete."""
        import warnings
        if self.n_islands < self.expected_n_islands:
            warnings.warn(
                f"{prefix}IslandChain incomplete: expected {self.expected_n_islands} "
                f"islands for m/n={self.m}/{self.n}, found {self.n_islands}."
            )

    # ── InvariantSet interface ─────────────────────────────────────────────

    def section_cut(self, section) -> list:
        """Return the list of Islands (already section-level objects)."""
        return list(self.islands)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'IslandChain',
            'label': self.label,
            'm': self.m,
            'n': self.n,
            'n_islands': self.n_islands,
            'expected_n_islands': self.expected_n_islands,
            'complete': self.n_islands == self.expected_n_islands,
            'connected': self.is_connected,
        }

    @classmethod
    def from_fixed_points(
        cls,
        O_points: List,
        X_points: List,
        m: int,
        n: int,
        proximity_tol: float = 1.0,
        **kwargs,
    ) -> "IslandChain":
        """Build an IslandChain from O-point and X-point coordinate arrays.

        Accepts either FixedPoint objects or ndarray/tuple [R, Z] pairs.
        Each O-point is matched with nearby X-points within proximity_tol.
        """
        def _to_fp(pt, phi=0.0) -> FixedPoint:
            if isinstance(pt, FixedPoint):
                return pt
            arr = np.asarray(pt, dtype=float).ravel()
            return FixedPoint(phi=phi, R=float(arr[0]), Z=float(arr[1]),
                              DPm=np.eye(2), kind='O')

        def _to_xfp(pt, phi=0.0) -> FixedPoint:
            if isinstance(pt, FixedPoint):
                return pt
            arr = np.asarray(pt, dtype=float).ravel()
            return FixedPoint(phi=phi, R=float(arr[0]), Z=float(arr[1]),
                              DPm=np.array([[2.0, 0.0], [0.0, 0.5]]), kind='X')

        o_fps = [_to_fp(p) for p in (O_points or [])]
        x_fps = [_to_xfp(p) for p in (X_points or [])]

        islands: List[Island] = []
        for o_fp in o_fps:
            nearby_x = [
                x_fp for x_fp in x_fps
                if np.hypot(x_fp.R - o_fp.R, x_fp.Z - o_fp.Z) < proximity_tol
            ]
            islands.append(Island(
                O_orbit=PeriodicOrbit(points=[o_fp]),
                X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in nearby_x],
            ))

        return cls(m=m, n=n, islands=islands, metadata={'n_tubes_included': len(islands) // max(1, (m // gcd(m, n))) if islands else 0})


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
