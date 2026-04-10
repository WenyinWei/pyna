"""Skeleton of the invariant-object class hierarchy.

Pure Python dataclasses — no cyna dependency.

The canonical base classes live in ``pyna.topo._base``:
  - ``InvariantSet``      — root ABC (no abstract methods)
  - ``InvariantManifold`` — intermediate, adds ``intrinsic_dim``
  - ``SectionCuttable``   — mixin protocol for section-cut support
  - ``InvariantObject``   — backward-compatible alias for ``InvariantSet``

Key design principle: **discrete maps are first-class dynamical systems**,
not subordinate "section objects" of continuous flows.  An ``Island`` and
``IslandChain`` are invariant structures of a map, whether that map is a
standalone discrete system (e.g. standard map) or a Poincaré return map of
a flow.

All skeleton classes provide concrete ``section_cut()`` and
``diagnostics()`` implementations for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np

# Use the single canonical hierarchy from _base
from pyna.topo._base import InvariantObject, InvariantSet, InvariantManifold, SectionCuttable

if TYPE_CHECKING:
    pass

CoordT = TypeVar("CoordT")

# ---------------------------------------------------------------------------
# Stability data
# ---------------------------------------------------------------------------

class Stability(Enum):
    ELLIPTIC = auto()
    HYPERBOLIC = auto()
    PARABOLIC = auto()
    UNKNOWN = auto()


@dataclass(eq=False)
class MonodromyData:
    DPm: np.ndarray          # (d, d) monodromy matrix
    eigenvalues: np.ndarray  # eigenvalues of DPm

    @cached_property
    def trace(self) -> float:
        return float(np.trace(self.DPm))

    @cached_property
    def stability(self) -> Stability:
        tr = self.trace
        if abs(tr) < 2.0 - 1e-10:
            return Stability.ELLIPTIC
        elif abs(tr) > 2.0 + 1e-10:
            return Stability.HYPERBOLIC
        else:
            return Stability.PARABOLIC

    @cached_property
    def greene_residue(self) -> float:
        return (2.0 - self.trace) / 4.0

    @cached_property
    def stability_index(self) -> float:
        return self.trace / 2.0

    # ── Spectral regularity diagnostic ────────────────────────────────────────

    def spectral_regularity(
        self,
        DPk_sequence: Optional[List[np.ndarray]] = None,
        *,
        k_max: Optional[int] = None,
    ) -> float:
        """Spectral regularity index: how close eigenvalues approach 1.

        In a **regular** (KAM) region the eigenvalues of DP^k for k=1,…,m
        approach 1 as k → m in a smooth, quasi-linear manner.  In a
        **chaotic** region the eigenvalue moduli fluctuate erratically.

        Parameters
        ----------
        DPk_sequence : list of ndarray, optional
            [DP^1, DP^2, …, DP^m].  If provided, uses the full sequence.
            If omitted, falls back to a single-matrix estimate from ``self.DPm``
            (the full period-m matrix).
        k_max : int, optional
            Used when computing the single-matrix fallback: the number of
            intermediate steps (default 1).

        Returns
        -------
        float
            The regularity index ∈ [0, ∞).  A value near 0 indicates a
            regular orbit; larger values indicate chaos.

            Defined as::

                regularity = (1/m) Σ_{k=1}^{m} max_i |log|λ_i(DP^k)|| / (k/m)

            When only ``DPm`` is available (no intermediate sequence), it
            reduces to ``max_i |log|λ_i(DPm)||``, which is the largest
            Lyapunov exponent over one full period.
        """
        if DPk_sequence is not None and len(DPk_sequence) > 0:
            return _spectral_regularity_from_sequence(DPk_sequence)

        # Fallback: single-matrix estimate from DPm
        return _spectral_regularity_single(self.eigenvalues)


# ---------------------------------------------------------------------------
# Spectral regularity helper functions
# ---------------------------------------------------------------------------

def _spectral_regularity_single(eigenvalues: np.ndarray) -> float:
    """Single-matrix regularity estimate: max |log|λ||.

    For a perfectly regular orbit at the full period, all eigenvalues
    lie on the unit circle (|λ|=1), giving regularity ≈ 0.
    """
    mods = np.abs(eigenvalues)
    mods = np.where(mods < 1e-30, 1e-30, mods)  # protect log(0)
    return float(np.max(np.abs(np.log(mods))))


def _spectral_regularity_from_sequence(DPk_sequence: List[np.ndarray]) -> float:
    r"""Regularity index from a sequence [DP^1, DP^2, …, DP^m].

    Measures how smoothly eigenvalue moduli approach 1 as k → m.

    Definition::

        R = (1/m) Σ_{k=1}^{m} max_i |log|λ_i(DP^k)||

    In a regular region each |λ_i| stays near 1 for all intermediate k,
    so R ≈ 0.  In a chaotic region |λ_i| grows exponentially with k,
    giving R ≫ 0.
    """
    m = len(DPk_sequence)
    if m == 0:
        return 0.0
    total = 0.0
    for DPk in DPk_sequence:
        eigs = np.linalg.eigvals(DPk)
        total += _spectral_regularity_single(eigs)
    return total / m


# ---------------------------------------------------------------------------
# FixedPoint
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class FixedPoint(InvariantManifold):
    """One point of a periodic orbit of a discrete map, with monodromy data.

    Mathematically, a ``FixedPoint`` is a single point x₀ where P^m(x₀) = x₀
    for a discrete map P and period m ≥ 1.  It carries the monodromy matrix
    DP^m evaluated at x₀.

    A **period-1** orbit contains exactly one FixedPoint; a **period-m** orbit
    contains m FixedPoints linked by the map (x₁ → x₂ → … → xₘ → x₁).  Use
    :class:`PeriodicOrbit` to represent the complete orbit.

    .. note::
       A FixedPoint is a *special case* of a single-point periodic orbit.
       For a period-1 orbit, ``PeriodicOrbit(points=[fp])`` and ``fp`` itself
       are interchangeable.

    ``FixedPoint`` is an :class:`InvariantManifold` with ``intrinsic_dim = 0``.
    It generalises to arbitrary phase-space dimension via the ``coords`` field.

    Parameters
    ----------
    phi : float
        Section angle (e.g. toroidal angle φ for MCF).  Stored also as
        ``section_angle`` for domain-agnostic code.
    R, Z : float
        **MCF backward-compat** — poloidal position.  For a generic
        dynamical system, use ``coords`` directly and leave ``R``/``Z`` at
        their defaults (they are auto-filled from ``coords`` when possible).
    DPm : ndarray (d, d)
        Monodromy matrix (period-m Jacobian of the Poincaré map).
    kind : str
        Stability type: ``'X'`` (hyperbolic) or ``'O'`` (elliptic).
        Auto-derived from ``DPm`` when empty.
    DX_pol_accum : ndarray or None
        Accumulated variational matrix up to this section.
    ambient_dim : int or None
        Ambient phase-space dimension.
    coords : ndarray or None
        Generic phase-space coordinates.  When ``None`` (default) it is
        auto-constructed from ``(R, Z)`` for backward compatibility.
    coordinate_names : tuple of str or None
        Human-readable names for each coordinate axis, e.g.
        ``('R', 'Z')`` or ``('q1', 'q2', 'q3', 'p1', 'p2', 'p3')``.
    section_angle : float or None
        Domain-agnostic alias for ``phi`` (the section parameter).
    """
    phi: float = 0.0
    R: float = 0.0
    Z: float = 0.0
    DPm: np.ndarray = field(default_factory=lambda: np.eye(2))
    kind: str = ''                          # 'X' or 'O'; auto-derived when empty
    DX_pol_accum: Optional[np.ndarray] = None
    ambient_dim: Optional[int] = None
    coords: Optional[np.ndarray] = field(default=None, repr=False)
    coordinate_names: Optional[Tuple[str, ...]] = field(default=None, repr=False)
    section_angle: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        # --- Sync coords ↔ (R, Z) -----------------------------------------
        if self.coords is None:
            # Build coords from R, Z (MCF backward compat)
            self.coords = np.array([self.R, self.Z], dtype=float)
        else:
            self.coords = np.asarray(self.coords, dtype=float)
            # Back-fill R, Z from coords when they were left at their
            # dataclass defaults (exactly 0.0).  This is an intentional
            # exact float comparison against the default sentinel value.
            if len(self.coords) >= 2 and self.R == 0.0 and self.Z == 0.0:
                self.R = float(self.coords[0])
                self.Z = float(self.coords[1])

        # --- Sync section_angle ↔ phi ------------------------------------
        if self.section_angle is None:
            self.section_angle = self.phi
        elif self.phi == 0.0:  # phi at dataclass default → take section_angle
            self.phi = self.section_angle

        # --- Auto-derive kind from DPm -----------------------------------
        if not self.kind:
            tr = float(np.trace(self.DPm))
            self.kind = 'O' if abs(tr) < 2.0 - 1e-10 else 'X'

    # ── InvariantManifold properties ──────────────────────────────────────────

    @property
    def intrinsic_dim(self) -> int:
        return 0

    # ── Monodromy helpers ────────────────────────────────────────────────────

    @cached_property
    def monodromy(self) -> MonodromyData:
        eigs = np.linalg.eigvals(self.DPm)
        return MonodromyData(DPm=self.DPm, eigenvalues=eigs)

    @property
    def stability(self) -> Stability:
        return self.monodromy.stability

    @property
    def greene_residue(self) -> float:
        """Greene's residue = (2 - Tr(DPm)) / 4."""
        return float((2.0 - np.trace(self.DPm)) / 4.0)

    # ── Array-like interface (backward compat with ndarray O_point) ───────────

    def __getitem__(self, idx: int) -> float:
        """fp[0] → coords[0],  fp[1] → coords[1], …  (treats fp as a vector)."""
        return float(self.coords[idx])

    def __len__(self) -> int:
        return len(self.coords)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """np.asarray(fp) → coords array."""
        arr = self.coords.copy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    # ── InvariantSet interface ────────────────────────────────────────────────

    def section_cut(self, section=None) -> list:
        """A FixedPoint is already a section-level object; return [self]."""
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'invariant_type': 'FixedPoint',
            'phi': self.phi,
            'kind': self.kind,
            'greene_residue': self.greene_residue,
            'coords': self.coords.tolist(),
        }
        # MCF convenience keys
        if len(self.coords) >= 2:
            d['R'] = float(self.coords[0])
            d['Z'] = float(self.coords[1])
        return d

    # ── Orbit construction ───────────────────────────────────────────────────

    def as_orbit(self) -> "PeriodicOrbit":
        """Wrap this single FixedPoint into a period-1 :class:`PeriodicOrbit`."""
        return PeriodicOrbit(points=[self])


# ---------------------------------------------------------------------------
# PeriodicOrbit  (discrete-map periodic orbit)
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class PeriodicOrbit(InvariantManifold):
    """A periodic orbit of a discrete map.

    A period-*m* orbit consists of *m* points {x₁, x₂, …, xₘ} satisfying
    P(xₖ) = xₖ₊₁ (indices mod m) and P^m(xₖ) = xₖ.  The monodromy matrix
    DP^m is the same at every orbit point and is inherited from the first
    :class:`FixedPoint`.

    For a map obtained as a Poincaré return map of a continuous flow, the
    orbit points are all on the **same** Poincaré section.  For a standalone
    map (e.g. standard map, Hénon map), no section concept is required.

    ``PeriodicOrbit`` is a first-class invariant object of a discrete map,
    **not** a subordinate "section object" of a flow.

    Parameters
    ----------
    points : list of FixedPoint
        The *m* orbit points, **ordered by map iteration**: P(points[k]) =
        points[(k+1) % m].
    ambient_dim : int or None
        Ambient phase-space dimension.

    Relationship to other classes
    -----------------------------
    - **FixedPoint** is a period-1 PeriodicOrbit (convenience special case).
    - **Cycle** is the continuous-time counterpart: a closed orbit of a flow
      that intersects multiple Poincaré sections.  ``Cycle.section_cut(phi)``
      produces the ``PeriodicOrbit`` at that section.
    - **Island** is the region around an elliptic PeriodicOrbit.
    """
    points: List[FixedPoint] = field(default_factory=list)
    ambient_dim: Optional[int] = None

    @property
    def period(self) -> int:
        """Period *m*: number of points in the orbit."""
        return len(self.points)

    @property
    def intrinsic_dim(self) -> int:
        """A discrete-map periodic orbit is a finite set → intrinsic_dim = 0."""
        return 0

    @property
    def DPm(self) -> np.ndarray:
        """Monodromy matrix (DP^m), inherited from the first orbit point."""
        if self.points:
            return self.points[0].DPm
        return np.eye(2)

    @property
    def monodromy(self) -> Optional[MonodromyData]:
        """MonodromyData from the first orbit point (None if no points)."""
        return self.points[0].monodromy if self.points else None

    @property
    def stability(self) -> Stability:
        m = self.monodromy
        return m.stability if m is not None else Stability.UNKNOWN

    @property
    def kind(self) -> str:
        """'X' (hyperbolic) or 'O' (elliptic), from the first orbit point."""
        return self.points[0].kind if self.points else ''

    @property
    def coords(self) -> np.ndarray:
        """Coordinates of the first orbit point (convenience)."""
        return self.points[0].coords if self.points else np.array([])

    # ── Iteration ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> FixedPoint:
        """orbit[k] → the k-th orbit point (0-indexed)."""
        return self.points[idx]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    # ── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def from_fixed_point(cls, fp: FixedPoint) -> "PeriodicOrbit":
        """Create a period-1 orbit from a single FixedPoint."""
        return cls(points=[fp], ambient_dim=fp.ambient_dim)

    # ── InvariantSet interface ───────────────────────────────────────────────

    def section_cut(self, section=None) -> list:
        """Return the orbit points as a list."""
        return list(self.points)

    def diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'invariant_type': 'PeriodicOrbit',
            'period': self.period,
            'kind': self.kind,
        }
        if self.points:
            d['coords_first'] = self.points[0].coords.tolist()
        return d


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Cycle(InvariantManifold):
    winding: Tuple[int, ...]
    sections: Dict = field(default_factory=dict)  # phi -> List[FixedPoint], ordered by flow
    monodromy: Optional[MonodromyData] = None
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> int:
        """A cycle (periodic orbit) has intrinsic dimension 1."""
        return 1

    @property
    def stability(self) -> Stability:
        if self.monodromy is not None:
            return self.monodromy.stability
        for fps in self.sections.values():
            fp = fps[0] if isinstance(fps, list) else fps
            return fp.stability
        return Stability.UNKNOWN

    def section_points(self, phi: float, tol: float = 1e-6) -> list:
        """Return all FixedPoints at this section, ordered by flow direction."""
        for key, fps in self.sections.items():
            if abs(key - phi) < tol:
                return fps if isinstance(fps, list) else [fps]
        return []

    def section_cut(self, section=None) -> list:
        """Return FixedPoints at the given section.

        Parameters
        ----------
        section : float or object with .phi attribute, optional
            If float, treated as phi angle.  If omitted, return all points
            from the first section.
        """
        if section is None:
            # Return all points from first section
            for fps in self.sections.values():
                return fps if isinstance(fps, list) else [fps]
            return []
        phi = float(section) if isinstance(section, (int, float)) else float(getattr(section, 'phi', section))
        pts = self.section_points(phi)
        if not pts:
            raise KeyError(phi)
        return pts

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Cycle',
            'winding': self.winding,
            'n_sections': len(self.sections),
            'stability': self.stability.name if self.stability else 'UNKNOWN',
        }

    def unstable_seeds(self, phi: float, n_seeds: int, init_length: float):
        fps = self.section_cut(phi)
        fp = fps[0]
        mono = self.monodromy or fp.monodromy
        eigs = mono.eigenvalues
        idx = int(np.argmax(np.abs(eigs)))
        evec = np.linalg.eig(mono.DPm)[1][:, idx]
        evec = evec.real / np.linalg.norm(evec.real)
        angles = np.linspace(0, 2 * np.pi, n_seeds, endpoint=False)
        R_arr = fp.R + init_length * evec[0] * np.cos(angles)
        Z_arr = fp.Z + init_length * evec[1] * np.sin(angles)
        return R_arr, Z_arr

    def stable_seeds(self, phi: float, n_seeds: int, init_length: float):
        fps = self.section_cut(phi)
        fp = fps[0]
        mono = self.monodromy or fp.monodromy
        eigs = mono.eigenvalues
        idx = int(np.argmin(np.abs(eigs)))
        evec = np.linalg.eig(mono.DPm)[1][:, idx]
        evec = evec.real / np.linalg.norm(evec.real)
        angles = np.linspace(0, 2 * np.pi, n_seeds, endpoint=False)
        R_arr = fp.R + init_length * evec[0] * np.cos(angles)
        Z_arr = fp.Z + init_length * evec[1] * np.sin(angles)
        return R_arr, Z_arr


# ---------------------------------------------------------------------------
# InvariantTorus
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class InvariantTorus(InvariantManifold):
    rotation_vector: Tuple[float, ...]
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> int:
        """Torus dimension = number of independent rotation angles."""
        return len(self.rotation_vector)

    def section_cut(self, section: Any = None) -> "InvariantTorus":
        rv = self.rotation_vector[:-1] if len(self.rotation_vector) > 1 else self.rotation_vector
        adim = self.ambient_dim - 1 if self.ambient_dim else None
        return InvariantTorus(rotation_vector=rv, ambient_dim=adim)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'InvariantTorus',
            'rotation_vector': self.rotation_vector,
            'ambient_dim': self.ambient_dim,
            'intrinsic_dim': self.intrinsic_dim,
        }


# ---------------------------------------------------------------------------
# _ToriMixin — shared torus management
# ---------------------------------------------------------------------------

class _ToriMixin:
    """Mixin for objects that manage a radial stack of tori."""

    def __init__(self):
        self._tori: List[InvariantTorus] = []
        self._r_vals: List[float] = []

    def add_torus(self, torus: InvariantTorus, r: float):
        import bisect
        idx = bisect.bisect_left(self._r_vals, r)
        self._r_vals.insert(idx, r)
        self._tori.insert(idx, torus)
        # invalidate cached rotation_profile
        if "rotation_profile" in self.__dict__:
            del self.__dict__["rotation_profile"]

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        raise NotImplementedError

    @cached_property
    def rotation_profile(self):
        r_vals = [0.0] + list(self._r_vals)
        rv_list = [self._central_rotation_vector()] + [t.rotation_vector for t in self._tori]
        r_arr = np.array(r_vals)
        dim = len(rv_list[0])
        rv_arr = np.array([[rv[i] for rv in rv_list] for i in range(dim)])  # (dim, n)

        if len(r_arr) < 2:
            # Only one point → constant profile
            const_val = tuple(float(rv_list[0][i]) for i in range(dim))
            def profile_constant(r: float) -> Tuple[float, ...]:
                return const_val
            return profile_constant

        try:
            from scipy.interpolate import PchipInterpolator
            interps = [PchipInterpolator(r_arr, rv_arr[i]) for i in range(dim)]
        except ImportError:
            # Fallback: piecewise linear via numpy interp
            interps = [lambda x, i=i: np.interp(x, r_arr, rv_arr[i]) for i in range(dim)]

        def profile(r: float) -> Tuple[float, ...]:
            return tuple(float(interps[i](r)) for i in range(dim))

        return profile


# ---------------------------------------------------------------------------
# Island
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Island(_ToriMixin, InvariantObject):
    """A magnetic island: a region around an elliptic periodic orbit of a map.

    An Island is an invariant structure of a **discrete map** (either a
    standalone map or a Poincaré return map of a flow).  It is a first-class
    dynamical object, not a subordinate "section object".

    The core data are:

    - ``O_orbit`` — the elliptic :class:`PeriodicOrbit` at the centre.
    - ``X_orbits`` — the hyperbolic :class:`PeriodicOrbit`\\ (s) at the
      separatrix boundary.

    Parameters
    ----------
    O_orbit : PeriodicOrbit
        Elliptic periodic orbit at the island centre.
    X_orbits : list of PeriodicOrbit
        Hyperbolic periodic orbit(s) bounding the island (may be empty).
    """
    O_orbit: PeriodicOrbit = field(default_factory=lambda: PeriodicOrbit())
    X_orbits: List[PeriodicOrbit] = field(default_factory=list)
    ambient_dim: Optional[int] = None
    child_chains: List["IslandChain"] = field(default_factory=list)
    parent_chain: Optional["IslandChain"] = field(default=None, repr=False)
    _next: "Island | None" = field(default=None, init=False, repr=False)
    _prev: "Island | None" = field(default=None, init=False, repr=False)

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

    # ── Ring navigation ──────────────────────────────────────────────────────

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

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        mono = self.O_orbit.monodromy
        if mono is None:
            return (0.0,)
        eigs = mono.eigenvalues
        import cmath
        for eig in eigs:
            if abs(eig.imag) > 1e-10:
                angle = abs(cmath.phase(eig)) / (2 * np.pi)
                return (angle,)
        return (0.0,)

    def add_child_chain(self, chain: "IslandChain"):
        chain.parent_island = self
        self.child_chains.append(chain)

    def root_island(self) -> "Island":
        """Walk parent_chain.parent_island upward, return root Island."""
        isl = self
        while isl.parent_chain is not None and isl.parent_chain.parent_island is not None:
            isl = isl.parent_chain.parent_island
        return isl

    def axis_fixed_point(self) -> "FixedPoint":
        """Root Island's O_point is the magnetic axis point on the section."""
        return self.root_island().O_point

    # ── InvariantObject interface ─────────────────────────────────────────────

    def section_cut(self, section=None) -> list:
        """An Island is already a section-level object; return [self]."""
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Island',
            'R': self.O_point.R,
            'Z': self.O_point.Z,
            'phi': self.O_point.phi,
            'kind': self.O_point.kind,
            'n_X_points': len(self.X_points),
            'n_child_chains': len(self.child_chains),
        }


# ---------------------------------------------------------------------------
# IslandChain
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class IslandChain(InvariantObject):
    """A chain of islands of a discrete map sharing the same resonance.

    An ``IslandChain`` is a first-class invariant structure of a discrete
    map — it represents all the islands of one rational resonance m:n.

    Connectivity
    ------------
    Not all islands in a chain are connected by single map iterations.
    For example, W7X's 5/5 island chain has ``gcd(5, 5) = 5`` independent
    orbits, so each island is disconnected from the others.  Use
    :attr:`is_connected` and :attr:`orbit_groups` to inspect connectivity.
    """
    islands: List[Island] = field(default_factory=list)
    winding: Tuple[int, ...] = (1,)
    ambient_dim: Optional[int] = None
    parent_island: Optional[Island] = field(default=None, repr=False)

    # ── Convenience: aggregate O/X points ────────────────────────────────────

    @property
    def O_points(self) -> List[FixedPoint]:
        """All O-type FixedPoints (first point of each island's O_orbit)."""
        return [isl.O_point for isl in self.islands]

    @property
    def X_points(self) -> List[FixedPoint]:
        """All X-type FixedPoints (first point of each island's X_orbits, flattened)."""
        return [fp for isl in self.islands for fp in isl.X_points]

    # ── Connectivity (W7X 5/5 example: gcd(5,5) = 5 independent orbits) ─────

    @property
    def n_independent_orbits(self) -> int:
        """Number of independent map orbits within this chain.

        Equal to gcd(m, n) for winding = (m, n).  A chain is connected
        iff ``n_independent_orbits == 1``.
        """
        from math import gcd
        if len(self.winding) >= 2 and self.winding[1] > 0:
            return gcd(int(self.winding[0]), int(self.winding[1]))
        return 1

    @property
    def is_connected(self) -> bool:
        """True when all islands belong to one orbit (gcd(m,n) == 1)."""
        return self.n_independent_orbits == 1

    @property
    def orbit_groups(self) -> List[List[Island]]:
        """Group islands by which independent orbit they belong to.

        Returns a list of ``n_independent_orbits`` sub-lists.  Within
        each sub-list, consecutive islands are linked by map iteration.
        """
        n_orbs = self.n_independent_orbits
        groups: List[List[Island]] = [[] for _ in range(n_orbs)]
        for idx, isl in enumerate(self.islands):
            groups[idx % n_orbs].append(isl)
        return groups

    # ── Hierarchy ────────────────────────────────────────────────────────────

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent_island
        while p is not None:
            d += 1
            p = p.parent_chain.parent_island if p.parent_chain else None
        return d

    def ancestors(self) -> List[Island]:
        result: List[Island] = []
        p = self.parent_island
        while p is not None:
            result.append(p)
            p = p.parent_chain.parent_island if p.parent_chain else None
        return result

    def add_island(self, island: Island):
        island.parent_chain = self
        self.islands.append(island)

    def section_xpoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """X-points at section phi."""
        return [fp for fp in self.X_points if abs(fp.phi - phi) < tol]

    def section_opoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """O-points at section phi."""
        return [fp for fp in self.O_points if abs(fp.phi - phi) < tol]

    # ── InvariantObject interface ─────────────────────────────────────────────

    def section_cut(self, section=None) -> list:
        """Return the list of Islands."""
        return list(self.islands)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'IslandChain',
            'n_islands': len(self.islands),
            'n_O_points': len(self.O_points),
            'n_X_points': len(self.X_points),
            'winding': self.winding,
            'is_connected': self.is_connected,
            'n_independent_orbits': self.n_independent_orbits,
            'depth': self.depth,
        }


# ---------------------------------------------------------------------------
# Stable / Unstable Manifold
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class StableManifold(InvariantManifold):
    """Stable manifold of a hyperbolic cycle.

    ``intrinsic_dim`` equals the number of stable eigenvalue directions.
    """
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> Optional[int]:
        """Number of stable eigenvalue directions (None if unknown)."""
        if self.cycle.monodromy is None:
            return None
        eigs = self.cycle.monodromy.eigenvalues
        return int(np.sum(np.abs(eigs) < 1.0 - 1e-10))

    def section_cut(self, section=None) -> list:
        """Return manifold branches (*section* is ignored; branches are pre-computed)."""
        return list(self.branches)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'StableManifold',
            'n_branches': len(self.branches),
            'intrinsic_dim': self.intrinsic_dim,
        }


@dataclass(eq=False)
class UnstableManifold(InvariantManifold):
    """Unstable manifold of a hyperbolic cycle.

    ``intrinsic_dim`` equals the number of unstable eigenvalue directions.
    """
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> Optional[int]:
        """Number of unstable eigenvalue directions (None if unknown)."""
        if self.cycle.monodromy is None:
            return None
        eigs = self.cycle.monodromy.eigenvalues
        return int(np.sum(np.abs(eigs) > 1.0 + 1e-10))

    def section_cut(self, section=None) -> list:
        """Return manifold branches (*section* is ignored; branches are pre-computed)."""
        return list(self.branches)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'UnstableManifold',
            'n_branches': len(self.branches),
            'intrinsic_dim': self.intrinsic_dim,
        }


# ---------------------------------------------------------------------------
# Tube
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Tube(_ToriMixin, InvariantObject):
    O_cycle: Cycle
    X_cycles: List[Cycle] = field(default_factory=list)
    ambient_dim: Optional[int] = None
    child_chains: List["TubeChain"] = field(default_factory=list)
    parent_chain: Optional["TubeChain"] = field(default=None, repr=False)

    def __post_init__(self):
        _ToriMixin.__init__(self)

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        import cmath
        if self.O_cycle.monodromy is not None:
            eigs = self.O_cycle.monodromy.eigenvalues
            for eig in eigs:
                if abs(eig.imag) > 1e-10:
                    angle = abs(cmath.phase(eig)) / (2 * np.pi)
                    return (angle,)
        return (0.0,)

    def add_child_chain(self, chain: "TubeChain"):
        chain.parent_tube = self
        self.child_chains.append(chain)

    def root_tube(self) -> "Tube":
        """Walk parent_chain.parent_tube upward, return root Tube (no parent)."""
        t = self
        while t.parent_chain is not None and t.parent_chain.parent_tube is not None:
            t = t.parent_chain.parent_tube
        return t

    def axis_fixed_point(self, phi: float) -> "FixedPoint | None":
        """Find magnetic axis fixed point at section phi (from root Tube O_cycle)."""
        root = self.root_tube()
        pts = root.O_cycle.section_points(phi)
        return pts[0] if pts else None

    def section_cut(self, phi: float) -> list:  # List[Island]
        """Return ordered Island list (m islands) for section phi.

        Ordering: if root Tube (axis) available, sort by polar angle arctan2(Z-Z0, R-R0).
        Otherwise use original data order.
        """
        o_fps = self.O_cycle.section_points(phi)
        if not o_fps:
            return []

        axis_fp = self.axis_fixed_point(phi)
        R0, Z0 = None, None
        if axis_fp is not None and len(o_fps) > 1:
            R0, Z0 = axis_fp.R, axis_fp.Z
            o_fps = sorted(o_fps, key=lambda fp: np.arctan2(fp.Z - Z0, fp.R - R0))

        m = len(o_fps)
        islands = []
        for k in range(m):
            o_fp = o_fps[k]
            x_fps = []
            for xc in self.X_cycles:
                xpts = xc.section_points(phi)
                if R0 is not None and len(xpts) > 1:
                    xpts = sorted(xpts, key=lambda fp: np.arctan2(fp.Z - Z0, fp.R - R0))
                if k < len(xpts):
                    x_fps.append(xpts[k])
            islands.append(Island(
                O_orbit=PeriodicOrbit(points=[o_fp]),
                X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in x_fps],
                ambient_dim=(self.ambient_dim - 1) if self.ambient_dim else None,
            ))

        for k in range(m):
            islands[k]._next = islands[(k + 1) % m]
            islands[k]._prev = islands[(k - 1) % m]

        return islands

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Tube',
            'n_X_cycles': len(self.X_cycles),
            'n_child_chains': len(self.child_chains),
            'ambient_dim': self.ambient_dim,
        }


# ---------------------------------------------------------------------------
# TubeChain
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class TubeChain(InvariantObject):
    O_cycles: List[Cycle] = field(default_factory=list)
    X_cycles: List[Cycle] = field(default_factory=list)
    tubes: List[Tube] = field(default_factory=list)
    ambient_dim: Optional[int] = None
    parent_tube: Optional[Tube] = field(default=None, repr=False)

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent_tube
        while p is not None:
            d += 1
            p = p.parent_chain.parent_tube if p.parent_chain else None
        return d

    def ancestors(self) -> List[Tube]:
        result: List[Tube] = []
        p = self.parent_tube
        while p is not None:
            result.append(p)
            p = p.parent_chain.parent_tube if p.parent_chain else None
        return result

    def add_tube(self, tube: Tube):
        tube.parent_chain = self
        self.tubes.append(tube)

    def section_cut(self, phi: float) -> IslandChain:
        all_islands = []
        for tube in self.tubes:
            all_islands.extend(tube.section_cut(phi))
        chain = IslandChain(
            islands=[],
            winding=self.tubes[0].O_cycle.winding if self.tubes else (1,),
            ambient_dim=(self.ambient_dim - 1) if self.ambient_dim else None,
        )
        for isl in all_islands:
            chain.add_island(isl)
        return chain

    def section_xpoints(self, phi: float) -> List[FixedPoint]:
        """所有 X-cycles 在截面 phi 的 FixedPoint 列表（tol 容忍匹配）。"""
        result: List[FixedPoint] = []
        for xc in self.X_cycles:
            result.extend(xc.section_points(phi))
        return result

    def section_opoints(self, phi: float) -> List[FixedPoint]:
        """所有 O-cycles 在截面 phi 的 FixedPoint 列表（tol 容忍匹配）。"""
        result: List[FixedPoint] = []
        for oc in self.O_cycles:
            result.extend(oc.section_points(phi))
        return result

    def summary(self) -> str:
        return (
            f"TubeChain(O_cycles={len(self.O_cycles)}, X_cycles={len(self.X_cycles)}, "
            f"tubes={len(self.tubes)}, depth={self.depth})"
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'TubeChain',
            'n_O_cycles': len(self.O_cycles),
            'n_X_cycles': len(self.X_cycles),
            'n_tubes': len(self.tubes),
            'depth': self.depth,
        }
