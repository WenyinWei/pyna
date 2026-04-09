"""Skeleton of the new invariant-object class hierarchy.

Pure Python dataclasses — no cyna dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

if TYPE_CHECKING:
    pass


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
    DPm: np.ndarray          # 2×2 matrix
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


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

CoordT = TypeVar("CoordT")


class InvariantObject(ABC):
    """Abstract base for all invariant objects."""

    def __init__(self, ambient_dim: Optional[int] = None):
        self.ambient_dim = ambient_dim


# ---------------------------------------------------------------------------
# FixedPoint
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class FixedPoint(InvariantObject):
    phi: float
    R: float
    Z: float
    DPm: np.ndarray
    ambient_dim: Optional[int] = None

    def __post_init__(self):
        # monodromy is lazily computed from DPm via the cached_property
        pass

    @cached_property
    def monodromy(self) -> MonodromyData:
        eigs = np.linalg.eigvals(self.DPm)
        return MonodromyData(DPm=self.DPm, eigenvalues=eigs)

    @property
    def stability(self) -> Stability:
        return self.monodromy.stability


# ---------------------------------------------------------------------------
# Cycle
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Cycle(InvariantObject):
    winding: Tuple[int, ...]
    sections: Dict = field(default_factory=dict)  # phi -> List[FixedPoint], ordered by flow
    monodromy: Optional[MonodromyData] = None
    ambient_dim: Optional[int] = None

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

    def section_cut(self, phi: float) -> "FixedPoint":
        """Backward-compat: return first FixedPoint at phi."""
        pts = self.section_points(phi)
        if not pts:
            raise KeyError(phi)
        return pts[0]

    def unstable_seeds(self, phi: float, n_seeds: int, init_length: float):
        fp = self.section_cut(phi)
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
        fp = self.section_cut(phi)
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
class InvariantTorus(InvariantObject):
    rotation_vector: Tuple[float, ...]
    ambient_dim: Optional[int] = None

    def section_cut(self, section: Any = None) -> "InvariantTorus":
        rv = self.rotation_vector[:-1] if len(self.rotation_vector) > 1 else self.rotation_vector
        adim = self.ambient_dim - 1 if self.ambient_dim else None
        return InvariantTorus(rotation_vector=rv, ambient_dim=adim)


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

if TYPE_CHECKING:
    _IslandBase = InvariantObject  # type hint only
else:
    _IslandBase = object


@dataclass(eq=False)
class Island(_ToriMixin, InvariantObject):
    O_point: FixedPoint
    X_points: List[FixedPoint] = field(default_factory=list)
    ambient_dim: Optional[int] = None
    child_chains: List["IslandChain"] = field(default_factory=list)
    parent_chain: Optional["IslandChain"] = field(default=None, repr=False)
    _next: "Island | None" = field(default=None, init=False, repr=False)
    _prev: "Island | None" = field(default=None, init=False, repr=False)

    def __post_init__(self):
        _ToriMixin.__init__(self)

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
        mono = self.O_point.monodromy
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


# ---------------------------------------------------------------------------
# IslandChain
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class IslandChain(InvariantObject):
    O_points: List[FixedPoint] = field(default_factory=list)
    X_points: List[FixedPoint] = field(default_factory=list)
    ambient_dim: Optional[int] = None
    islands: List[Island] = field(default_factory=list)
    parent_island: Optional[Island] = field(default=None, repr=False)

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

    def section_xpoints(self, phi: float) -> List[FixedPoint]:
        return [fp for fp in self.X_points if fp.phi == phi]

    def section_opoints(self, phi: float) -> List[FixedPoint]:
        return [fp for fp in self.O_points if fp.phi == phi]


# ---------------------------------------------------------------------------
# Stable / Unstable Manifold
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class StableManifold(InvariantObject):
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None


@dataclass(eq=False)
class UnstableManifold(InvariantObject):
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None


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
                O_point=o_fp,
                X_points=x_fps,
                ambient_dim=(self.ambient_dim - 1) if self.ambient_dim else None,
            ))

        for k in range(m):
            islands[k]._next = islands[(k + 1) % m]
            islands[k]._prev = islands[(k - 1) % m]

        return islands


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
        o_points = [isl.O_point for isl in all_islands]
        x_points = [fp for isl in all_islands for fp in isl.X_points]
        chain = IslandChain(
            O_points=o_points,
            X_points=x_points,
            ambient_dim=(self.ambient_dim - 1) if self.ambient_dim else None,
        )
        for isl in all_islands:
            chain.add_island(isl)
        return chain

    def section_xpoints(self, phi: float) -> List[FixedPoint]:
        result: List[FixedPoint] = []
        for xc in self.X_cycles:
            if phi in xc.sections:
                result.append(xc.sections[phi])
        return result

    def section_opoints(self, phi: float) -> List[FixedPoint]:
        result: List[FixedPoint] = []
        for oc in self.O_cycles:
            if phi in oc.sections:
                result.append(oc.sections[phi])
        return result

    def summary(self) -> str:
        return (
            f"TubeChain(O_cycles={len(self.O_cycles)}, X_cycles={len(self.X_cycles)}, "
            f"tubes={len(self.tubes)}, depth={self.depth})"
        )
