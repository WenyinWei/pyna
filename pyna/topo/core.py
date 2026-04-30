"""pyna.topo.core — Generic finite-dimensional topology objects.

This module is the canonical domain-agnostic core.  Nothing here assumes
toroidal geometry or uses hard-coded coordinate names such as R/Z/phi.

Class hierarchy
---------------
SectionPoint(InvariantManifold, intrinsic_dim=0)
  → toroidal.FixedPoint

PeriodicOrbit(InvariantManifold, intrinsic_dim=0)
  → toroidal.PeriodicOrbit

Cycle(InvariantManifold, intrinsic_dim=1)
  → toroidal.Cycle

Island(InvariantSet)
  → toroidal.Island

IslandChain(InvariantSet)
  → toroidal.IslandChain

Tube(InvariantSet)
  → toroidal.Tube

TubeChain(InvariantSet)
  → toroidal.TubeChain
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyna.topo._base import GeometricObject, InvariantManifold


# ────────────────────────────────────────────────────────────────────────────
# Stability
# ────────────────────────────────────────────────────────────────────────────

class Stability(Enum):
    ELLIPTIC = auto()
    HYPERBOLIC = auto()
    PARABOLIC = auto()
    UNKNOWN = auto()


@dataclass(eq=False)
class LinearStabilityData(GeometricObject):
    """Linearised stability payload for a periodic object or section point."""

    jacobian: np.ndarray
    eigenvalues: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.jacobian = np.asarray(self.jacobian, dtype=float)
        if self.eigenvalues is None:
            self.eigenvalues = np.linalg.eigvals(self.jacobian)
        else:
            self.eigenvalues = np.asarray(self.eigenvalues)

    @property
    def trace(self) -> float:
        return float(np.trace(self.jacobian))

    @property
    def stability_index(self) -> float:
        return float(self.trace / 2.0)

    @property
    def classification(self) -> Stability:
        if self.jacobian.shape == (2, 2):
            tr = self.trace
            if abs(tr) < 2.0 - 1e-10:
                return Stability.ELLIPTIC
            if abs(tr) > 2.0 + 1e-10:
                return Stability.HYPERBOLIC
            return Stability.PARABOLIC

        if self.eigenvalues is None or len(self.eigenvalues) == 0:
            return Stability.UNKNOWN

        mods = np.abs(self.eigenvalues)
        if np.allclose(mods, 1.0, atol=1e-8, rtol=1e-8):
            return Stability.ELLIPTIC
        if np.any(mods > 1.0 + 1e-8) and np.any(mods < 1.0 - 1e-8):
            return Stability.HYPERBOLIC
        return Stability.UNKNOWN

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "object_type": "LinearStabilityData",
            "trace": self.trace,
            "stability_index": self.stability_index,
            "classification": self.classification.name,
            "eigenvalues": [complex(v) for v in self.eigenvalues],
        }


# ────────────────────────────────────────────────────────────────────────────
# SectionPoint  — a point on a Poincaré section
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class SectionPoint(GeometricObject):
    """A point on a section of a finite-dimensional dynamical system.

    This is the generic root for toroidal.FixedPoint.
    The ``state`` field may be None when constructed by a subclass
    that fills it in ``__post_init__`` (e.g. FixedPoint from R,Z).
    """

    state: Optional[np.ndarray] = None
    section_value: Optional[float] = None
    section_label: Optional[str] = None
    stability_data: Optional[LinearStabilityData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state is not None:
            object.__setattr__(self, 'state', np.asarray(self.state, dtype=float))

    @property
    def ambient_dim(self) -> Optional[int]:
        if self.state is not None:
            return int(self.state.shape[0])
        return None

    @property
    def coordinate_names(self) -> Optional[Tuple[str, ...]]:
        names = self.metadata.get("coordinate_names")
        return tuple(names) if names is not None else None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "object_type": "SectionPoint",
            "state": self.state.tolist() if self.state is not None else None,
            "section_value": self.section_value,
            "section_label": self.section_label,
            "has_stability": self.stability_data is not None,
        }


# ────────────────────────────────────────────────────────────────────────────
# Trajectory  — sampled continuous-time curve
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Trajectory(GeometricObject):
    """Finite sampled trajectory of a continuous-time system."""

    states: np.ndarray
    times: np.ndarray
    time_name: str = "t"
    coordinate_names: Optional[Tuple[str, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        if self.states.ndim != 2:
            raise ValueError("Trajectory.states must have shape (N, d)")
        if self.times.ndim != 1:
            raise ValueError("Trajectory.times must have shape (N,)")
        if self.states.shape[0] != self.times.shape[0]:
            raise ValueError("Trajectory.states and Trajectory.times length mismatch")
        if self.coordinate_names is not None:
            self.coordinate_names = tuple(self.coordinate_names)
            if len(self.coordinate_names) != self.states.shape[1]:
                raise ValueError("coordinate_names length must match state dimension")

    @property
    def ambient_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.states.shape[0])

    def interpolate_at(self, t: float) -> np.ndarray:
        t = float(t)
        if self.n_samples == 0:
            raise ValueError("cannot interpolate an empty trajectory")
        if self.n_samples == 1:
            return self.states[0].copy()
        if t <= self.times[0]:
            return self.states[0].copy()
        if t >= self.times[-1]:
            return self.states[-1].copy()
        idx = int(np.searchsorted(self.times, t) - 1)
        idx = max(0, min(idx, self.n_samples - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        if abs(t1 - t0) < 1e-30:
            return self.states[idx].copy()
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * self.states[idx] + w * self.states[idx + 1]

    def component(self, i: int) -> np.ndarray:
        return self.states[:, i]

    def section_cut(self, section, *, tol: float = 1e-10) -> List[SectionPoint]:
        hits: List[SectionPoint] = []
        for i in range(self.n_samples - 1):
            x0 = self.states[i]
            x1 = self.states[i + 1]
            x_hit = section.detect_crossing(x0, x1, tol=tol)
            if x_hit is None:
                continue
            t0 = self.times[i]
            t1 = self.times[i + 1]
            f0 = section.f(x0) if hasattr(section, 'f') else 0.0
            f1 = section.f(x1) if hasattr(section, 'f') else 0.0
            denom = abs(f0) + abs(f1)
            w = 0.5 if denom < 1e-30 else abs(f0) / denom
            t_hit = (1.0 - w) * t0 + w * t1
            hits.append(SectionPoint(
                state=np.asarray(section.project(x_hit), dtype=float),
                section_value=float(t_hit),
                section_label=getattr(section, 'label', None),
                metadata={
                    'ambient_state': np.asarray(x_hit, dtype=float),
                    'time_name': self.time_name,
                    'coordinate_names': self.coordinate_names,
                    'section_object': section.__class__.__name__,
                },
            ))
        return hits

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "object_type": "Trajectory",
            "ambient_dim": self.ambient_dim,
            "n_samples": self.n_samples,
            "time_name": self.time_name,
            "coordinate_names": list(self.coordinate_names) if self.coordinate_names else None,
        }


# ────────────────────────────────────────────────────────────────────────────
# Orbit  — sampled discrete-map orbit
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Orbit(GeometricObject):
    """Finite sampled orbit of a discrete-time map."""

    states: np.ndarray
    steps: Optional[np.ndarray] = None
    coordinate_names: Optional[Tuple[str, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=float)
        if self.states.ndim != 2:
            raise ValueError("Orbit.states must have shape (N, d)")
        if self.steps is not None:
            self.steps = np.asarray(self.steps)
            if self.steps.ndim != 1 or self.steps.shape[0] != self.states.shape[0]:
                raise ValueError("Orbit.steps must have shape (N,)")
        if self.coordinate_names is not None:
            self.coordinate_names = tuple(self.coordinate_names)
            if len(self.coordinate_names) != self.states.shape[1]:
                raise ValueError("coordinate_names length must match state dimension")

    @property
    def ambient_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.states.shape[0])

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "object_type": "Orbit",
            "ambient_dim": self.ambient_dim,
            "n_samples": self.n_samples,
            "coordinate_names": list(self.coordinate_names) if self.coordinate_names else None,
        }


# ────────────────────────────────────────────────────────────────────────────
# PeriodicOrbit  — periodic orbit of a discrete map
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class PeriodicOrbit(InvariantManifold):
    """Periodic orbit of a discrete map in arbitrary finite dimension.

    This is the generic root for toroidal.PeriodicOrbit.
    """

    points: List[SectionPoint] = field(default_factory=list)
    period: Optional[int] = None
    stability_data: Optional[LinearStabilityData] = None
    representative_state: Optional[np.ndarray] = None
    orbit_trace: Optional[Orbit] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.representative_state is not None:
            self.representative_state = np.asarray(self.representative_state, dtype=float)
        if self.period is None:
            self.period = len(self.points) if self.points else None

    @property
    def intrinsic_dim(self) -> int:
        return 0

    @property
    def ambient_dim(self) -> Optional[int]:
        if self.representative_state is not None:
            return int(self.representative_state.shape[0])
        if self.points:
            return self.points[0].ambient_dim
        if self.orbit_trace is not None:
            return self.orbit_trace.ambient_dim
        return None

    @property
    def section_value(self) -> Optional[float]:
        return self.points[0].section_value if self.points else None

    @property
    def section_label(self) -> Optional[str]:
        return self.points[0].section_label if self.points else None

    def section_cut(self, section=None) -> "PeriodicOrbit":
        if section is None:
            return self
        raise ValueError(
            "PeriodicOrbit is already a reduced discrete object; cut the parent "
            "Cycle instead of selecting another section on the orbit."
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "PeriodicOrbit",
            "period": self.period,
            "ambient_dim": self.ambient_dim,
            "n_points": len(self.points),
            "has_orbit_trace": self.orbit_trace is not None,
            "has_stability": self.stability_data is not None,
        }


# ────────────────────────────────────────────────────────────────────────────
# Cycle  — periodic orbit of a continuous flow
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Cycle(InvariantManifold):
    """Periodic orbit of a continuous-time flow in arbitrary finite dimension.

    This is the generic root for toroidal.Cycle.
    The ``trajectory`` field is optional — toroidal subclasses may
    represent the cycle via a ``sections`` dict instead.
    """

    trajectory: Optional[Trajectory] = None
    period_value: Optional[float] = None
    return_map_orbit: Optional[PeriodicOrbit] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def ambient_dim(self) -> Optional[int]:
        if self.trajectory is not None:
            return self.trajectory.ambient_dim
        if self.return_map_orbit is not None:
            return self.return_map_orbit.ambient_dim
        return None

    def section_cut(self, section=None) -> PeriodicOrbit:
        if section is None:
            if self.return_map_orbit is not None:
                return self.return_map_orbit
            return PeriodicOrbit(points=[])

        if self.trajectory is not None and hasattr(section, 'detect_crossing'):
            pts = self.trajectory.section_cut(section)
            return PeriodicOrbit(
                points=pts,
                period=len(pts),
                stability=(self.return_map_orbit.stability_data if self.return_map_orbit is not None else None),
                representative_state=(pts[0].state.copy() if pts else None),
                metadata=dict(self.metadata),
            )

        if self.return_map_orbit is not None:
            if isinstance(section, tuple) and len(section) == 2:
                section_value = float(section[0]) if section[0] is not None else None
                section_label = section[1]
            elif hasattr(section, 'label'):
                section_value = None
                section_label = getattr(section, 'label', None)
            else:
                section_value = float(section)
                section_label = None
            pts = self.section_points(section_value=section_value, section_label=section_label)
            return PeriodicOrbit(
                points=list(pts),
                period=len(pts),
                stability=self.return_map_orbit.stability_data,
                representative_state=(pts[0].state.copy() if pts else self.return_map_orbit.representative_state),
                orbit_trace=self.trajectory,
                metadata=dict(self.metadata),
            )
        return PeriodicOrbit(points=[])

    def section_points(self, section_value: Optional[float] = None, section_label: Optional[str] = None, tol: float = 1e-9) -> List[SectionPoint]:
        if self.return_map_orbit is None:
            return []
        pts = list(self.return_map_orbit.points)
        if section_value is not None:
            pts = [pt for pt in pts if pt.section_value is not None and abs(pt.section_value - section_value) <= tol]
        if section_label is not None:
            pts = [pt for pt in pts if pt.section_label == section_label]
        return pts

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "Cycle",
            "ambient_dim": self.ambient_dim,
            "period_value": self.period_value,
            "has_return_map_orbit": self.return_map_orbit is not None,
            "trajectory_samples": self.trajectory.n_samples if self.trajectory else None,
        }


# ────────────────────────────────────────────────────────────────────────────
# Island  — discrete-map resonance island
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Island(InvariantManifold):
    """One island of a discrete Poincaré-map resonance structure.

    This is the generic root for toroidal.Island.

    Fields
    ------
    O_orbit : PeriodicOrbit
        Elliptic periodic orbit at the island centre.
    X_orbits : list of PeriodicOrbit
        Hyperbolic periodic orbit(s) bounding the island (may be empty).
    child_chains : list of IslandChain
        Sub-chains nested inside this island.
    parent_chain : IslandChain or None
        The IslandChain that contains this island.
    """

    O_orbit: PeriodicOrbit = field(default_factory=PeriodicOrbit)
    X_orbits: List[PeriodicOrbit] = field(default_factory=list)
    child_chains: List["IslandChain"] = field(default_factory=list)
    parent_chain: Optional["IslandChain"] = field(default=None, repr=False)
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _next: Optional["Island"] = field(default=None, init=False, repr=False)
    _prev: Optional["Island"] = field(default=None, init=False, repr=False)

    @property
    def intrinsic_dim(self) -> int:
        return 0

    @property
    def O_point(self) -> Optional[SectionPoint]:
        return self.O_orbit.points[0] if self.O_orbit.points else None

    @property
    def X_points(self) -> List[SectionPoint]:
        return [orb.points[0] for orb in self.X_orbits if orb.points]

    def step(self) -> "Island":
        if self._next is None:
            raise RuntimeError("Island not linked inside an IslandChain")
        return self._next

    def step_back(self) -> "Island":
        if self._prev is None:
            raise RuntimeError("Island not linked inside an IslandChain")
        return self._prev

    def add_child_chain(self, chain: "IslandChain") -> None:
        chain.parent_island = self
        self.child_chains.append(chain)

    def section_cut(self, section=None) -> list:
        if section is not None:
            raise ValueError(
                "Island is already a reduced discrete object; cut the parent "
                "continuous geometry instead."
            )
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "Island",
            "label": self.label,
            "has_O_orbit": bool(self.O_orbit.points),
            "n_X_orbits": len(self.X_orbits),
            "n_child_chains": len(self.child_chains),
        }


# ────────────────────────────────────────────────────────────────────────────
# IslandChain  — chain of discrete-map islands
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class IslandChain(InvariantManifold):
    """Discrete island chain obtained on a section of a flow or map.

    This is the generic root for toroidal.IslandChain.
    """

    islands: List[Island] = field(default_factory=list)
    period: Optional[int] = None
    label: Optional[str] = None
    parent_island: Optional[Island] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def intrinsic_dim(self) -> int:
        return 0

    def __post_init__(self):
        for i, isl in enumerate(self.islands):
            isl.parent_chain = self
            if self.islands:
                isl._next = self.islands[(i + 1) % len(self.islands)]
                isl._prev = self.islands[(i - 1) % len(self.islands)]
        if self.period is None and self.islands:
            self.period = len(self.islands)

    @property
    def n_islands(self) -> int:
        return len(self.islands)

    @property
    def O_points(self) -> List[SectionPoint]:
        return [isl.O_point for isl in self.islands if isl.O_point is not None]

    @property
    def X_points(self) -> List[SectionPoint]:
        pts: List[SectionPoint] = []
        for isl in self.islands:
            pts.extend(isl.X_points)
        return pts

    def add_island(self, island: Island) -> None:
        island.parent_chain = self
        if self.islands:
            prev = self.islands[-1]
            prev._next = island
            island._prev = prev
            island._next = self.islands[0]
            self.islands[0]._prev = island
        else:
            island._next = island
            island._prev = island
        self.islands.append(island)
        if self.period is None:
            self.period = len(self.islands)

    @property
    def section_value(self) -> Optional[float]:
        return self.islands[0].O_point.section_value if self.islands and self.islands[0].O_point is not None else None

    @property
    def section_label(self) -> Optional[str]:
        return self.islands[0].O_point.section_label if self.islands and self.islands[0].O_point is not None else None

    def section_cut(self, section=None) -> list:
        if section is not None:
            raise ValueError(
                "IslandChain is already a reduced discrete object; cut the parent "
                "continuous geometry instead."
            )
        return list(self.islands)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "IslandChain",
            "label": self.label,
            "period": self.period,
            "n_islands": self.n_islands,
            "n_O_points": len(self.O_points),
            "n_X_points": len(self.X_points),
        }


# ────────────────────────────────────────────────────────────────────────────
# Tube  — continuous-flow resonance zone
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Tube(InvariantManifold):
    """A resonance zone surrounding an elliptic periodic orbit of a continuous flow.

    A Tube is the continuous-time analogue of an Island: it consists of
    the nested family of invariant tori around an elliptic cycle (O_cycle),
    bounded by hyperbolic cycles (X_cycles).

    For MCF field lines, this is a magnetic island tube.  For Hamiltonian
    flows, it is a resonance zone bounded by separatrices.

    This is the generic root for toroidal.Tube.

    Fields
    ------
    O_cycle : Cycle
        The elliptic periodic orbit at the core.
    X_cycles : list of Cycle
        Hyperbolic periodic orbit(s) bounding the resonance zone.
    label : str or None
    """

    O_cycle: Cycle
    X_cycles: List[Cycle] = field(default_factory=list)
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def is_skeleton_complete(self) -> bool:
        return len(self.X_cycles) > 0

    def section_cut(self, section) -> IslandChain:
        """Cut this Tube with a section → IslandChain."""
        chain = IslandChain(label=self.label)
        # Cut O_cycle → O-point islands
        o_po = self.O_cycle.section_cut(section)
        # Cut X_cycles → X-point islands
        x_pos = [xc.section_cut(section) for xc in self.X_cycles]

        if o_po.points:
            for i, o_pt in enumerate(o_po.points):
                x_pts_for_this = []
                for x_po in x_pos:
                    if i < len(x_po.points):
                        x_pts_for_this.append(x_po.points[i])
                isl = Island(
                    O_orbit=PeriodicOrbit(points=[o_pt], period=1),
                    X_orbits=[PeriodicOrbit(points=[xp], period=1) for xp in x_pts_for_this],
                    label=self.label,
                )
                chain.add_island(isl)
        else:
            # No O-points — X-only islands (degenerate)
            for x_po in x_pos:
                for x_pt in x_po.points:
                    isl = Island(
                        O_orbit=PeriodicOrbit(points=[], period=0),
                        X_orbits=[PeriodicOrbit(points=[x_pt], period=1)],
                        label=self.label,
                    )
                    chain.add_island(isl)

        return chain

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "Tube",
            "label": self.label,
            "n_X_cycles": len(self.X_cycles),
            "has_o_cycle": self.O_cycle is not None,
        }


# ────────────────────────────────────────────────────────────────────────────
# TubeChain  — all Tubes of one resonance
# ────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class TubeChain(InvariantManifold):
    """All Tubes sharing the same resonance in a continuous flow.

    This is the generic root for toroidal.TubeChain.
    """

    tubes: List[Tube] = field(default_factory=list)
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def n_tubes(self) -> int:
        return len(self.tubes)

    def section_cut(self, section) -> IslandChain:
        """Cut all Tubes with a section → merged IslandChain."""
        chain = IslandChain(label=self.label)
        for tube in self.tubes:
            sub = tube.section_cut(section)
            for isl in sub.islands:
                chain.add_island(isl)
        return chain

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "TubeChain",
            "label": self.label,
            "n_tubes": self.n_tubes,
        }

    def summary(self) -> str:
        return f"TubeChain(label={self.label!r}, n_tubes={self.n_tubes})"


__all__ = [
    "Stability",
    "LinearStabilityData",
    "SectionPoint",
    "Trajectory",
    "Orbit",
    "PeriodicOrbit",
    "Cycle",
    "Island",
    "IslandChain",
    "Tube",
    "TubeChain",
]
