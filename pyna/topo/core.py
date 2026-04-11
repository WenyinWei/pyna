from __future__ import annotations

"""Generic finite-dimensional topology objects.

This module is the canonical domain-agnostic core for pyna.topo.
Nothing here assumes toroidal geometry or uses hard-coded coordinate names
such as R/Z/phi.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pyna.topo._base import GeometricObject, InvariantManifold


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


@dataclass(eq=False)
class SectionPoint(GeometricObject):
    """A point on a section of a finite-dimensional dynamical system."""

    state: np.ndarray
    section_value: Optional[float] = None
    section_label: Optional[str] = None
    stability: Optional[LinearStabilityData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float)

    @property
    def ambient_dim(self) -> int:
        return int(self.state.shape[0])

    @property
    def coordinate_names(self) -> Optional[Tuple[str, ...]]:
        names = self.metadata.get("coordinate_names")
        return tuple(names) if names is not None else None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "object_type": "SectionPoint",
            "state": self.state.tolist(),
            "section_value": self.section_value,
            "section_label": self.section_label,
            "has_stability": self.stability is not None,
        }


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
        """Linear interpolation in the trajectory parameter."""
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
        """Intersect the sampled trajectory with a generic section.

        Returns a list of :class:`SectionPoint` objects carrying projected
        section coordinates plus metadata about the ambient crossing point and
        trajectory parameter value.
        """
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


@dataclass(eq=False)
class PeriodicOrbit(InvariantManifold):
    """Periodic orbit of a discrete map in arbitrary finite dimension."""

    points: List[SectionPoint] = field(default_factory=list)
    period: Optional[int] = None
    stability: Optional[LinearStabilityData] = None
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

    def section_points(self, section_value: Optional[float] = None, section_label: Optional[str] = None, tol: float = 1e-9) -> List[SectionPoint]:
        if section_value is None and section_label is None:
            return list(self.points)
        result: List[SectionPoint] = []
        for pt in self.points:
            if section_label is not None and pt.section_label != section_label:
                continue
            if section_value is not None:
                if pt.section_value is None or abs(pt.section_value - section_value) > tol:
                    continue
            result.append(pt)
        return result

    def section_cut(self, section=None) -> "PeriodicOrbit":
        if section is None:
            return self
        if isinstance(section, tuple) and len(section) == 2:
            pts = self.section_points(section_value=section[0], section_label=section[1])
        elif hasattr(section, 'label'):
            pts = self.section_points(section_label=getattr(section, 'label', None))
        else:
            pts = self.section_points(section_value=float(section))
        return PeriodicOrbit(
            points=list(pts),
            period=len(pts),
            stability=self.stability,
            representative_state=(pts[0].state.copy() if pts else self.representative_state),
            orbit_trace=self.orbit_trace,
            metadata=dict(self.metadata),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "PeriodicOrbit",
            "period": self.period,
            "ambient_dim": self.ambient_dim,
            "n_points": len(self.points),
            "has_orbit_trace": self.orbit_trace is not None,
            "has_stability": self.stability is not None,
        }


@dataclass(eq=False)
class Cycle(InvariantManifold):
    """Periodic orbit of a continuous-time flow in arbitrary finite dimension."""

    trajectory: Trajectory
    period_value: Optional[float] = None
    return_map_orbit: Optional[PeriodicOrbit] = None
    winding: Optional[Tuple[int, ...]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def ambient_dim(self) -> int:
        return self.trajectory.ambient_dim

    def section_points(self, section_value: Optional[float] = None, section_label: Optional[str] = None, tol: float = 1e-9) -> List[SectionPoint]:
        if self.return_map_orbit is not None:
            return self.return_map_orbit.section_points(section_value=section_value, section_label=section_label, tol=tol)
        return []

    def section_cut(self, section=None) -> PeriodicOrbit:
        if section is None:
            if self.return_map_orbit is not None:
                return self.return_map_orbit
            return PeriodicOrbit(points=[])

        if hasattr(section, 'detect_crossing'):
            pts = self.trajectory.section_cut(section)
            return PeriodicOrbit(
                points=pts,
                period=len(pts),
                stability=(self.return_map_orbit.stability if self.return_map_orbit is not None else None),
                representative_state=(pts[0].state.copy() if pts else None),
                metadata=dict(self.metadata),
            )

        if self.return_map_orbit is not None:
            return self.return_map_orbit.section_cut(section)
        return PeriodicOrbit(points=[])

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "Cycle",
            "ambient_dim": self.ambient_dim,
            "period_value": self.period_value,
            "winding": self.winding,
            "has_return_map_orbit": self.return_map_orbit is not None,
            "trajectory_samples": self.trajectory.n_samples,
        }
