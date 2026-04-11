from __future__ import annotations

"""Generic invariant objects for arbitrary finite-dimensional dynamical systems.

This module is intentionally domain-agnostic.  It does not hard-code toroidal
coordinates such as ``R/Z/phi``.  Toroidal / MCF-specific invariant objects now
live in :mod:`pyna.topo.toroidal_invariants`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from pyna.topo._base import InvariantManifold
from pyna.topo.core import (
    Stability,
    LinearStabilityData,
    SectionPoint,
    PeriodicOrbit,
    Cycle,
)


MonodromyData = LinearStabilityData
"""Generic alias kept for linear stability / return-map Jacobian data."""


@dataclass(eq=False)
class FixedPoint(InvariantManifold):
    """Generic period-1 fixed point of a discrete map.

    The state lives in arbitrary finite-dimensional coordinates.  Optional
    section metadata may be attached when the fixed point is represented on a
    chosen section of a flow.
    """

    state: np.ndarray
    stability: Optional[LinearStabilityData] = None
    section_value: Optional[float] = None
    section_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float)

    @property
    def intrinsic_dim(self) -> int:
        return 0

    @property
    def ambient_dim(self) -> int:
        return int(self.state.shape[0])

    def as_section_point(self) -> SectionPoint:
        return SectionPoint(
            state=self.state.copy(),
            section_value=self.section_value,
            section_label=self.section_label,
            stability=self.stability,
            metadata=dict(self.metadata),
        )

    def as_orbit(self) -> PeriodicOrbit:
        return PeriodicOrbit(
            points=[self.as_section_point()],
            period=1,
            stability=self.stability,
            representative_state=self.state.copy(),
            metadata=dict(self.metadata),
        )

    def section_cut(self, section=None) -> list:
        return [self.as_section_point()]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "invariant_type": "FixedPoint",
            "ambient_dim": self.ambient_dim,
            "state": self.state.tolist(),
            "section_value": self.section_value,
            "section_label": self.section_label,
            "has_stability": self.stability is not None,
        }


__all__ = [
    "Stability",
    "LinearStabilityData",
    "MonodromyData",
    "SectionPoint",
    "FixedPoint",
    "PeriodicOrbit",
    "Cycle",
]
