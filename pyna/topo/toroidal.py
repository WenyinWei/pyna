from __future__ import annotations

"""Toroidal / MCF specializations and compatibility facades.

This module collects coordinate-system-specific wrappers so the generic core in
``pyna.topo.core`` stays independent of hard-coded ``R/Z/phi`` semantics.
"""

from pyna.topo.toroidal_trajectory import ToroidalTrajectory, trace_toroidal_trajectory
from pyna.topo.toroidal_invariants import FixedPoint as ToroidalSectionPoint
from pyna.topo.toroidal_invariants import PeriodicOrbit as ToroidalPeriodicOrbit
from pyna.topo.toroidal_invariants import Cycle as ToroidalCycle

__all__ = [
    "ToroidalTrajectory",
    "ToroidalSectionPoint",
    "ToroidalPeriodicOrbit",
    "ToroidalCycle",
    "trace_toroidal_trajectory",
]
