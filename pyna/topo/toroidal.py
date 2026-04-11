from __future__ import annotations

"""Toroidal / MCF specializations and compatibility facades.

This module collects coordinate-system-specific wrappers so the generic core in
``pyna.topo.core`` stays independent of hard-coded ``R/Z/phi`` semantics.
"""

from pyna.topo.trajectory3d import Trajectory3DToroidal as ToroidalTrajectory
from pyna.topo.trajectory3d import Trajectory3D as ToroidalTrajectoryAlias
from pyna.topo.toroidal_invariants import FixedPoint as ToroidalSectionPoint
from pyna.topo.toroidal_invariants import PeriodicOrbit as ToroidalPeriodicOrbit
from pyna.topo.toroidal_invariants import Cycle as ToroidalCycle
from pyna.topo.trajectory3d import trace_toroidal_trajectory

__all__ = [
    "ToroidalTrajectory",
    "ToroidalTrajectoryAlias",
    "ToroidalSectionPoint",
    "ToroidalPeriodicOrbit",
    "ToroidalCycle",
    "trace_toroidal_trajectory",
]
