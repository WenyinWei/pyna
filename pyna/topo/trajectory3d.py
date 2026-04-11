from __future__ import annotations

"""Legacy shim for toroidal trajectory storage.

The concrete implementation now lives in :mod:`pyna.topo.toroidal_trajectory`.
Use :class:`pyna.topo.toroidal_trajectory.ToroidalTrajectory` directly.
"""

from pyna.topo.toroidal_trajectory import ToroidalTrajectory, trace_toroidal_trajectory

Trajectory3DToroidal = ToroidalTrajectory
Trajectory3D = ToroidalTrajectory

__all__ = ["Trajectory3DToroidal", "Trajectory3D", "trace_toroidal_trajectory"]
