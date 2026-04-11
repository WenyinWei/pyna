from __future__ import annotations

"""Legacy shim for toroidal periodic-orbit traces.

The real toroidal implementation now lives in :mod:`pyna.topo.toroidal_cycle`.
"""

from pyna.topo.toroidal_cycle import (
    ToroidalPeriodicOrbitTrace,
    find_cycle,
    find_all_cycles_near_resonance,
    poincare_map_n,
    poincare_map_n_trajectory,
    jacobian_of_poincare_map,
)

PeriodicOrbit = ToroidalPeriodicOrbitTrace

__all__ = [
    "ToroidalPeriodicOrbitTrace",
    "PeriodicOrbit",
    "find_cycle",
    "find_all_cycles_near_resonance",
    "poincare_map_n",
    "poincare_map_n_trajectory",
    "jacobian_of_poincare_map",
]
