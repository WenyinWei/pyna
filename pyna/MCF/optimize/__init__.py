"""pyna.MCF.optimize — Multi-objective stellarator optimization.

This subpackage collects objective functions and optimisation utilities for
stellarator configuration design.  The primary focus is on:

- Neoclassical transport (effective ripple ε_eff)
- Magnetic axis positioning
- Divertor performance (X-point field-line parallelism)
- Operational safety margin (wall clearance)

Typical usage
-------------
>>> from pyna.MCF.optimize import compute_all_objectives
>>> objs = compute_all_objectives(equilibrium, wall_R=R_w, wall_Z=Z_w)
"""

from pyna.toroidal.optimize import (
    neoclassical_epsilon_eff,
    xpoint_field_parallelism,
    magnetic_axis_position,
    wall_clearance,
    compute_all_objectives,
)

__all__ = [
    "neoclassical_epsilon_eff",
    "xpoint_field_parallelism",
    "magnetic_axis_position",
    "wall_clearance",
    "compute_all_objectives",
]
