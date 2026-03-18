# `pyna.MCF.optimize` — Stellarator Optimisation Objectives

## Overview

`pyna.MCF.optimize` collects scalar physics objective functions used in
multi-objective stellarator optimisation.  Each function accepts an
*equilibrium* object and returns a single float.  **Lower is better** for all
objectives except `xpoint_field_parallelism` (higher is better).

---

## Objectives

| Function | Quantity | Lower = better? |
|----------|----------|----------------|
| `neoclassical_epsilon_eff` | Effective ripple ε_eff (neoclassical transport proxy) | ✓ |
| `xpoint_field_parallelism` | Average cos(θ) between neighbouring field-line tangents near X-points — parallelism metric for heat-load spreading | ✗ (higher = better) |
| `magnetic_axis_position` | `(R_axis, Z_axis)` — use Euclidean distance from target as objective | — |
| `wall_clearance` | Minimum LCFS-to-wall distance [m] (positive = safe) | target-dependent |
| `compute_all_objectives` | Convenience wrapper returning a `dict` | — |

---

## Usage

```python
from pyna.MCF.optimize.objectives import (
    neoclassical_epsilon_eff,
    xpoint_field_parallelism,
    compute_all_objectives,
)

eps = neoclassical_epsilon_eff(eq, n_field_lines=50, n_transits=100)
par = xpoint_field_parallelism(eq, x_points=[(1.5, -1.2)], n_fieldlines=20)

objs = compute_all_objectives(
    eq,
    wall_R=R_wall, wall_Z=Z_wall,
    x_points=[(1.5, -1.2), (1.5, 1.2)],
)
print(objs)
# {'magnetic_axis': (1.65, 0.0), 'epsilon_eff': 0.023,
#  'wall_clearance': 0.18, 'xpoint_parallelism': 0.94}
```

---

## References

- Nemov et al. (1999), *Phys. Plasmas* 6(12):4622 — ε_eff definition.
- Boozer (2015), *Rev. Mod. Phys.* 76:1071 — stellarator optimisation review.
