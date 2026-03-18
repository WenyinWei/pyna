# `pyna.interact` — Interactive Utilities

## Overview

`pyna.interact` provides interactive helpers for Jupyter notebooks and
matplotlib figure exploration.

---

## Subpackages

### `pyna.interact.matplotlib`

| Module | Contents |
|--------|----------|
| `pickpoints.py` | `PickPoints` — interactive matplotlib widget for clicking (R, Z) seed points on a Poincaré-section plot |

### Example

```python
from pyna.interact.matplotlib.pickpoints import PickPoints

picker = PickPoints(ax)
# Click on the figure to select seed points
# When done, picker.points contains the (R, Z) list
seed_pts = picker.points
```

Useful for interactively choosing initial conditions for field-line tracing
or island-width measurements.
