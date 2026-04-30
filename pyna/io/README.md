# `pyna.io` — Poincaré Orbit File I/O

## Overview

`pyna.io` provides lightweight helpers for saving and loading Poincaré
orbit datasets to/from disk.

---

## Module reference

| Module | Key exports |
|--------|-------------|
| `poincare_io.py` | `save_Poincare_orbits`, `load_Poincare_orbits` |

---

## Usage

```python
from pyna.io.poincare_io import save_Poincare_orbits, load_Poincare_orbits

# Save a list of (N_i, 2) arrays (one per field line)
save_Poincare_orbits('poincare_scan.npz', orbit_list)

# Load back
orbit_list = load_Poincare_orbits('poincare_scan.npz')
```

Files are stored as NumPy `.npz` archives.  Each orbit is stored under the
key `orbit_{i}`.  Metadata (date, equilibrium parameters) can optionally be
stored under the `meta` key as a JSON string.
