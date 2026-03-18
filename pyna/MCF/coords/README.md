# `pyna.MCF.coords` — Flux-Surface Coordinate Systems

## Overview

`pyna.MCF.coords` implements the principal flux-surface coordinate systems used
in tokamak and stellarator analysis.  Starting from an axisymmetric equilibrium
object, these modules build structured meshes in the target coordinates and
provide Jacobians and metric coefficients.

---

## Coordinate systems implemented

| Module | Coordinates | Key function |
|--------|-------------|--------------|
| `PEST.py` | `(psi_P, theta_P, phi)` — PEST straight-fieldline | `build_PEST_mesh`, `RZmesh_isoSTET`, `g_i_g__i_from_STET_mesh` |
| `Boozer.py` | `(psi_B, theta_B, phi_B)` — Boozer (MHD) | `build_Boozer_mesh`, `_compute_pest_jacobian` |
| `Hamada.py` | Hamada (straight J) | `build_Hamada_mesh` |
| `EqualArc.py` | Equal-arc-length poloidal angle | `build_equal_arc_mesh` |
| `coordinate.py` | Low-level helpers | `rzphi_to_xyz`, `xyz_to_rzphi`, `Jac_rz2stheta`, `RZ2STET`, `STET2RZ` |

---

## Coordinate labels (STYLE.md §4)

| System | Labels |
|--------|--------|
| Cylindrical | `R, Z, phi` |
| PEST | `psi_P, theta_P, phi` |
| Boozer | `psi_B, theta_B, phi_B` |
| Flux (normalised) | `psi_norm` ∈ [0, 1] (0 = axis, 1 = LCFS) |

---

## Quick-start: PEST mesh

```python
from pyna.MCF.coords.PEST import build_PEST_mesh

mesh = build_PEST_mesh(
    equilibrium,
    psi_grid=np.linspace(0.05, 0.95, 30),
    n_theta=64,
)
# mesh.R, mesh.Z — (N_psi, N_theta) arrays of cylindrical coordinates
# mesh.theta_P   — PEST poloidal angle grid
```

## Quick-start: Boozer mesh

```python
from pyna.MCF.coords.Boozer import build_Boozer_mesh

bmesh = build_Boozer_mesh(
    equilibrium,
    psi_grid=np.linspace(0.1, 0.9, 20),
    n_theta=64, n_phi=32,
)
# bmesh.B_mn — Fourier harmonics of |B| in Boozer coordinates
```

---

## Backward-compat shim

`pyna.coord` (top-level) is a thin re-export of this subpackage.
**Do not add new code to `pyna.coord`.**
