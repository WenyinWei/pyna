# `pyna.MCF.control` — legacy compatibility layer for toroidal topology control

## Overview

`pyna.MCF.control` now forwards to `pyna.toroidal.control`, which owns the toroidal control slice. The package provides magnetic-confinement-fusion specific compatibility exports on top of the generic `pyna.control` (FPT) framework.

For the generic FPT topology controller and theory background,
see `pyna/control/README.md`.

---

## Package status

`pyna.MCF.control` is now a **package-level facade only**.
The old leaf modules such as `wall.py`, `gap_response.py`, `island_control.py`,
`island_optimizer.py`, and `qprofile_response.py` were removed because they had
no independent implementation left beyond forwarding imports.

Use either:

- `pyna.toroidal.control` for all new code, or
- `pyna.MCF.control` package-root imports while migrating old notebooks.

Current package-root exports include:

- wall helpers: `WallGeometry`, `make_east_like_wall`
- gap response: `gap_response_matrix_fpt`
- q-profile response: `q_from_flux_surface_integral`, `q_by_fieldline_tracing`,
  `q_by_fieldline_winding`, `q_response_matrix_analytic`,
  `q_response_matrix_fd`, `iota_response_matrix`, `build_qprofile_response`
- island control / optimisation: `compute_resonant_amplitude`,
  `island_suppression_current`, `phase_control_current`,
  `multi_mode_control`, `IslandOptimizer`, `OptimisationResult`,
  `UnperturbedSurfaceReconstructor`, `compute_surface_deformation`,
  `epsilon_eff_proxy`

---

## Gap response matrix

The gap response matrix `R_gap[i, k] = ∂g_i/∂I_k` relates changes in
plasma-wall gap at monitoring point *i* to unit-current perturbations in coil *k*.

It is computed via the FPT stable-manifold shift formula, avoiding expensive
Poincaré map recomputation.

```python
from pyna.toroidal.control.gap_response import gap_response_matrix_fpt
from pyna.toroidal.control.wall import WallGeometry

wall = WallGeometry(...)
R_gap, gap_names = gap_response_matrix_fpt(
    base_field_func,
    coil_field_funcs,    # list of callables, one per coil
    wall,
    x_point,             # XPointState from pyna.control.topology_state
    field_func_key='my_equilibrium_v1',  # change to invalidate cache
)
```

The stable manifold is cached in a module-level dict keyed by
`(field_func_key, R_xpt, Z_xpt, s_max, ds)`.
Call `clear_manifold_cache()` when the equilibrium changes.

---

## q-profile response

```python
from pyna.toroidal.control.qprofile_response import q_response_matrix_analytic

# ∂q(psi_i)/∂I_k at a set of flux surfaces
dq_dI = q_response_matrix_analytic(
    equilibrium, coil_field_funcs,
    psi_grid=np.linspace(0.3, 0.7, 10),
)
```
