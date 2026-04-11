# `pyna.MCF` — Legacy compatibility namespace

## Overview

`pyna.MCF` is retained as a **legacy compatibility layer** for older toroidal /
magnetic-confinement-fusion code. New user code should prefer the
`pyna.toroidal` namespace.

Use `pyna.MCF` only when maintaining historical notebooks or scripts that have
not yet been migrated.

---

## Subpackage map

```
MCF/
├── equilibrium/       MHD equilibria (§1)
├── coils/             Vacuum field, Biot-Savart, RMP coil sets (§2)
├── coords/            Flux-surface coordinate systems (§3)
├── plasma_response/   Linear MHD plasma response (§4)
├── control/           MCF-specific control (gap response, q-profile) (§5)
├── diagnostics/       Connection-length and endpoint diagnostics (§6)
├── optimize/          Stellarator optimisation objectives (§7)
└── visual/            MCF-specific plotting helpers (§8)
```

---

## §1  `MCF/equilibrium/`

`pyna.MCF.equilibrium` is now a **package-level facade only**.
The old module-by-module wrappers were removed; import from
`pyna.toroidal.equilibrium` for all new code, or from the package root
`pyna.MCF.equilibrium` while migrating older notebooks.

Representative package-root exports include:

- `EquilibriumAxisym`, `EquilibriumTokamakCircularSynthetic`
- `EquilibriumSolovev`
- `recover_pressure_simplest`, `solve_GS_perturbed`
- `StellaratorSimple`, `simple_stellarator`
- `BoozerSurface`, `BoozerPerturbation`, `compute_boozer_response`
- `CylindricalGrid`, `PerturbationField`, `PlasmaResponse`,
  `compute_plasma_response`, `feedback_correction_field`,
  `iterative_equilibrium_correction`

Boozer response tools are valid only where flux surfaces exist
(non-chaotic regions). Cylindrical response tools work everywhere,
including chaotic and divertor regions.

## §2  `MCF/coils/`

| Module | Contents |
|--------|----------|
| `base.py` | `CoilFieldVacuum` (abstract), `CoilFieldSuperposition`, `CoilFieldScaled` |
| `coil.py` | `BRBZ_induced_by_current_loop`, `BRBZ_induced_by_thick_finitelen_solenoid` |
| `coil_system.py` | `CoilSet`, `Biot_Savart_field` |
| `RMP.py` | `normalize_b`, `RMP_spectrum_2d`, `island_width_at_rational_surfaces` |
| `vector_potential.py` | Vector potential for axisymmetric coil fields |
| `field.py` | Thin canonical re-export layer (imports from `pyna.fields`) |

**Key function name:** `Biot_Savart_field` — capitals per STYLE.md §1.

## §3  `MCF/coords/`

See `MCF/coords/README.md` for details.

## §4  `MCF/plasma_response/`

Linear MHD plasma response via perturbed Grad-Shafranov equation.
Primary module: `PerturbGS.py`.

## §5  `MCF/control/`

`pyna.MCF.control` is likewise a **package-level facade only**.
The old per-module wrappers were removed; import from
`pyna.toroidal.control` for new code, or from the package root
`pyna.MCF.control` while migrating.

Representative package-root exports include:

- `WallGeometry`, `make_east_like_wall`
- `gap_response_matrix_fpt`
- `q_from_flux_surface_integral`, `q_by_fieldline_tracing`,
  `q_by_fieldline_winding`, `q_response_matrix_analytic`,
  `q_response_matrix_fd`, `iota_response_matrix`, `build_qprofile_response`
- `compute_resonant_amplitude`, `island_suppression_current`,
  `phase_control_current`, `multi_mode_control`
- `IslandOptimizer`, `OptimisationResult`,
  `UnperturbedSurfaceReconstructor`, `compute_surface_deformation`,
  `epsilon_eff_proxy`

For the generic FPT-based topology controller see `pyna.control`. For the preferred toroidal import path, see `pyna.toroidal.control`.

## §6  `MCF/diagnostics/`

```python
from pyna.toroidal.diagnostics import field_line_length, field_line_endpoints

Lc = field_line_length(field_func, start_pts, phi_max=200.0)
```

## §7  `MCF/optimize/`

Scalar physics objectives for stellarator optimisation (lower = better):

| Function | Quantity |
|----------|----------|
| `neoclassical_epsilon_eff` | Effective ripple ε_eff |
| `xpoint_field_parallelism` | Divertor field-line parallelism metric |
| `magnetic_axis_position` | (R_axis, Z_axis) |
| `wall_clearance` | Minimum LCFS-to-wall distance |
| `compute_all_objectives` | Convenience wrapper returning a dict |

## §8  `MCF/visual/`

- `RMP_spectrum.py` — `plot_RMP_spectrum_2d`, mode-spectrum heatmaps
- `equilibrium.py` — flux-surface cross-section plots
- `tokamak_manifold.py` — stable/unstable manifold overlaid on Poincaré section

---

## Preferred imports

For user-facing examples and new development, prefer `pyna.toroidal.*` imports.
`pyna.MCF.*` remains valid only as a compatibility facade.

## Naming conventions (STYLE.md §1)

| ❌ Wrong | ✅ Correct |
|---------|----------|
| `biot_savart_field` | `Biot_Savart_field` |
| `rmp_coils` | `RMP_coils` |
| `mhd_response_operator` | `MHD_response_operator` |
| `grad_shafranov` | `Grad_Shafranov` / `GradShafranov` |
