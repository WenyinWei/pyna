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

| Module | Key exports |
|--------|-------------|
| `axisymmetric.py` | `EquilibriumAxisym` (abstract), `EquilibriumTokamakCircularSynthetic` |
| `Solovev.py` | `EquilibriumSolovev` — analytic Solov'ev solution |
| `GradShafranov.py` | `recover_pressure_simplest`, `solve_GS_perturbed` |
| `stellarator.py` | `StellaratorSimple`, `simple_stellarator` factory |
| `feedback_boozer.py` | `BoozerSurface`, `BoozerPerturbation`, `MHD_response_operator`, `compute_boozer_response`, `island_width_with_response` |
| `feedback_cylindrical.py` | `CylindricalGrid`, `PerturbationField`, `PlasmaResponse`, `compute_plasma_response`, `feedback_correction_field`, `iterative_equilibrium_correction` |

`feedback_boozer.py` is valid only where flux surfaces exist (non-chaotic regions).
`feedback_cylindrical.py` works everywhere, including chaotic and divertor regions.

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

MCF-specific control modules:
- `wall.py` — `WallGeometry` with gap monitoring points
- `gap_response.py` — `gap_response_matrix_fpt`, `grow_stable_manifold_cached`
- `island_control.py` — island width control primitives
- `qprofile_response.py` — q-profile response matrix

For the generic FPT-based topology controller see `pyna.control`.

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
