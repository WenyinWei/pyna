# `pyna.toroidal.equilibrium` — MHD Equilibria

## Overview

This subpackage provides axisymmetric and stellarator MHD equilibria,
as well as linear-response models for the effect of small external perturbations.

---

## Package status

`pyna.MCF.equilibrium` is now a **package-level facade only**.
The old leaf wrappers such as `axisymmetric.py`, `Solovev.py`,
`GradShafranov.py`, `stellarator.py`, `feedback_boozer.py`,
`feedback_cylindrical.py`, `feedback_cylindrical_utils.py`,
`fenicsx_corrector.py`, and `finite_beta_perturbation.py` were removed because
they no longer carried independent implementation.

Use either:

- `pyna.toroidal.equilibrium` for all new code, or
- `pyna.MCF.equilibrium` package-root imports while migrating old notebooks.

Representative package-root exports include:

- `EquilibriumAxisym`, `EquilibriumTokamakCircularSynthetic`
- `EquilibriumSolovev`
- `solve_GS_perturbed`, `recover_pressure_simplest`
- `StellaratorSimple`, `simple_stellarator`
- `BoozerSurface`, `BoozerPerturbation`, `compute_boozer_response`
- `CylindricalGrid`, `PerturbationField`, `PlasmaResponse`,
  `compute_plasma_response`, `feedback_correction_field`,
  `iterative_equilibrium_correction`
- FEniCS-x helpers such as `AndersonMixer`, `build_rz_mesh`,
  `solve_force_balance_correction`, `fpt_fenicsx_beta_step`
- cylindrical utility helpers such as `greens_function_cylinder`,
  `lundquist_number`, `toroidal_fft`, `convergence_monitor`

---

## Equilibrium interface

All concrete equilibria expose:

```python
eq.R0           # float, major radius [m]
eq.r0           # float, minor radius [m]
eq.B0           # float, toroidal field at axis [T]
eq.q_of_psi(psi_norm)   # safety factor profile
eq.field_func           # callable [R,Z,phi] → [dR/dl, dZ/dl, dphi/dl]
eq.psi(R, Z)            # poloidal flux ψ(R,Z)
eq.magnetic_axis        # (R_axis, Z_axis)
```

---

## Solov'ev equilibrium

```python
from pyna.toroidal.equilibrium.Solovev import EquilibriumSolovev

eq = EquilibriumSolovev(R0=1.65, a=0.5, B0=1.0, q0=2.0, kappa=1.6, delta=0.3)
psi_val = eq.psi(1.7, 0.1)
q_val   = eq.q_of_psi(0.5)
```

---

## Boozer plasma response

The Boozer formulation is fast (response diagonal in Fourier (m,n) space)
but valid only where flux surfaces exist.

```python
from pyna.toroidal.equilibrium.feedback_boozer import (
    BoozerPerturbation, compute_boozer_response, island_width_with_response,
)

# Convert external perturbation to Boozer modes
pert_boozer = BoozerPerturbation.from_cylindrical_perturbation(
    cylindrical_pert, equilibrium, psi_grid=np.linspace(0.1, 0.9, 20),
)

# Apply plasma response
total_field = compute_boozer_response(
    equilibrium, pert_boozer, model='resistive', lundquist=1e6,
)

# Island-width comparison
result = island_width_with_response(equilibrium, (3, 1), pert_boozer, total_field)
print(result['amplification'])
```

---

## Cylindrical plasma response

For chaotic regions and divertor studies:

```python
from pyna.toroidal.equilibrium.feedback_cylindrical import (
    CylindricalGrid, PerturbationField, compute_plasma_response,
    iterative_equilibrium_correction,
)

grid = CylindricalGrid.uniform(1.0, 2.5, -1.2, 1.2, NR=32, NZ=32, Nphi=16)
pert = PerturbationField.from_callable(grid, my_rmp_field)

# Single-pass plasma response
response = compute_plasma_response(equilibrium, pert, model='ideal_mhd')

# Iterative self-consistent correction
final_pert, info = iterative_equilibrium_correction(
    equilibrium, pert, n_iterations=5, convergence_tol=1e-4,
)
print("converged:", info['converged'], "in", info['n_iter'], "iterations")
```
