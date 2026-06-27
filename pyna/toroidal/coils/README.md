# `pyna.toroidal.coils` — Vacuum Coil Fields

## Overview

`pyna.toroidal.coils` provides:

- Abstract base classes for external vacuum magnetic fields (`CoilFieldVacuum`, `CoilFieldSuperposition`)
- Analytic field formulas for finite-length solenoids and current loops
- Biot-Savart integration for arbitrary filamentary coil sets (`Biot_Savart_field`)
- Vacuum coil fields.  Magnetic-spectrum island-chain analysis lives in
  `pyna.toroidal.perturbation_spectrum`.

---

## Module reference

| Module | Key exports |
|--------|-------------|
| `base.py` | `CoilFieldVacuum` (abstract), `CoilFieldSuperposition`, `CoilFieldScaled` |
| `coil.py` | `BRBZ_induced_by_current_loop`, `BRBZ_induced_by_thick_finitelen_solenoid`, `CoilFieldAnalyticCircular`, `CoilFieldAnalyticRectangularSection` |
| `coil_system.py` | `CoilSet`, `Biot_Savart_field` |
| `vector_potential.py` | `vector_potential_axisymmetric` for ψ reconstruction |
| `field.py` | Thin re-export layer → `pyna.fields` |

---

## Class hierarchy

```
CoilFieldVacuum  (abstract, base.py)
│   Interface: field_func(rzphi) → (dR/dl, dZ/dl, dphi/dl)
│              B_field(R, Z, phi) → (BR, BZ, Bphi)
│
├── CoilFieldSuperposition     weighted sum of CoilFieldVacuum instances
├── CoilFieldScaled            single coil × scalar factor
└── (concrete subclasses in MCF — extend CoilFieldVacuum)
```

**Design rule:** `pyna.fields` is the canonical field hierarchy.
`CoilFieldVacuum` is a physics interface (exposes `field_func`) layered on top.
`field.py` re-exports `VectorFieldCylind` etc. from `pyna.fields` directly.

---

## Biot-Savart field

```python
from pyna.toroidal.coils.coil_system import CoilSet, Biot_Savart_field

coils = CoilSet(...)
BR, BZ, Bphi = Biot_Savart_field(
    coil_pts,        # ndarray (N, 3) — [R, Z, phi] filament points
    coil_current,    # float, amperes
    R_grid, Z_grid, Phi_grid,
)
```

**Naming rule:** `Biot_Savart_field` — capitals per STYLE.md §1 (Biot and Savart
are proper names).

---

## Magnetic perturbation spectrum

```python
from pyna import toroidal

tilde_b1 = toroidal.nardon_radial_perturbation(
    R_surf, Z_surf, phi_vals, theta_vals,
    delta_BR, delta_BZ, delta_Bphi, radial_labels,
    denominator_B_phi=B0_phi,
)
spectrum = toroidal.radial_perturbation_Fourier_spectrum(
    tilde_b1, theta_vals, phi_vals, radial_labels=radial_labels,
)
chains = toroidal.analyze_resonant_island_chains(spectrum, q_profile, n=2)
```

---

## Superposition of coils

```python
from pyna.toroidal.coils.base import CoilFieldSuperposition, CoilFieldScaled

combined = CoilFieldSuperposition([
    CoilFieldScaled(coil_efc, current=3e3),
    CoilFieldScaled(coil_ic,  current=-1e3),
])
BR, BZ, Bphi = combined.B_field(R, Z, phi)
```
