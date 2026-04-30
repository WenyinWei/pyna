# `pyna.toroidal.coils` — Vacuum Coil Fields

## Overview

`pyna.toroidal.coils` provides:

- Abstract base classes for external vacuum magnetic fields (`CoilFieldVacuum`, `CoilFieldSuperposition`)
- Analytic field formulas for finite-length solenoids and current loops
- Biot-Savart integration for arbitrary filamentary coil sets (`Biot_Savart_field`)
- RMP (Resonant Magnetic Perturbation) spectral analysis

---

## Module reference

| Module | Key exports |
|--------|-------------|
| `base.py` | `CoilFieldVacuum` (abstract), `CoilFieldSuperposition`, `CoilFieldScaled` |
| `coil.py` | `BRBZ_induced_by_current_loop`, `BRBZ_induced_by_thick_finitelen_solenoid`, `CoilFieldAnalyticCircular`, `CoilFieldAnalyticRectangularSection` |
| `coil_system.py` | `CoilSet`, `Biot_Savart_field` |
| `RMP.py` | `normalize_b`, `RMP_spectrum_2d`, `island_width_at_rational_surfaces` |
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
`field.py` re-exports `VectorField3DCylindrical` etc. from `pyna.fields` directly.

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

## RMP spectrum

```python
from pyna.toroidal.coils.RMP import RMP_spectrum_2d, island_width_at_rational_surfaces

# Compute (m, n) spectrum of RMP field on flux surfaces
spectrum = RMP_spectrum_2d(equilibrium, coil_field_func, m_max=10, n_max=5)

# Estimate island widths at each rational surface
widths = island_width_at_rational_surfaces(equilibrium, spectrum)
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
