# `pyna.gc` — Guiding-Centre Motion

## Overview

`pyna.gc` implements equations of motion and utilities for guiding-centre
particle dynamics in combined electric and magnetic fields.

---

## Module reference

| Module | Contents |
|--------|----------|
| `EBField.py` | `EBField` — combined electric and magnetic field container |
| `electromagnetics.py` | Maxwell's equations helpers, field energy density |
| `motion.py` | Guiding-centre drift velocity equations (grad-B, curvature, E×B drifts) |
| `LoopBetweenCapacitor.py` | Inductive loop / capacitor circuit for RF field studies |

---

## Quick-start: guiding-centre drifts

```python
from pyna.gc.motion import ExB_drift, grad_B_drift, curvature_drift

vE  = ExB_drift(E_field, B_field, mass, charge)
vgB = grad_B_drift(B_field, mu, mass, charge)
vkB = curvature_drift(B_field, v_par, mass, charge)
```

---

## Coordinate convention

All guiding-centre computations use cylindrical coordinates `(R, Z, phi)`.
Magnetic field components are `(BR, BZ, Bphi)` in tesla.
Electric field components are `(ER, EZ, Ephi)` in V/m.
