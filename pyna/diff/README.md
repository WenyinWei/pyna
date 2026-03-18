# `pyna.diff` — Numerical Differentiation Helpers

## Overview

`pyna.diff` provides numerical differentiation utilities for dynamical
systems and field-line computations.

---

## Module reference

| Module | Contents |
|--------|----------|
| `diff.py` | Finite-difference Jacobian, gradient, and Hessian helpers |
| `fieldline.py` | `_FieldDifferentiableRZ` — finite-difference Jacobian of the field-line vector field in the (R, Z) plane |
| `fixedpoint.py` | Fixed-point Newton iteration with analytic or numerical Jacobian |
| `cycle.py` | Cycle-detection differentiation helpers |

---

## Fieldline Jacobian

The A-matrix used by FPT is computed from the Jacobian of the
normalised poloidal-to-toroidal field ratio:

```
A = ∂g/∂(R, Z),   g = [R·BR/Bphi, R·BZ/Bphi]
```

`_FieldDifferentiableRZ` wraps a `field_func` and computes this Jacobian
via central finite differences.

```python
from pyna.diff.fieldline import _FieldDifferentiableRZ

fdiff = _FieldDifferentiableRZ(field_func, eps=1e-5)
A = fdiff.A_matrix(R, Z, phi)
```
