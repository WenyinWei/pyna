# `pyna.utils` — Utility Subpackages

## Overview

`pyna.utils` groups lightweight utility modules that do not belong in any
domain-specific subpackage.

---

## `pyna.utils.symutil` — SymPy Helpers

Symbolic math utilities for constructing and manipulating SymPy expressions
in the context of dynamical systems and plasma physics.

| Module | Contents |
|--------|----------|
| `basics.py` | Symbol creation, simplification, substitution helpers |
| `op.py` | Symbolic differential operators (gradient, divergence, curl in cylindrical coords) |
| `characteristics.py` | Method of characteristics for first-order PDEs |
| `vector.py` | Symbolic vector operations and coordinate transforms |

### Example

```python
from pyna.utils.symutil.basics import make_symbols
from pyna.utils.symutil.op import symbolic_curl_cylindrical

R, Z, phi = make_symbols('R Z phi')
AR, AZ, Aphi = make_symbols('AR AZ Aphi')
BR, BZ, Bphi = symbolic_curl_cylindrical(AR, AZ, Aphi, R, Z, phi)
```
