# `pyna.fields` — Unified Field Hierarchy

## Overview

`pyna.fields` is the **sole canonical hierarchy** for all field-like objects in pyna.
Every class that represents a scalar, vector, or tensor function of space must descend
from the abstract base classes defined here.

---

## Class tree

```
Field  (abstract — fields/base.py)
│   Properties: domain_dim, range_rank, properties (FieldProperty), name, units
│   Interface : __call__(coords) → ndarray
│
├── ScalarField  (range_rank = 0)
│   ├── ScalarField1D / 2D / 3D / 4D   (abstract, domain_dim specialised)
│   └── ScalarFieldCylind        ← concrete implementation
│       └── ScalarFieldCylindAxisym   ← ∂/∂φ = 0 specialisation
│
├── VectorField  (range_rank = 1)
│   ├── VectorField1D / 2D / 3D / 4D   (abstract)
│   │   └── VectorField3D
│   │       ├── VectorFieldCylind    ← concrete: (R,Z,φ) grid
│   │       └── VectorFieldCylindAxisym  ← no φ variation
│   └── (VectorField4D)
│
└── TensorField  (range_rank ≥ 2)
    ├── Tensor2FieldCylind   ← rank-2 tensor on (R,Z,φ) domain
    └── TensorField4DRank2
```

---

## File reference

| File | Contents |
|------|----------|
| `base.py` | Abstract `Field`, `ScalarField`, `VectorField`, `TensorField` and their 1-D–4-D specialisations |
| `cylindrical.py` | Concrete grid-based implementations: `ScalarFieldCylind`, `VectorFieldCylind`, `VectorFieldCylindAxisym`, `ScalarFieldCylindAxisym` |
| `toroidal.py` | Fixed-section toroidal compatibility helpers: `VectorFieldCylind`, `VectorFieldCylindAxisym`, `Equilibrium`, `compute_J_by_curl` |
| `coords.py` | Coordinate-system metadata: `Coords3DCylindrical`, `Coords3DSpherical`, `Coords3DToroidal`, `Coords4D*` |
| `properties.py` | `FieldProperty` flag enum: `DIVERGENCE_FREE`, `CURL_FREE`, `HARMONIC`, etc. |
| `diff_ops.py` | Differential operators returning `Field` instances: `gradient`, `divergence`, `curl`, `laplacian` |
| `tensor.py` | Rank-2 tensor field helpers |

---

## Quick-start

```python
from pyna.fields import VectorFieldCylind, ScalarFieldCylind

# Construct a magnetic field on a (R, Z, φ) grid
B = VectorFieldCylind(
    R=R_arr, Z=Z_arr, Phi=Phi_arr,
    VR=VR_data, VZ=VZ_data, VPhi=VPhi_data,
    name='B_field', units='T',
)
# Constructor parameters use generic vector names (VR, VZ, VPhi);
# magnetic aliases (BR, BZ, BPhi) are available as properties.

# Evaluate at a single point
RZPhi = [1.7, 0.0, 0.0]
bR, bZ, bPhi = B(RZPhi)

# Batch evaluation
pts = np.column_stack([R_pts, Z_pts, Phi_pts])  # shape (N, 3)
field_vals = B(pts)  # shape (N, 3)
```

---

## Design rules

1. **Single canonical hierarchy** — do not create parallel field classes
   elsewhere.  Extend `VectorFieldCylind` or add a new concrete subclass.
2. **Compatibility helpers live here** — legacy imports such as
   `pyna.topo.field.VectorFieldCylind` must re-export `pyna.fields` classes.
3. **No new backward-compat aliases** — rename and update all call sites immediately.
4. **Differential operators propagate `FieldProperty`** — e.g. `curl` of a
   `CURL_FREE` field raises `ValueError` before computing.
