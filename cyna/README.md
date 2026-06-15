# cyna — C++ Acceleration Layer for pyna

**cyna** is the C++ core of the [pyna](../pyna/) ecosystem, providing
high-performance implementations of the computationally intensive operations
that are bottlenecks in pure Python.

Production field-line and Poincare tracing should land here or behind the
`pyna._cyna`/`pyna.toroidal.flt` wrappers, not in downstream Python hot loops.
See [../FIELD_LINE_TRACING_POLICY.md](../FIELD_LINE_TRACING_POLICY.md).

## Trilingual array

```
pyna  (Python)  ←→  cyna  (C++)  ←→  Jyna / Jynamics.jl  (Julia)
```

All three share the same physical models and algorithms; only the
implementation language differs.  Use whichever tier fits your workflow:
- **pyna** — interactive analysis, Jupyter notebooks, rapid prototyping
- **cyna** — batch production runs, embedded deployments, maximum throughput
- **Jyna** — Julia-native pipelines that need AD or GPU support

## What cyna accelerates

cyna is *not* a mirror of the full pyna API.  It targets only the proven
bottlenecks:

| Operation | pyna module | cyna header |
|-----------|------------|-------------|
| Field-line tracing (RK4 / Ascent ODE) | `pyna.flt` | `cyna/flt.hpp` |
| Regular-grid interpolation | `pyna.coord` | `cyna/interpolate.hpp` |
| NumPy `.npz` I/O | `numpy` | `cyna/io.hpp` |
| FPT cycle shift `delta_X_pol`, `delta_X_cyc` | `pyna.topo.fpt` | `cyna/poincare.hpp` |

Everything else stays in Python — there is no benefit in rewriting it.

## Package structure

```
cyna/
├── xmake.lua                ← build system (xmake ≥ 2.8)
├── include/
│   ├── cyna/
│   │   ├── flt.hpp          ← Field-Line Tracing (RK4 + Ascent ODE)
│   │   ├── interpolate.hpp  ← RegularGridInterpolator
│   │   └── io.hpp           ← npz I/O helpers
│   └── BS_thread_pool.hpp   ← thread-pool (header-only, bundled)
├── app/
│   ├── flt3d.cpp            ← standalone FLT binary
│   └── construct_flux_coordinate.cpp
├── bindings/
│   └── flt_bindings.cpp     ← pybind11 Python bindings (stub)
└── tests/
    └── test_flt.cpp         ← C++ unit tests
```

The Python-facing glue lives in `pyna/_cyna/` (a subpackage of pyna).
It tries to import the compiled extension `_cyna_ext` and falls back
silently to pure Python if the extension is not installed.

## Building

### Prerequisites

- C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019+)
- [xmake](https://xmake.io) ≥ 2.8
- (optional) xtensor, pybind11

### Standalone C++ apps

```bash
cd cyna/
xmake build flt3d
xmake run flt3d
```

### Python extension

```bash
cd cyna/
xmake build cyna_python
# Copy the resulting _cyna_ext.* into pyna/_cyna/
xmake install -o ../pyna/_cyna/
```

Then from Python:

```python
from pyna._cyna import is_available, get_version
print(is_available())   # True if compiled extension is present
print(get_version())
```

## Python acceleration surface

The public Python wrappers live in `pyna._cyna`.  Magnetic-field component
arguments use the canonical cylindrical order:

```
BR, BZ, BPhi, R_grid, Z_grid, Phi_grid
```

Important exported functions:

- `trace_orbit_along_phi`: field-line trace with DPm samples.
- `compute_A_matrix_batch`: batch finite-difference Jacobian of
  `f=(R*BR/BPhi, R*BZ/BPhi)`.
- `progress_DX_pol_along_orbit`: tangent-map progression along any sampled
  orbit, returning `DX_pol(phi_e, phi0)` for every sampled `phi_e`;
  periodicity is not required.
- `progress_delta_X_along_orbit`: inhomogeneous FPT response with fixed
  source angle `phi_s`; only `phi_e` advances.  Use this for `delta_X_pol`.
- `evolve_delta_X_cycle_along_orbit`: semantic alias for the same
  inhomogeneous ODE when the supplied initial condition is already the closed
  periodic displacement `delta_X_cyc(phi0)`.  Use this when `phi_s` and
  `phi_e = phi_s + 2*pi*m` move together along a cycle.
- `compute_cycle_perturbation_shift`: integrates the FPT response equation
  and returns both `delta_X_pol` and periodic `delta_X_cyc`.

Naming convention for FPT APIs:

- `progress_*` means a fixed source section `phi_s` and a moving endpoint
  `phi_e`.  These functions produce open-orbit / endpoint responses such as
  `DX_pol(phi_s, phi_e)` or `delta_X_pol(phi_s, phi_e)`.
- `evolve_*_cycle_*` means the object is attached to a periodic orbit and is
  evaluated as the base phase `phi` moves; for cycle shift, `phi_s` and
  `phi_e = phi_s + 2*pi*m` move together.

Prefer the high-level wrapper `pyna.topo.fpt.compute_cycle_shift_from_cache`
in application code.  Use the `_cyna` functions only at bridge boundaries or
for low-level diagnostics.

## Design principles

1. **Header-only core** — `include/cyna/*.hpp` can be dropped into any C++
   project with no CMake / xmake required.
2. **Optional Python bindings** — pyna works without cyna.  If cyna is
   present it is used automatically; if not, pure-Python fallbacks run.
3. **No Python runtime dependency** — the C++ core never calls back into
   Python.  Bindings are one-way (C++ → Python).
4. **Minimal surface area** — expose only what's measurably slow in Python.

## License

Same as pyna — see [../LICENSE](../LICENSE).
