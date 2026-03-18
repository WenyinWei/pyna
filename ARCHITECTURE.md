# pyna Architecture

**pyna** is a Python library for dynamical-systems analysis and magnetic-confinement-fusion (MCF) plasma physics.
Its theoretical backbone is **Functional Perturbation Theory (FPT)** — a framework that
computes, analytically or semi-analytically, how geometric structures in phase space
(fixed points, invariant manifolds, flux surfaces) respond to small perturbations of the
underlying vector field.

---

## Top-level layout

```
pyna/                        ← pip-installable Python package
│
├── fields/                  ← Unified field class hierarchy  (§1)
├── system.py                ← Dynamical-system abstract base classes  (§2)
├── flt.py / flt_cuda.py     ← Field-line tracer (CPU / CUDA)  (§3)
├── flow.py / map.py         ← Continuous-time flow & discrete-map wrappers  (§4)
├── topo/                    ← Topological analysis (Poincaré, islands, manifolds)  (§5)
├── control/                 ← FPT-based real-time topology control  (§6)
├── MCF/                     ← Magnetic-confinement-fusion physics  (§7)
│   ├── equilibrium/         ← Axisymmetric & stellarator equilibria
│   ├── coils/               ← Vacuum field, Biot-Savart, RMP
│   ├── coords/              ← Flux-surface coordinate systems
│   ├── plasma_response/     ← Linear MHD plasma response
│   ├── control/             ← MCF-specific control (gap response, q-profile)
│   ├── diagnostics/         ← Connection-length, field-line endpoint diagnostics
│   ├── optimize/            ← Stellarator optimisation objectives
│   └── visual/              ← MCF-specific plotting helpers
├── coord/                   ← Backward-compat shim → MCF.coords
├── mag/                     ← Backward-compat shim → MCF.*
├── plasma_response/         ← Backward-compat shim → MCF.plasma_response
├── diff/                    ← Numerical differentiation helpers
├── draw/                    ← Generic geometry drawing (manifolds, resonances)
├── gc/                      ← Guiding-centre motion
├── interact/                ← Interactive matplotlib utilities
├── io/                      ← Poincaré orbit file I/O
├── utils/symutil/           ← SymPy helper routines
├── progress.py              ← Progress-reporting protocol
├── cache.py                 ← Lightweight disk-cache decorator
├── polynomial.py            ← 2-D polynomial type
├── polymap.py               ← Polynomial Poincaré map
├── withparam.py             ← Parametric/symbolic object mixin
├── sysutil.py               ← Utility functions for dynamical systems
├── vector_calc.py           ← Legacy vector calculus helpers
├── field_data.py            ← Legacy field-data storage
└── imas_compat.py           ← IMAS / OMAS data-dictionary adapter
```

The companion C++ acceleration layer lives in `cyna/` (sibling directory).
See `cyna/README.md` for build and usage instructions.

---

## §1  `pyna.fields` — Unified Field Hierarchy

All field-like objects in pyna descend from a single abstract tree.

```
Field  (abstract)
├── ScalarField  (range rank = 0)
│   ├── ScalarField1D / 2D / 3D / 4D
│   └── ScalarField3DCylindrical   ← concrete: (R,Z,φ) grid + interpolation
│       └── ScalarField3DAxiSymmetric  ← ∂/∂φ = 0
└── VectorField  (range rank = 1)
    ├── VectorField1D / 2D / 3D / 4D
    │   └── VectorField3D
    │       ├── VectorField3DCylindrical   ← concrete: (R,Z,φ) grid
    │       └── VectorField3DAxiSymmetric  ← no φ variation
    └── TensorField  (range rank ≥ 2)
        ├── TensorField3DRank2
        └── TensorField4DRank2
```

Differential operators (`gradient`, `divergence`, `curl`, `laplacian`) live in
`fields/diff_ops.py` and return new `Field` instances, propagating
`FieldProperty` flags (e.g. `DIVERGENCE_FREE`, `CURL_FREE`).

Coordinate metadata is attached via `fields/coords.py`:
`Coords3DCylindrical`, `Coords3DSpherical`, `Coords3DToroidal`, `Coords4D*`.

**Design rule:** `pyna.fields` is the sole canonical field hierarchy.
`pyna.MCF.coils.field` is a thin re-export layer only.

---

## §2  `pyna.system` — Dynamical System Abstractions

```
DynamicalSystem  (abstract)
├── NonAutonomousDynamicalSystem    ẋ = f(x, t)
└── AutonomousDynamicalSystem       ẋ = f(x)
    └── VectorField                 a VectorField IS a DynamicalSystem
        ├── VectorField1D / 2D / 3D / 4D
        └── VectorField3DAxiSymmetric
```

Key contracts: `state_dim` (int), `__call__(coords)` → velocity.

`system.py` also defines the `_LegacyVectorField3D` shim for any code that
still subclasses the old name.

---

## §3  `pyna.flt` — Field-Line Tracer

`FieldLineTracer` integrates the ODE

```
dR/dφ = R · BR / Bφ,   dZ/dφ = R · BZ / Bφ
```

using SciPy `solve_ivp` (RK45 or DOP853) with optional parallel execution
via `ThreadPoolExecutor` (default) or `ProcessPoolExecutor`.

```python
tracer = FieldLineTracer(field_func, method='RK45')
trajectory = tracer.trace(x0, t_span=(0, 2*np.pi*20))
trajectories = tracer.trace_many(start_pts, t_max=100.0, progress=TqdmProgress())
```

The `get_backend(mode)` factory selects:
- `'cpu'`   → `FieldLineTracer` (pure Python / SciPy)
- `'cuda'`  → `FieldLineTracerCUDA` (CuPy, see `flt_cuda.py`)
- `'opencl'`→ reserved (raises `NotImplementedError`)

The legacy `bundle_tracing_with_t_as_DeltaPhi(...)` function is fully preserved.

---

## §4  `pyna.flow` and `pyna.map`

| Module | Class | Semantics |
|--------|-------|-----------|
| `flow.py` | `Flow`, `FlowSympy`, `FlowCallable` | Continuous-time solution Φ(t; x₀) |
| `map.py`  | `Map`, `MapSympy`, `MapCallable`, `MapSympyComposite` | Discrete iterate P(x₀) |

Both share a `WithParam` mixin (`withparam.py`) for parametric/symbolic objects.

**Terminology** (per STYLE.md §10):
- *trajectory*: solution curve of an ODE / continuous flow.
- *orbit*: iterates of a Poincaré map or any discrete system.

---

## §5  `pyna.topo` — Topological Analysis

```
topo/
├── poincare.py            Section (abstract), ToroidalSection, PoincareMap
├── fixed_points.py        poincare_map(), find_periodic_orbit(), classify_fixed_point()
├── island.py              Island, IslandChain dataclasses
├── monodromy.py           MonodromyAnalysis: eigenvalues, stability_index, Greene_residue
├── variational.py         PoincareMapVariationalEquations, tangent_map()
├── manifold_improve.py    StableManifold, UnstableManifold extraction
├── manifold.py            grow_manifold_from_Xcycle()
├── topology_analysis.py   analyse_topology(), TopologyReport
├── island_extract.py      detect and measure island chains
├── cycle.py               cycle detection utilities
├── classical_maps.py      HenonMap, StandardMap (test cases / benchmarks)
└── chaos.py               ftle_field(), chirikov_overlap(), chaotic_boundary_estimate()
```

### Poincaré map Jacobian naming (STYLE.md §2)

| Symbol | Meaning |
|--------|---------|
| `DX`   | Orbital Jacobian of the continuous flow |
| `DP`   | Poincaré map Jacobian (one section crossing) |
| `DPm`  | Monodromy matrix after m full turns |

---

## §6  `pyna.control` — FPT-Based Topology Control

`pyna.control` implements **Functional Perturbation Theory** for real-time
multi-objective control of magnetic topology.  It is independent of fusion
details — the same API works for any area-preserving 2-D system.

```
control/
├── fpt.py               Core FPT: A_matrix, DPm_axisymmetric, cycle_shift,
│                        DPm_change, delta_A_total, manifold_shift,
│                        flux_surface_deformation
├── topology_state.py    TopologyState, XPointState, OPointState, SurfaceFate,
│                        compute_topology_state()
├── response_matrix.py   build_response_matrix(), build_full_response_matrix()
├── optimizer.py         ControlWeights, ControlConstraints, TopologyController
├── surface_fate.py      greene_residue(), classify_surface_fate()
├── _cache.py            Internal caching utilities
└── _cached_fpt.py       CachedFPTAnalyzer — high-level cached FPT workflow
```

See `pyna/control/README.md` for theory background and usage examples.

---

## §7  `pyna.MCF` — Magnetic Confinement Fusion

MCF-specific physics is grouped under `pyna.MCF`.

### `MCF/equilibrium/`

| Module | Key classes / functions |
|--------|------------------------|
| `axisymmetric.py` | `EquilibriumAxisym` (abstract), `EquilibriumTokamakCircularSynthetic` |
| `Solovev.py` | `EquilibriumSolovev` — analytic Solov'ev equilibrium |
| `GradShafranov.py` | `recover_pressure_simplest`, `solve_GS_perturbed` |
| `stellarator.py` | `StellaratorSimple`, `simple_stellarator` factory |
| `feedback_boozer.py` | `BoozerSurface`, `BoozerPerturbation`, `MHD_response_operator`, `compute_boozer_response` |
| `feedback_cylindrical.py` | `CylindricalGrid`, `PerturbationField`, `PlasmaResponse`, `compute_plasma_response`, `feedback_correction_field`, `iterative_equilibrium_correction` |

### `MCF/coils/`

| Module | Contents |
|--------|----------|
| `base.py` | `CoilFieldVacuum` (abstract), `CoilFieldSuperposition`, `CoilFieldScaled` |
| `coil.py` | `BRBZ_induced_by_current_loop`, `BRBZ_induced_by_thick_finitelen_solenoid` |
| `coil_system.py` | `CoilSet`, `Biot_Savart_field` |
| `RMP.py` | `normalize_b`, `RMP_spectrum_2d`, `island_width_at_rational_surfaces` |
| `vector_potential.py` | Vector potential computation for coil fields |
| `field.py` | Thin canonical re-export layer |

### `MCF/coords/`

Flux-surface coordinate systems (PEST, Boozer, Hamada, EqualArc).
See `MCF/coords/` for details.

### `MCF/plasma_response/`

Linear MHD plasma response via perturbed Grad-Shafranov equation
(`PerturbGS.py`).

### `MCF/control/`

MCF-specific control modules (gap response, island control, q-profile response).

### `MCF/diagnostics/`

`field_line_length`, `field_line_endpoints`, `field_line_min_psi`.

### `MCF/optimize/`

Stellarator optimisation objectives:
`neoclassical_epsilon_eff`, `xpoint_field_parallelism`, `magnetic_axis_position`,
`wall_clearance`, `compute_all_objectives`.

### `MCF/visual/`

MCF-specific plotting: RMP spectrum, equilibrium cross-sections, tokamak manifolds.

---

## §8  Backward-Compatibility Shims

Three top-level packages are thin re-export layers for legacy code.
**Do not add new code here.**

| Shim | Points to |
|------|-----------|
| `pyna.mag` | `pyna.MCF.*` |
| `pyna.coord` | `pyna.MCF.coords` |
| `pyna.plasma_response` | `pyna.MCF.plasma_response` |

---

## §9  Optional C++ Acceleration (`_cyna`)

`pyna._cyna` tries to import a compiled C++ extension `_cyna_ext` at startup.
If unavailable, all calls fall back silently to pure Python.

```python
from pyna._cyna import is_available, get_version
print(is_available())   # True when compiled extension is present
```

The C++ source lives in the sibling `cyna/` directory.
See `cyna/README.md` for build instructions.

---

## §10  Naming Conventions (Summary)

See `STYLE.md` for the full style guide.  Quick reference:

| Category | Rule | Example |
|----------|------|---------|
| Proper nouns / acronyms | Keep capitalisation | `Biot_Savart_field`, `RMP_coils`, `MHD_response_operator` |
| Poincaré map Jacobian | `DP` / `DPm` | not `J`, not `M` |
| Orbital Jacobian | `DX` | not `J` |
| Class names | Noun first, qualifier last | `VectorField3DCylindrical`, `EquilibriumSolovev` |
| Coordinates | `R, Z, phi` (cylindrical); `psi_B, theta_B, phi_B` (Boozer) | |
| Continuous solutions | *trajectory* | `trajectory_RZPhi` |
| Discrete iterates | *orbit* | `orbit_RZ` |

---

## §11  Testing

Tests live in `tests/`.  Run with:

```bash
pytest tests/
```

Key test modules:

| Test file | Covers |
|-----------|--------|
| `test_fields.py` | `pyna.fields` hierarchy |
| `test_system.py` | `pyna.system` |
| `test_flt.py` | Field-line tracer |
| `test_fpt.py`, `test_fpt_validation.py` | FPT formulae |
| `test_fixed_points.py` | Poincaré map fixed points |
| `test_topo_island.py`, `test_island_extract.py` | Island analysis |
| `test_variational.py` | Tangent-map / variational equations |
| `test_solovev.py` | Solov'ev equilibrium |
| `test_mag_field.py`, `test_mag_coil.py` | Magnetic field & coils |
| `test_coordinates.py`, `test_mag_coordinate.py` | Coordinate systems |
| `test_classical_maps.py` | Hénon / Standard map |
| `test_gap_response.py` | FPT gap-response matrix |
