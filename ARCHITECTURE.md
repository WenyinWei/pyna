# pyna Architecture

**pyna** is a Python library for dynamical-systems analysis and magnetic-confinement-fusion (MCF) plasma physics.
Its theoretical backbone is **Functional Perturbation Theory (FPT)** — a framework that
computes, analytically or semi-analytically, how geometric structures in phase space
(fixed points, invariant manifolds, flux surfaces) respond to small perturbations of the
underlying vector field.

---

## ⚠️ Fundamental Principle: All Geometric Objects are 3D

> **All geometric objects (LCFS, flux surfaces, X-points, O-points, manifolds) are
> 3D objects. Their appearance at different section planes must be computed as
> intersections of the 3D object with the section plane, using a SINGLE
> multi-section field-line integration pass. NEVER compute geometric objects
> independently at each section.**

When tracing Poincaré maps for stellarators and non-axisymmetric configurations:

1. **Find one seed** at a reference section via binary search or Poincaré-survival.
2. **Trace once** using a multi-section recorder for N turns.
3. **Crossings at each section** are self-consistent 2D slices of the same 3D object.
4. **Never re-seed independently** at each section — that produces geometrically
   inconsistent objects.

---�?a framework that
computes, analytically or semi-analytically, how geometric structures in phase space
(fixed points, invariant manifolds, flux surfaces) respond to small perturbations of the
underlying vector field.

---

## Top-level layout

```
pyna/                        �?pip-installable Python package
�?
├── fields/                  �?Unified field class hierarchy  (§1)
├── system.py                �?Dynamical-system abstract base classes  (§2)
├── flt.py / flt_cuda.py     �?Field-line tracer (CPU / CUDA)  (§3)
├── flow.py / map.py         �?Continuous-time flow & discrete-map wrappers  (§4)
├── topo/                    �?Topological analysis (Poincaré, islands, manifolds)  (§5)
├── control/                 �?FPT-based real-time topology control  (§6)
├── toroidal/                �?Canonical toroidal / MHD physics namespace  (§7)
│   ├── perturbation/        �?Toroidal perturbative equilibrium / plasma-response landing zone
│
│   Note: the historical ``pyna.MCF`` package tree has been removed.
│   Toroidal / magnetic-specific code now lives directly under ``pyna.toroidal``.
├── diff/                    �?Numerical differentiation helpers
├── draw/                    �?Generic geometry drawing (manifolds, resonances)
├── gc/                      �?Guiding-centre motion
├── interact/                �?Interactive matplotlib utilities
├── io/                      �?Poincaré orbit file I/O
├── utils/symutil/           �?SymPy helper routines
├── progress.py              �?Progress-reporting protocol
├── cache.py                 �?Lightweight disk-cache decorator
├── polynomial.py            �?2-D polynomial type
├── polymap.py               �?Polynomial Poincaré map
├── withparam.py             �?Parametric/symbolic object mixin
├── sysutil.py               �?Utility functions for dynamical systems
├── vector_calc.py           �?Legacy vector calculus helpers
├── field_data.py            �?Legacy field-data storage
└── imas_compat.py           �?IMAS / OMAS data-dictionary adapter
```

The companion C++ acceleration layer lives in `cyna/` (sibling directory).
See `cyna/README.md` for build and usage instructions.

---

## §1  `pyna.fields` �?Unified Field Hierarchy

All field-like objects in pyna descend from a single abstract tree.

```
Field  (abstract)
├── ScalarField  (range rank = 0)
�?  ├── ScalarField1D / 2D / 3D / 4D
�?  └── ScalarField3DCylindrical   �?concrete: (R,Z,φ) grid + interpolation
�?      └── ScalarField3DAxiSymmetric  �?�?∂�?= 0
└── VectorField  (range rank = 1)
    ├── VectorField1D / 2D / 3D / 4D
    �?  └── VectorField3D
    �?      ├── VectorField3DCylindrical   �?concrete: (R,Z,φ) grid
    �?      └── VectorField3DAxiSymmetric  �?no φ variation
    └── TensorField  (range rank �?2)
        ├── TensorField3DRank2
        └── TensorField4DRank2
```

Differential operators (`gradient`, `divergence`, `curl`, `laplacian`) live in
`fields/diff_ops.py` and return new `Field` instances, propagating
`FieldProperty` flags (e.g. `DIVERGENCE_FREE`, `CURL_FREE`).

Coordinate metadata is attached via `fields/coords.py`:
`Coords3DCylindrical`, `Coords3DSpherical`, `Coords3DToroidal`, `Coords4D*`.

**Design rule:** `pyna.fields` is the sole canonical field hierarchy.

---

## §2  `pyna.system` �?Dynamical System Abstractions

```
DynamicalSystem  (abstract)
├── NonAutonomousDynamicalSystem    �?= f(x, t)
└── AutonomousDynamicalSystem       �?= f(x)
    └── VectorField                 a VectorField IS a DynamicalSystem
        ├── VectorField1D / 2D / 3D / 4D
        └── VectorField3DAxiSymmetric
```

Key contracts: `state_dim` (int), `__call__(coords)` �?velocity.

`system.py` also defines the `_LegacyVectorField3D` shim for any code that
still subclasses the old name.

---

## §3  `pyna.flt` �?Field-Line Tracer

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
- `'cpu'`   �?`FieldLineTracer` (pure Python / SciPy)
- `'cuda'`  �?`FieldLineTracerCUDA` (CuPy, see `flt_cuda.py`)
- `'opencl'`�?reserved (raises `NotImplementedError`)

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
├── _base.py               InvariantSet (root ABC), InvariantManifold, SectionCuttable
├── invariants.py           FixedPoint, PeriodicOrbit, Cycle, Island, IslandChain,
│                           Tube, TubeChain, StableManifold, UnstableManifold
├── island.py              Island, IslandChain (MCF-enriched versions)
├── tube.py                Tube, TubeChain (MCF-enriched: trajectory, section-view bridge)
├── poincare.py            Section (abstract), ToroidalSection, PoincareMap
├── fixed_points.py        poincare_map(), find_periodic_orbit(), classify_fixed_point()
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

### Invariant-object hierarchy

```
InvariantSet (root ABC)                     [_base.py]
├── InvariantManifold (adds intrinsic_dim)  [_base.py]
│   ├── FixedPoint          — one point of a map periodic orbit (intrinsic_dim=0)
│   ├── PeriodicOrbit       — period-m orbit of a discrete map (intrinsic_dim=0)
│   ├── Cycle               — closed orbit of a continuous flow (intrinsic_dim=1)
│   ├── InvariantTorus      — KAM torus (intrinsic_dim=2)
│   ├── StableManifold      — stable manifold of a hyperbolic cycle
│   └── UnstableManifold    — unstable manifold of a hyperbolic cycle
│
├── Island                  — region around an elliptic PeriodicOrbit
│     O_orbit: PeriodicOrbit        (elliptic orbit at centre)
│     X_orbits: List[PeriodicOrbit] (hyperbolic orbits at separatrix)
│     O_point, X_points             (convenience properties: first orbit point)
│
├── IslandChain             — all islands of one resonance m:n
│     islands: List[Island]
│     winding: Tuple[int, ...]
│     is_connected, orbit_groups    (W7X 5/5: gcd=5 → 5 disconnected sub-groups)
│
├── Tube                    — continuous-time resonance (holds Cycles)
│     O_cycle: Cycle, X_cycles: List[Cycle]
│     section_cut(phi) → List[Island]
│
└── TubeChain               — all Tubes of one resonance
      tubes: List[Tube]
      section_cut(phi) → IslandChain
```

### Design principle: Maps are first-class dynamical systems

**A discrete map is an autonomous dynamical system**, not a subordinate
"section object" of a flow.  `Island` and `IslandChain` are invariant
structures of a map — whether that map is a standalone discrete system
(e.g. standard map, Hénon map) or a Poincaré return map of a continuous
flow.  There is no `SectionObject` base class; map-level objects inherit
directly from `InvariantSet`.

**Continuous-time ↔ discrete-time bridge:**
- `Tube` holds Cycles (continuous flow orbits).
- `Tube.section_cut(phi)` produces Islands (discrete map structures).
- This is a projection operation, not a class hierarchy relationship.

**FixedPoint is a single-point convenience**, not a fundamental concept.
A fixed point is one point of a periodic orbit where P^m(x₀) = x₀.  Use
`PeriodicOrbit` to represent the full period-m orbit.  `FixedPoint.as_orbit()`
wraps a single FixedPoint into a period-1 PeriodicOrbit.

**IslandChain connectivity:**
Not all islands in a chain are connected by single map iterations.
`gcd(m, n)` gives the number of independent orbits.  Example:
W7X 5/5 → `gcd(5,5) = 5` independent orbits, each island disconnected.

### Poincaré map Jacobian naming (STYLE.md §2)

| Symbol | Meaning |
|--------|---------|
| `DX`   | Orbital Jacobian of the continuous flow |
| `DP`   | Poincaré map Jacobian (one section crossing) |
| `DPm`  | Monodromy matrix after m full turns |

---

## §6  `pyna.control` �?FPT-Based Topology Control

`pyna.control` implements **Functional Perturbation Theory** for real-time
multi-objective control of magnetic topology.  It is independent of fusion
details �?the same API works for any area-preserving 2-D system.

```
control/
├── fpt.py               Core FPT: A_matrix, DPm_axisymmetric, cycle_shift,
�?                       DPm_change, delta_A_total, manifold_shift,
�?                       flux_surface_deformation
├── topology_state.py    TopologyState, XPointState, OPointState, SurfaceFate,
�?                       compute_topology_state()
├── response_matrix.py   build_response_matrix(), build_full_response_matrix()
├── optimizer.py         ControlWeights, ControlConstraints, TopologyController
├── surface_fate.py      Greene_residue(), classify_surface_fate()
├── _cache.py            Internal caching utilities
└── _cached_fpt.py       CachedFPTAnalyzer �?high-level cached FPT workflow
```

See `pyna/control/README.md` for theory background and usage examples.

---

## §7  `pyna.toroidal`

Toroidal / magnetic-geometry helpers now live under `pyna.toroidal`. The
legacy `pyna.MCF` compatibility tree has been removed.

### `toroidal/equilibrium/`

| Module | Key classes / functions |
|--------|------------------------|
| `axisymmetric.py` | `EquilibriumAxisym` (abstract), `EquilibriumTokamakCircularSynthetic` |
| `Solovev.py` | `EquilibriumSolovev` �?analytic Solov'ev equilibrium |
| `GradShafranov.py` | `recover_pressure_simplest`, `solve_GS_perturbed` |
| `stellarator.py` | `StellaratorSimple`, `simple_stellarator` factory |
| `feedback_boozer.py` | `BoozerSurface`, `BoozerPerturbation`, `MHD_response_operator`, `compute_boozer_response` |
| `feedback_cylindrical.py` | `CylindricalGrid`, `PerturbationField`, `PlasmaResponse`, `compute_plasma_response`, `feedback_correction_field`, `iterative_equilibrium_correction` |

### `toroidal/coils/`

| Module | Contents |
|--------|----------|
| `base.py` | `CoilFieldVacuum` (abstract), `CoilFieldSuperposition`, `CoilFieldScaled` |
| `coil.py` | `BRBZ_induced_by_current_loop`, `BRBZ_induced_by_thick_finitelen_solenoid` |
| `coil_system.py` | `CoilSet`, `Biot_Savart_field` |
| `RMP.py` | `normalize_b`, `RMP_spectrum_2d`, `island_width_at_rational_surfaces` |
| `vector_potential.py` | Vector potential computation for coil fields |
| `field.py` | Thin canonical re-export layer |

### `toroidal/coords/`

Flux-surface coordinate systems (PEST, Boozer, Hamada, EqualArc).
See `toroidal/coords/` for details.

### `toroidal/perturbation/`

Canonical architectural landing zone for **toroidal perturbative theory**.
Use this namespace sparingly for toroidal perturbative helpers that remain in
repo for now; plasma-behaviour analysis should ultimately move out to topoquest.

Current sub-buckets:
- `perturbation.equilibrium` → finite-β continuation, perturbed
  Grad-Shafranov, force-balance correction workflows.
- `perturbation.response` → plasma-response closures, coupled GS / MHD
  response solvers, vacuum→plasma response operators.

**Boundary rule:**
- `pyna.control` = generic, dynamical-systems FPT and control
- `pyna.toroidal.perturbation` = toroidal / MHD perturbative theory

### `toroidal/plasma_response/`

Legacy in-repo plasma-response implementation kept only until the planned
move of plasma-behaviour workflows to topoquest. Do not treat this as a
long-term public pyna namespace.

### `toroidal/control/` (legacy `MCF/control/`)

Toroidal control modules (gap response, island control, q-profile response).

### `toroidal/diagnostics/` (legacy `MCF/diagnostics/`)

`field_line_length`, `field_line_endpoints`, `field_line_min_psi`.

### `toroidal/optimize/` (legacy `MCF/optimize/`)

Stellarator optimisation objectives:
`neoclassical_epsilon_eff`, `xpoint_field_parallelism`, `magnetic_axis_position`,
`wall_clearance`, `compute_all_objectives`.

### `toroidal/visual/` (legacy `MCF/visual/`)

Toroidal plotting: RMP spectrum, equilibrium cross-sections, tokamak manifolds.

---

## §8  Backward-Compatibility Shims

Three top-level packages are thin re-export layers for legacy code.
**Do not add new code here.**

| Shim | Points to |
|------|-----------|
| `pyna.mag` | `pyna.toroidal.*` |
| `pyna.coord` | `pyna.toroidal.coords` |

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

---

## 3D Magnetic Object Principle

All geometric objects in 3D magnetic topology (field lines, flux surfaces, LCFS,
manifolds, X/O-point orbits) are fundamentally 3D objects.

Design rule: ALWAYS represent objects as 3D trajectories first, then derive
2D cross-sections by intersection. NEVER independently compute a geometric
object at each phi section.

The canonical representation is ToroidalTrajectory, which stores the full
(R, Z, phi) trajectory and provides:
  - .intersect(phi) -> cross-section points at any toroidal angle
  - .volume() -> enclosed volume via shoelace + toroidal integration
  - .save(path) / .load(path) -> HDF5 persistence (never recompute)
  - .plot3d() -> 3D visualization

This principle enables:
  1. Physical consistency: all cross-sections show the SAME 3D object
  2. Efficiency: integrate once, query many times
  3. 3D visualization: all data available for full 3D plots
  4. Reproducibility: save orbits to disk, reload without reintegration
