# pyna Style Guide

Language and naming conventions for code, docstrings, notebooks, and documentation.
All contributors and AI agents should follow these rules.

---

## 1. Scientific Naming 鈥?Proper Nouns Stay Capitalized

Physical laws, equations, and methods named after people **keep the person's name capitalized**,
even inside `snake_case` identifiers.

| 鉂?Wrong | 鉁?Correct | Reason |
|---|---|---|
| `biot_savart_field` | `Biot_Savart_field` | Biot & Savart are surnames |
| `grad_shafranov` | `Grad_Shafranov` | Grad & Shafranov are surnames |
| `Greene_residue` | `Greene_residue` | Greene is a surname |
| `gauss_legendre` | `Gauss_Legendre` | Gauss & Legendre are surnames |
| `coulomb_gauge` | `Coulomb_gauge` | Coulomb is a surname |
| `ampere_law` | `Ampere_law` | Ampere (Ampère) is a surname |
| `fourier_transform` | `Fourier_transform` | Fourier is a surname |
| `laplace_equation` | `Laplace_equation` | Laplace is a surname |
| `rmp_coils` | `RMP_coils` | RMP = Resonant Magnetic Perturbation (acronym) |
| `fpt_response` | `FPT_response` | FPT = Field Period Transform (acronym) |
| `mhd_mode` | `MHD_mode` | MHD = MagnetoHydroDynamics (acronym) |
| `pest_coords` | `PEST_coords` | PEST = Peskin鈥揈rdelyi鈥揝torm鈥揟odd (acronym) |
| `lcfs_radius` | `LCFS_radius` | LCFS = Last Closed Flux Surface (acronym) |
| `mcf_equilibrium` | `MCF_equilibrium` | MCF = Magnetic Confinement Fusion (acronym) |
| `nbi_power` | `NBI_power` | NBI = Neutral Beam Injection (acronym) |
| `ecrh_heating` | `ECRH_heating` | ECRH = Electron Cyclotron Resonance Heating (acronym) |
| `eccd_current` | `ECCD_current` | ECCD = Electron Cyclotron Current Drive (acronym) |
| `icrh_heating` | `ICRH_heating` | ICRH = Ion Cyclotron Resonance Heating (acronym) |
| `iccd_current` | `ICCD_current` | ICCD = Ion Cyclotron Current Drive (acronym) |

**Rule:** If it's an acronym or contains a person's name 鈫?preserve its conventional capitalization
in both Python identifiers and prose. This applies to **all** contexts: module names, function
names, variable names, class names, docstrings, comments, and notebook prose.

**Backwards compatibility:** This codebase has a single owner. Do **not** add
lowercase aliases. Rename the function and update **all** call sites immediately.
```python
# 鉁?Correct: rename in place, update all callers
def Biot_Savart_field(...):
    ...
```

---

## 2. Matrix / Operator Notation 鈥?Follow Scientific Literature

Avoid generic single-letter names like `J` or `M` for matrices. Use the notation
standard in plasma physics and dynamical systems literature.

### Jacobian Matrices

| Context | Symbol | Meaning |
|---|---|---|
| Continuous orbit (flow map) | `DX` | $DX = \partial X(\phi) / \partial X_0$, orbital Jacobian |
| Poincar茅 map (discrete) | `DP` | $DP = \partial P(X_0) / \partial X_0$, monodromy matrix |
| Monodromy (one full turn) | `DP_m` | $DP_m = DP(\phi_0 + 2\pi m)$ after $m$ turns |

**In code:**
```python
# 鉁?Correct
DP_arr = [orbit.Jac for orbit in monodromy_O.Jac_arr]
DP_eigvals = np.array([np.linalg.eigvals(DP) for DP in DP_arr])
DP_dets = np.array([np.linalg.det(DP) for DP in DP_arr])

# 鉂?Wrong
J_arr = ...
J_eigvals = ...
```

**In LaTeX/markdown:**
```markdown
鉁? The monodromy matrix $DP_m = \partial P^m / \partial X_0$ has eigenvalues ...
鉂? The monodromy matrix $M = J(\phi_0 + 2\pi)$ has eigenvalues ...
```

### Response / Coupling Matrices

| Symbol | Meaning |
|---|---|
| `A` | Response matrix (control theory: $\delta B = A \cdot \delta I$) |
| `C_mn` | Mode coupling coefficient for toroidal mode $n$, poloidal mode $m$ |
| `DPm` | Discrete Poincar茅 map Jacobian (m turns) 鈥?same as `DP_m` above |

---

## 3. Variable Names 鈥?Physical Meaning First

Prefer physics-inspired names over generic ones:

| 鉂?Avoid | 鉁?Prefer | Notes |
|---|---|---|
| `x`, `y` | `R`, `Z` | Cylindrical coordinates |
| `pts` | `RZ_pts` or `Poincare_pts` | Be explicit about what kind of points |
| `field` | `B_field` or `RMP_field` | |
| `result` | `island_width` or `Greene_residue` | |
| `data` | `Poincare_scan` | |

---

## 4. Coordinate System Labels

Always label which coordinate system is in use:

| System | Labels | Notes |
|---|---|---|
| Cylindrical | `R, Z, phi` | Standard tokamak/stellarator |
| Boozer | `psi_B, theta_B, phi_B` | Subscript B distinguishes from cylindrical |
| PEST | `psi_P, theta_P, phi` | |
| Flux | `psi_norm` | Normalized: 0 = axis, 1 = LCFS |

---

## 5. Docstring Style

Use NumPy docstring format. Physical quantities must include units:

```python
def Biot_Savart_field(coil_pts, coil_current, R_grid, Z_grid, Phi_grid=None):
    """Compute the magnetic field of a coil set via Biot-Savart integration.

    Parameters
    ----------
    coil_pts : ndarray, shape (N, 3)
        Coil filament points [R, Z, phi] in meters/radians.
    coil_current : float
        Current in amperes (A).
    R_grid : ndarray
        Radial grid points (m).
    Z_grid : ndarray
        Vertical grid points (m).

    Returns
    -------
    BR, BZ, Bphi : ndarray
        Magnetic field components (T).
    """
```

---

## 6. Notebook Conventions

- Each notebook must have a **title cell** (level-1 markdown) explaining the physics scenario.
- Section headers follow the logical steps: Setup 鈫?Computation 鈫?Visualization 鈫?Summary.
- Figures must have:
  - LaTeX axis labels (`$R$ (m)`, `$Z$ (m)`, `$\psi_\mathrm{norm}$`)
  - `plt.rcParams` with `font.family: serif`, `figure.dpi: 150`
  - Colorbars where color encodes a physical quantity
- Use `plasma` colormap for 蠄_norm coloring.
- Stable manifold $W^s$: cool colormap (`GnBu`). Unstable manifold $W^u$: warm colormap (`Oranges`).

---

## 7. Commit Message Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(topo): add ftle_field for chaotic region detection
fix(tutorial): rename J鈫扗P for Poincar茅 map Jacobian
fix(naming): Biot_Savart capitalization per scientific convention
docs(api): add Grad_Shafranov module docstring
refactor(coils): rename all rmp_ functions to RMP_ (no compat aliases)
```

---

## 8. Public API vs Backend Names

Public API names should describe the **mathematical object or physical operation**,
not the implementation backend.

- Prefer `StableManifold`, `UnstableManifold`, `monodromy_matrix`
- Avoid backend-branded public names like `CynaStableManifold` in new code
- cyna is the default accelerated backend inside pyna when available
- If a pure-Python / SciPy fallback is needed, expose it only as an explicit
  secondary name such as `ScipyStableManifold`

This keeps the user-facing API physics-first and lets implementations evolve
without renaming scientific concepts.

## 9. Class Naming 鈥?Noun First, Qualifier Last

Class names should be read as **"what it is" first, "what kind" second**.
This keeps related classes adjacent in IDE autocomplete and `dir()` output.

**Rule:** Primary noun (the thing being described) comes first; backend, algorithm, symmetry, or variant qualifiers come after 鈥?and **qualifiers themselves are ordered from most fundamental to most specific**. A qualifier that is a prerequisite for another comes first.

| 鉂?Wrong | 鉁?Correct | Reason |
|---|---|---|
| `CUDAFieldLineTracer` | `FieldLineTracerCUDA` | Tracer is the noun; CUDA is the backend |
| `BiotSavartCoilField` | `CoilFieldBiotSavart` | CoilField is the noun; Biot-Savart is the method |
| `AnalyticCircularCoilField` | `CoilFieldAnalyticCircular` | CoilField is the noun; analytic+circular is the variant |
| `VectorPotentialField` | `CoilFieldVectorPotential` | CoilField is the noun; vector-potential is the method |
| `AxiSymmetricVectorField3D` | `VectorField3DAxiSymmetric` | `3D` before `AxiSymmetric`: dimensionality is prerequisite for symmetry |
| `AxiSymmetricScalarField3D` | `ScalarField3DAxiSymmetric` | same: you must first know it's 3D, then constrain to axisymmetric |
| `CylindricalVectorField3D` | `VectorField3DCylindrical` | VectorField is the noun; Cylindrical is the coord-system qualifier |
| `CylindricalScalarField3D` | `ScalarField3DCylindrical` | ScalarField is the noun; Cylindrical is the coord-system qualifier |
| `CylindricalGridVectorField3D` | `VectorField3DCylindrical` | same rule; "Grid" is implicit in "Cylindrical" (regular grid) |
| `CylindricalCoords3D` | `Coords3DCylindrical` | Coords is the noun; 3D is the dimension; Cylindrical is the geometry |
| `SphericalCoords3D` | `Coords3DSpherical` | same ordering rule |
| `VacuumCoilField` | `CoilFieldVacuum` | CoilField is the noun; Vacuum is the physics qualifier |
| `SuperpositionField` | `CoilFieldSuperposition` | CoilField is the noun; Superposition is the method |
| `AxisymEquilibrium` | `EquilibriumAxisym` | Equilibrium is the noun; Axisym(metric) is the qualifier |
| `SolovevEquilibrium` | `EquilibriumSolovev` | Equilibrium is the noun; Solov'ev is the (proper-name) qualifier |
| `SimpleStellarartor` | `StellaratorSimple` | Stellarator is the noun; Simple is the variant qualifier (also fixes typo) |

**Qualifier ordering principle:** Ask "does qualifier A need to exist before qualifier B makes sense?" If yes, A comes first. Examples:
- `3D` before `AxiSymmetric` 鈥?axisymmetry is a constraint on a 3D space; without 3D there is no axisymmetry to speak of
- `3D` before `Cylindrical` 鈥?cylindrical coordinates describe a 3D space
- Backend type (e.g. `CUDA`) is always last 鈥?it's the most incidental qualifier

**Single canonical hierarchy.** If a class already exists in the canonical hierarchy (`pyna.fields`), do **not** create a parallel class elsewhere. Instead extend or import from the canonical source. The `pyna.MCF.coils.field` module is a thin re-export layer only.

**No backward-compat aliases.** Rename and update all call sites immediately.

---

## 9. Quick Reference Cheatsheet

```
Person/acronym names    鈫? Capitalize: Biot_Savart, Grad_Shafranov, RMP, FPT, MHD, PEST
Poincar茅 map Jacobian  鈫? DP  (not J, not M)
Orbital Jacobian       鈫? DX  (not J)
Monodromy matrix       鈫? DP_m
Cylindrical coords     鈫? R, Z, phi
Normalized flux        鈫? psi_norm  (0=axis, 1=LCFS)
Stable manifold        鈫? W^s, GnBu colormap
Unstable manifold      鈫? W^u, Oranges colormap
Island island width    鈫? w_mn or island_width
Lundquist number       鈫? S  (capital S, not lundquist)
Connection length      鈫? Lc  (forward: Lc_plus, backward: Lc_minus, total: Lc_sum)
No compat aliases      鈫? rename and update all call sites immediately (no snake_case shims)
```

---

## 10. Trajectory vs. Orbit Terminology

Distinguish the two types of solution curves depending on whether time is continuous or discrete.

| English term | Chinese term | System type | Examples in pyna |
|---|---|---|---|
| **trajectory** | **杞ㄩ亾** | Continuous-time systems (ODEs, field-line flow) | Field-line tracer, guiding-centre orbit, `flt.py` solutions |
| **orbit** | **韪抗** | Discrete-time systems (maps, Poincar茅 iterates) | Poincar茅 map iterates, standard map, H茅non map |
| **orbit** | **杞ㄨ抗** | Both (or unspecified) | Generic dynamical-systems contexts |

**Rules:**

- Use `trajectory` / `杞ㄩ亾` when the underlying system has **continuous time** (an ODE or flow).
- Use `orbit` / `韪抗` when the underlying system is a **discrete map** (iterating a Poincar茅 section, a symplectic map, etc.).
- When a statement applies to **both** continuous and discrete systems, or when the distinction is not relevant, use `orbit` in English and `杞ㄨ抗` in Chinese.

**In code:**

```python
# 鉁?Correct 鈥?field-line tracing is continuous time
def trace_trajectory(field, x0, phi_span): ...

# 鉁?Correct 鈥?Poincar茅 section iterates are discrete time
def compute_orbit(poincare_map, x0, n_iterations): ...

# 鉁?Correct 鈥?generic context valid for both
def plot_orbit(pts, ax): ...   # applies to any type of solution curve

# 鉂?Wrong 鈥?do not call Poincar茅 iterates "trajectories"
def compute_trajectory(poincare_map, x0, n_iterations): ...
```

**Variable naming:**

```python
# Continuous-time solution
trajectory_RZPhi = flt.trace(field, x0, phi_span)

# Discrete-time Poincar茅 iterates
orbit_RZ = poincare_map.iterate(x0, n=100)

# Generic (both)
orbit_pts = ...          # acceptable in mixed or unspecified contexts
```

