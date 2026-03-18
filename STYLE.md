# pyna Style Guide

Language and naming conventions for code, docstrings, notebooks, and documentation.
All contributors and AI agents should follow these rules.

---

## 1. Scientific Naming — Proper Nouns Stay Capitalized

Physical laws, equations, and methods named after people **keep the person's name capitalized**,
even inside `snake_case` identifiers.

| ❌ Wrong | ✅ Correct | Reason |
|---|---|---|
| `biot_savart_field` | `Biot_Savart_field` | Biot & Savart are surnames |
| `grad_shafranov` | `Grad_Shafranov` | Grad & Shafranov are surnames |
| `rmp_coils` | `RMP_coils` | RMP = Resonant Magnetic Perturbation (acronym) |
| `fpt_response` | `FPT_response` | FPT = Field Period Transform (acronym) |
| `mhd_mode` | `MHD_mode` | MHD = MagnetoHydroDynamics (acronym) |
| `pest_coords` | `PEST_coords` | PEST = Peskin–Erdelyi–Storm–Todd (acronym) |
| `lcfs_radius` | `LCFS_radius` | LCFS = Last Closed Flux Surface (acronym) |

**Rule:** If it's an acronym or contains a person's name → preserve its conventional capitalization
in both Python identifiers and prose.

**Backwards compatibility:** This codebase has a single owner. Do **not** add
lowercase aliases. Rename the function and update **all** call sites immediately.
```python
# ✅ Correct: rename in place, update all callers
def Biot_Savart_field(...):
    ...
```

---

## 2. Matrix / Operator Notation — Follow Scientific Literature

Avoid generic single-letter names like `J` or `M` for matrices. Use the notation
standard in plasma physics and dynamical systems literature.

### Jacobian Matrices

| Context | Symbol | Meaning |
|---|---|---|
| Continuous orbit (flow map) | `DX` | $DX = \partial X(\phi) / \partial X_0$, orbital Jacobian |
| Poincaré map (discrete) | `DP` | $DP = \partial P(X_0) / \partial X_0$, monodromy matrix |
| Monodromy (one full turn) | `DP_m` | $DP_m = DP(\phi_0 + 2\pi m)$ after $m$ turns |

**In code:**
```python
# ✅ Correct
DP_arr = [orbit.Jac for orbit in monodromy_O.Jac_arr]
DP_eigvals = np.array([np.linalg.eigvals(DP) for DP in DP_arr])
DP_dets = np.array([np.linalg.det(DP) for DP in DP_arr])

# ❌ Wrong
J_arr = ...
J_eigvals = ...
```

**In LaTeX/markdown:**
```markdown
✅  The monodromy matrix $DP_m = \partial P^m / \partial X_0$ has eigenvalues ...
❌  The monodromy matrix $M = J(\phi_0 + 2\pi)$ has eigenvalues ...
```

### Response / Coupling Matrices

| Symbol | Meaning |
|---|---|
| `A` | Response matrix (control theory: $\delta B = A \cdot \delta I$) |
| `C_mn` | Mode coupling coefficient for toroidal mode $n$, poloidal mode $m$ |
| `DPm` | Discrete Poincaré map Jacobian (m turns) — same as `DP_m` above |

---

## 3. Variable Names — Physical Meaning First

Prefer physics-inspired names over generic ones:

| ❌ Avoid | ✅ Prefer | Notes |
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
- Section headers follow the logical steps: Setup → Computation → Visualization → Summary.
- Figures must have:
  - LaTeX axis labels (`$R$ (m)`, `$Z$ (m)`, `$\psi_\mathrm{norm}$`)
  - `plt.rcParams` with `font.family: serif`, `figure.dpi: 150`
  - Colorbars where color encodes a physical quantity
- Use `plasma` colormap for ψ_norm coloring.
- Stable manifold $W^s$: cool colormap (`GnBu`). Unstable manifold $W^u$: warm colormap (`Oranges`).

---

## 7. Commit Message Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(topo): add ftle_field for chaotic region detection
fix(tutorial): rename J→DP for Poincaré map Jacobian
fix(naming): Biot_Savart capitalization per scientific convention
docs(api): add Grad_Shafranov module docstring
refactor(coils): rename all rmp_ functions to RMP_ (no compat aliases)
```

---

## 8. Class Naming — Noun First, Qualifier Last

Class names should be read as **"what it is" first, "what kind" second**.
This keeps related classes adjacent in IDE autocomplete and `dir()` output.

**Rule:** Primary noun (the thing being described) comes first; backend, algorithm, symmetry, or variant qualifiers come after — and **qualifiers themselves are ordered from most fundamental to most specific**. A qualifier that is a prerequisite for another comes first.

| ❌ Wrong | ✅ Correct | Reason |
|---|---|---|
| `CUDAFieldLineTracer` | `FieldLineTracerCUDA` | Tracer is the noun; CUDA is the backend |
| `BiotSavartCoilField` | `CoilFieldBiotSavart` | CoilField is the noun; Biot-Savart is the method |
| `AnalyticCircularCoilField` | `CoilFieldAnalyticCircular` | CoilField is the noun; analytic+circular is the variant |
| `VectorPotentialField` | `CoilFieldVectorPotential` | CoilField is the noun; vector-potential is the method |
| `AxiSymmetricVectorField3D` | `VectorField3DAxiSymmetric` | `3D` before `AxiSymmetric`: dimensionality is prerequisite for symmetry |
| `AxiSymmetricScalarField3D` | `ScalarField3DAxiSymmetric` | same: you must first know it's 3D, then constrain to axisymmetric |

**Qualifier ordering principle:** Ask "does qualifier A need to exist before qualifier B makes sense?" If yes, A comes first. Examples:
- `3D` before `AxiSymmetric` — axisymmetry is a constraint on a 3D space; without 3D there is no axisymmetry to speak of
- `Cylindrical` before `AxiSymmetric` — `VectorFieldCylindrical3DAxiSymmetric` if fully spelled out
- Backend type (e.g. `CUDA`) is always last — it's the most incidental qualifier

**Why:** When you type `CoilField` in your IDE, you immediately see all coil field variants together. When you type `FieldLineTracer`, you see CPU and CUDA backends side by side. Consistent qualifier ordering makes the hierarchy readable without opening the source.

**No backward-compat aliases.** Rename and update all call sites immediately.

---

## 9. Quick Reference Cheatsheet

```
Person/acronym names    →  Capitalize: Biot_Savart, Grad_Shafranov, RMP, FPT, MHD, PEST
Poincaré map Jacobian  →  DP  (not J, not M)
Orbital Jacobian       →  DX  (not J)
Monodromy matrix       →  DP_m
Cylindrical coords     →  R, Z, phi
Normalized flux        →  psi_norm  (0=axis, 1=LCFS)
Stable manifold        →  W^s, GnBu colormap
Unstable manifold      →  W^u, Oranges colormap
Island island width    →  w_mn or island_width
Lundquist number       →  S  (capital S, not lundquist)
Connection length      →  Lc  (forward: Lc_plus, backward: Lc_minus, total: Lc_sum)
No compat aliases      →  rename and update all call sites immediately (no snake_case shims)
```

---

## 10. Trajectory vs. Orbit Terminology

Distinguish the two types of solution curves depending on whether time is continuous or discrete.

| English term | Chinese term | System type | Examples in pyna |
|---|---|---|---|
| **trajectory** | **轨道** | Continuous-time systems (ODEs, field-line flow) | Field-line tracer, guiding-centre orbit, `flt.py` solutions |
| **orbit** | **踪迹** | Discrete-time systems (maps, Poincaré iterates) | Poincaré map iterates, standard map, Hénon map |
| **orbit** | **轨迹** | Both (or unspecified) | Generic dynamical-systems contexts |

**Rules:**

- Use `trajectory` / `轨道` when the underlying system has **continuous time** (an ODE or flow).
- Use `orbit` / `踪迹` when the underlying system is a **discrete map** (iterating a Poincaré section, a symplectic map, etc.).
- When a statement applies to **both** continuous and discrete systems, or when the distinction is not relevant, use `orbit` in English and `轨迹` in Chinese.

**In code:**

```python
# ✅ Correct — field-line tracing is continuous time
def trace_trajectory(field, x0, phi_span): ...

# ✅ Correct — Poincaré section iterates are discrete time
def compute_orbit(poincare_map, x0, n_iterations): ...

# ✅ Correct — generic context valid for both
def plot_orbit(pts, ax): ...   # applies to any type of solution curve

# ❌ Wrong — do not call Poincaré iterates "trajectories"
def compute_trajectory(poincare_map, x0, n_iterations): ...
```

**Variable naming:**

```python
# Continuous-time solution
trajectory_RZPhi = flt.trace(field, x0, phi_span)

# Discrete-time Poincaré iterates
orbit_RZ = poincare_map.iterate(x0, n=100)

# Generic (both)
orbit_pts = ...          # acceptable in mixed or unspecified contexts
```
