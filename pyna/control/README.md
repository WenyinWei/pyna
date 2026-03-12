# `pyna.control` — Multi-objective Magnetic Topology Control via FPT

## Overview

`pyna.control` implements **Functional Perturbation Theory (FPT)** for real-time
multi-objective control of magnetic topology in fusion devices (tokamaks and
stellarators).

> **Reference:**  Wei, W. et al.  
> *"Functional perturbation theory under axisymmetry: Simplified formulae and
> their uses for tokamaks"*

---

## Theory Background

### The A-matrix

The fundamental object is the 2×2 Jacobian of the normalised poloidal-to-toroidal
field ratio:

```
A = ∂(R·B_pol / B_phi) / ∂(R, Z)
  = [[∂(R·BR/Bphi)/∂R,  ∂(R·BR/Bphi)/∂Z],
     [∂(R·BZ/Bphi)/∂R,  ∂(R·BZ/Bphi)/∂Z]]
```

Evaluated at an X-point or O-point (the "cycle"), **A encodes all local
topology**.

### Poincaré Map Jacobian DPm

| Configuration | Formula | Cost |
|---------------|---------|------|
| Axisymmetric (tokamak) | `DPm = exp(2π·A)` | O(1) — exact |
| 3D / RMP tokamak | integrate `dDPm/dφ = [A(φ), DPm]` along orbit | O(N_φ) |

`det(DPm) = 1` (area-preserving / Liouville theorem).

### Key FPT Results (Axisymmetric, Closed Form)

**Cycle shift:**
```
δx_cyc = -A⁻¹ · δg(x_cyc)
where δg = [R·δBR/Bphi - R·BR·δBphi/Bphi², R·δBZ/Bphi - R·BZ·δBphi/Bphi²]
```

**DPm change:**
```
δDPm = ∫₀¹ exp(α·2π·A) · δ(2π·A) · exp((1−α)·2π·A) dα
```
where `δA = δA_direct + δA_indirect` (local field change + chain-rule via cycle shift).

**Manifold shift:**  linear ODE along the arc ζ:
```
d/dζ (δX^{u/s}) = ±{ δB_pol + ∂B_pol/∂(R,Z) · δX^{u/s} }
```
Initial condition: `δX^{u/s}(0) = δx_cyc`.

**Flux surface deformation:**  Fourier least-squares on
```
δP^k(χ) + DP^k(χ)·δχ(θ,r) = δχ(θ + k·Δθ, r)
```

---

## Axisymmetric vs 3D: Computational Cost

```
Axisymmetric (tokamak)          3D / stellarator / RMP
──────────────────────          ─────────────────────────
A-matrix: 4 field evals         A(φ): evaluated at N_φ planes
DPm = exp(2πA): O(1)            dDPm/dφ integrated: O(N_φ)
δx_cyc: solve 2×2 system        same structure, φ-dependent
δDPm: 20-point quadrature       same, but A varies
Manifold shift: Euler along ζ   same
Total: < 1 ms                   ~ 10 – 100 ms (N_φ ≈ 100–1000)
```

---

## Multi-objective Control Workflow

```
         ┌─────────────────────────────────────────────┐
         │           pyna.control workflow              │
         └─────────────────────────────────────────────┘

  Equilibrium                    Coil geometry
  (field_func)                   (coil_field_funcs[k])
       │                                │
       ▼                                ▼
  compute_topology_state()     build_response_matrix()
  ┌───────────────────┐        ┌────────────────────────┐
  │  TopologyState    │        │  R[n_obs, n_coils]     │
  │  xpoints, opoints │──────►│  ∂obs_i / ∂I_k         │
  │  gap_gi, q_samples│        └─────────────┬──────────┘
  └──────────┬────────┘                      │
             │  current state                │
             ▼                               ▼
          TopologyController.solve(current, target, R, weights)
          ┌────────────────────────────────────────────────────┐
          │  min Σ_i w_i |s_i + R_ij δI_j − t_i|² + λ‖δI‖²  │
          │  s.t.  I_min ≤ δI ≤ I_max                         │
          └──────────────────────────────────────────────────┬─┘
                                                              │
                                                          δI (optimal)
                                                              │
                                                              ▼
                                                    Apply to coil currents
                                                    ──► Re-compute state
                                                    ──► Iterate if needed
```

---

## Module Structure

| File | Contents |
|------|---------|
| `fpt.py` | Core FPT formulae: `A_matrix`, `DPm_axisymmetric`, `cycle_shift`, `DPm_change`, `delta_A_total`, `manifold_shift`, `flux_surface_deformation` |
| `topology_state.py` | `TopologyState`, `XPointState`, `OPointState`, `SurfaceFate`, `compute_topology_state` |
| `response_matrix.py` | `build_response_matrix` — assembles ∂state/∂controls |
| `optimizer.py` | `TopologyController`, `ControlWeights`, `ControlConstraints` |
| `surface_fate.py` | `greene_residue`, `classify_surface_fate`, `scan_surface_fates` |

---

## Generic Design

Although written for magnetic fusion, the FPT framework applies to **any
area-preserving 2-D dynamical system with fixed points**:

- Replace `field_func` with your Hamiltonian vector field.
- Replace "coil fields" with your system's control perturbations.
- `A_matrix` → linearisation of the return map at any fixed point.
- `DPm_axisymmetric` → period-T linearised map when the system is autonomous.
- `TopologyController` → generic multi-objective controller for fixed-point
  positions, eigenvalues, and invariant manifold geometry.

---

## Example: Tokamak (axisymmetric)

```python
import numpy as np
from pyna.control import (
    compute_topology_state, build_response_matrix,
    TopologyController, ControlWeights, ControlConstraints,
)

# 1. Define equilibrium field function
def field(rzphi):
    R, Z, phi = rzphi
    # returns [BR/|B|, BZ/|B|, Bphi/(R|B|)]  — your equilibrium here
    ...

# 2. Coil perturbation fields (one per coil, per unit current)
coil_fields = [lambda rzphi, k=k: coil_delta_B(rzphi, k) for k in range(6)]

# 3. Compute current topology state
state = compute_topology_state(
    field,
    xpoint_guesses=[(1.5, -1.2)],
    opoint_guesses=[(1.7,  0.0)],
)

# 4. Build response matrix
R_mat, labels = build_response_matrix(field, coil_fields, state)

# 5. Define target state (modify X-point position slightly)
target = compute_topology_state(field, ...)   # or construct manually

# 6. Solve for optimal coil changes
ctrl = TopologyController(n_coils=6)
weights = ControlWeights(gap_gi=100.0, xpoint_position=20.0)
constraints = ControlConstraints(I_max=np.full(6, 5e3))
delta_I, result = ctrl.solve(state, target, R_mat, labels, weights, constraints)

print("Optimal δI [A]:", delta_I)
print("Predicted response:", ctrl.predict_response(state, delta_I, R_mat, labels))
```

## Example: Stellarator (3D)

```python
# Same API — only field_func and DPm differ.
# Set is_axisymmetric=False; DPm is computed via ODE integration internally.
state_3d = compute_topology_state(
    field_3d,
    xpoint_guesses=[...],
    opoint_guesses=[...],
    is_axisymmetric=False,   # triggers φ-integration path
)
```

---

## Greene's Residue and Surface Fate

```python
from pyna.control import classify_surface_fate, greene_residue

R0 = greene_residue(state.xpoints[0].DPm)
fate = classify_surface_fate(
    iota=op.iota,
    delta_iota=0.001,
    DPm=state.xpoints[0].DPm,
    delta_DPm=delta_DPm,
    epsilon_KAM=0.05,
    epsilon_chaos=0.30,
)
print(f"Greene residue: {R0:.4f}")
print(f"Surface fate under perturbation: {fate.name}")
```
