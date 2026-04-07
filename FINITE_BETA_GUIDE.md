# Finite-Beta Perturbation Framework — Usage Guide

## Overview

This framework studies how the magnetic topology of the HAO stellarator changes
as β (plasma pressure normalized to magnetic pressure) climbs from vacuum (β=0)
to finite values.  It uses **functional perturbation theory** to solve the coupled
MHD equilibrium system:

```
δJ × B + J × (δB_plasma + δB_external) = ∇δp           (force balance)
∇ · δB_plasma = 0                                        (div-free constraint)
∇ × δJ = μ₀ δB_plasma                                   (Ampère consistency)
```

### Current Components Included

| Component | Physics | Formula |
|-----------|---------|---------|
| **Diamagnetic** | MHD pressure balance | J_dia = (∇p × B) / B² |
| **Pfirsch-Schlüter** | Neoclassical collisional | J_PS ∝ (∇p × B)/B² · (1 + ε·cos θ) |
| **Bootstrap** | Collisionless trapped particles | J_BS ∝ C_BS · (dp/dψ) · B/⟨B²⟩ |
| **Parallel** | Ohmic + q-profile matching | J_∥ = σ_∥ · B |

## Quick Start

### 1. Activate the environment

```bash
wsl -d Ubuntu
source ~/mhd_env/bin/activate
cd ~/repos/pyna
```

### 2. Run a quick test

```bash
# Uses only 20 coils, β = [0.0, 0.01, 0.02]
python scripts/hao_beta_climb.py --test
```

### 3. Full production run

```bash
# All 332 coils, β = 0 → 0.05
python scripts/hao_beta_climb.py --full

# Custom β range
python scripts/hao_beta_climb.py --beta 0.0 0.01 0.02 0.03 0.04 0.05

# Specify output directory
python scripts/hao_beta_climb.py --full --output ./results/hao_beta_scan
```

## Programmatic Usage

### Basic usage

```python
from pyna.MCF.equilibrium.finite_beta_perturbation import (
    FiniteBetaPerturbation,
    load_hao_coils,
)

# Load all HAO coil vacuum field data
coil_files = load_hao_coils(
    vacuum_dir="/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields",
    exclude_indices={38, 122, 206, 290},  # excluded coils
)

# Define pressure profile shape (normalized)
def p_profile_func(psi_n: float) -> float:
    """p(ψ_n) = (1 - ψ_n)^α, normalized to 1 at magnetic axis."""
    return max(0.0, 1.0 - psi_n) ** 2.0

# Create solver
solver = FiniteBetaPerturbation(
    coil_files=coil_files,
    p_profile_func=p_profile_func,
    beta_values=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    alpha_pressure=2.0,
    max_outer_iter=20,
    tol=1e-4,
    verbose=True,
)

# Run the β climb
history = solver.run()

# Access results
for state in history:
    print(f"β = {state.beta:.4f}: residual = {state.residual:.4e}, "
          f"converged = {state.converged}")
    
    # B_total: shape (3, nR, nZ, nPhi)
    # J_total: shape (3, nR, nZ, nPhi)
    # p_profile: shape (nR, nZ, nPhi)
```

### Individual current components

```python
from pyna.MCF.equilibrium.finite_beta_perturbation import (
    compute_diamagnetic_current,
    compute_pfirsch_schlueter_current,
    compute_bootstrap_current,
    compute_pressure_gradient,
)

# Given B_field (3, nR, nZ, nPhi) and pressure p (nR, nZ, nPhi)
grad_p = compute_pressure_gradient(p, R_grid, Z_grid, Phi_grid)

J_dia = compute_diamagnetic_current(B_field, grad_p)
J_PS = compute_pfirsch_schlueter_current(B_field, grad_p, R_grid, Z_grid, Phi_grid)
J_BS = compute_bootstrap_current(B_field, p, psi_norm, R_grid, Z_grid, Phi_grid)

# Total current
J_total = J_dia + J_PS + J_BS
```

### Using individual coil data

```python
from pyna.MCF.equilibrium.finite_beta_perturbation import CoilVacuumField

# Load a single coil
coil = CoilVacuumField.from_npz(
    "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/"
    "dipole_coil_100_current-87600.00A.npz"
)

print(f"Coil index: {coil.coil_index}")
print(f"Coil current: {coil.coil_current:.0f} A")
print(f"Field shape: {coil.shape}")  # (nR, nZ, nPhi)
```

## Output Files

Each run creates a timestamped output directory with:

| File | Description |
|------|-------------|
| `metadata.json` | Run parameters and convergence info |
| `summary.csv` | β, convergence status, residuals |
| `state_beta_XXXXX.npz` | Full field data at each β step |
| `beta_convergence.png` | Convergence plots |
| `field_slice_beta*.png` | 2D field slices at each β |

## Physics Details

### Pressure Profile

```
p(ψ_n) = β · ⟨B²⟩ / (2μ₀) · (1 - ψ_n)^α
```

where:
- `ψ_n` = normalised flux label (0 at axis, 1 at edge)
- `α` = pressure peaking factor (default 2.0)
- `⟨B²⟩` = volume-averaged |B|²

### Bootstrap Current Model

The bootstrap coefficient uses a simplified Sauter-like model:

```
C_BS = 0.6 · (1.46·√ε - 0.46·ε) · 1/(1 + ν*)
```

where:
- `ε` = effective helical ripple (~0.3 for HAO)
- `ν*` = collisionality (~0.1)
- Factor 0.6 accounts for 3D stellarator reduction

### Pfirsch-Schlüter Enhancement

The PS current is enhanced by toroidal variation:

```
f_PS = 1 + ε · cos(θ)
```

where ε is the inverse aspect ratio.

## File Locations

| Item | Path |
|------|------|
| Framework code | `pyna/pyna/MCF/equilibrium/finite_beta_perturbation.py` |
| Workflow script | `pyna/scripts/hao_beta_climb.py` |
| Tests | `pyna/tests/test_finite_beta_perturbation.py` |
| Vacuum field data | `/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/` |
| Coil geometry | `/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/Coil_Points_full_1m/` |

## Virtual Environment

```bash
# Activate
source ~/mhd_env/bin/activate

# Verify installations
python -c "import dolfinx; print(dolfinx.__version__)"  # 0.10.0
python -c "import petsc4py; print(petsc4py.__version__)"  # 3.19.6
python -c "import numpy; print(numpy.__version__)"  # 1.26.4
```

## Next Steps / TODO

- [ ] Connect to FEniCSx nonlinear corrector (`fenicsx_corrector.py`)
- [ ] Integrate with topoquest topology analysis (iota, islands, LCFS)
- [ ] Add proper ψ_n computation from field-line tracing
- [ ] Implement full sparse matrix assembly for perturbation system
- [ ] Parallel MPI assembly and solve (use petsc4py)
- [ ] Add Poincaré section generation at each β step
- [ ] Add island chain detection and tracking
- [ ] Add rotational transform profile computation
