# `pyna.topo` — Topological Analysis

## Overview

`pyna.topo` provides all tools for analysing the topological structure of
phase space in both continuous (ODE / flow) and discrete (map) dynamical systems.
Core capabilities:

- Poincaré-section construction and fixed-point finding
- Island-chain detection and width measurement
- Monodromy matrix computation and stability classification
- Invariant-manifold extraction (stable W^s and unstable W^u branches)
- Variational / tangent-map integration
- Lyapunov exponents and chaos diagnostics
- Classical test maps (Hénon, Standard)

---

## Module reference

| File | Key public API |
|------|----------------|
| `poincare.py` | `Section` (abstract), `ToroidalSection`, `PoincareMap` |
| `fixed_points.py` | `poincare_map()`, `scan_fixed_point_seeds()`, `find_periodic_orbit()`, `classify_fixed_point()` |
| `island.py` | `Island`, `IslandChain` dataclasses |
| `island_extract.py` | `detect_islands()`, `island_halfwidth()` |
| `monodromy.py` | `MonodromyAnalysis`: `DPm`, `eigenvalues`, `stability_index`, `Greene_residue` |
| `variational.py` | `PoincareMapVariationalEquations`, `tangent_map()` |
| `manifold_improve.py` | `StableManifold`, `UnstableManifold` (arc-length parameterised extraction; accelerated backend by default) |
| `manifold.py` | `grow_manifold_from_Xcycle()` |
| `topology_analysis.py` | `analyse_topology()`, `TopologyReport` |

Public API names in `pyna.topo` describe the mathematical object, not the backend.
For example, `StableManifold` uses the accelerated cyna integrator by default when
available; backend-explicit names are compatibility aliases, not the preferred API.
| `cycle.py` | Cycle detection helpers |
| `classical_maps.py` | `HenonMap`, `StandardMap` (test cases / benchmarks) |
| `chaos.py` | `ftle_field()`, `chirikov_overlap()`, `chaotic_boundary_estimate()` |

---

## Poincaré map Jacobian naming (STYLE.md §2)

| Symbol | Meaning |
|--------|---------|
| `DX` | Orbital Jacobian of the continuous flow `∂X(φ)/∂X₀` |
| `DP` | Poincaré map Jacobian (one section crossing) `∂P(X₀)/∂X₀` |
| `DPm` | Monodromy matrix after *m* full turns `DP(φ₀ + 2πm)` |

Never use `J` or `M` for these.

---

## Quick-start: find X-points and O-points

```python
import numpy as np
from pyna.topo.fixed_points import poincare_map, find_periodic_orbit, classify_fixed_point

# Define a Poincaré section at phi=0
def field_func(rzphi):
    ...  # returns [dR/dphi, dZ/dphi, 1]

fps = poincare_map(field_func, seeds, n_iter=5)
for fp in fps:
    classification = classify_fixed_point(fp.DPm)
    print(fp.R, fp.Z, classification)
```

## Quick-start: island chain

```python
from pyna.topo.island_extract import detect_islands
from pyna.topo.monodromy import MonodromyAnalysis

islands = detect_islands(poincare_pts, q_target=3.0)
for island in islands:
    analysis = MonodromyAnalysis(island.x_point.DPm)
    print(f"q={island.q_mn}  width={island.width:.4f}  "
          f"Greene_residue={analysis.Greene_residue:.4f}")
```

---

## Greene's residue

For a period-m fixed point with monodromy `DPm`:

```
R₀ = (2 - Tr(DPm)) / 4
```

- `R₀ < 0` → hyperbolic (X-point, unstable)
- `0 < R₀ < 1` → elliptic (O-point, stable)
- `R₀ > 1` → hyperbolic with reflection
- `R₀ → 1` → onset of chaos (KAM surface breaks)

---

## Manifold colourmap convention (STYLE.md §6)

| Manifold | Colour map |
|----------|------------|
| Stable W^s | `GnBu` (cool) |
| Unstable W^u | `Oranges` (warm) |
