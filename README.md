# pyna -- Python DYNAmics

<p align="center">
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/pyversions/pyna-chaos" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License"></a>
  <a href="https://wenyinwei.github.io/pyna/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://github.com/WenyinWei/pyna/actions"><img src="https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/WenyinWei/pyna/actions/workflows/notebook-tests.yml"><img src="https://github.com/WenyinWei/pyna/actions/workflows/notebook-tests.yml/badge.svg" alt="Notebook Tests"></a>
</p>

**pyna** (**Py**thon **D**Y**NA**mics) is a research library for **dynamical systems** and **magnetic confinement fusion (MCF)** plasma physics. It covers the full workflow from analytic equilibria and field-line tracing to topological island analysis, manifold visualization, and non-resonant torus deformation theory.

> **Author:** [Wenyin Wei](https://github.com/WenyinWei) · **PyPI:** `pyna-chaos` · **Julia companion:** [Juna.jl](https://github.com/WenyinWei/Juna.jl)

---

## ✨ Highlights

| Feature | Details |
|---------|---------|
| 🔀 **Field-line tracing** | RK4 integrator, parallel CPU, optional CUDA (118× speedup) |
| 🌀 **Poincaré sections & island chains** | Multi-section crossing accumulation, island-chain extraction, X/O-point analysis |
| 🗺️ **Manifold visualization** | Publication-quality stable/unstable manifold plots for tokamaks |
| 🧲 **Toroidal geometry** | Coordinates, coils, diagnostics, and field-line tooling |
| 📐 **Magnetic coordinates** | PEST, Boozer, Hamada, Equal-arc transformations |
| 📡 **Torus deformation** | Non-resonant BNF-derived analytic spectral theory (Wei 2025) |
| ⚡ **C++ acceleration** | Optional `cyna` backend for performance-critical ops |

---

## 📦 Installation

```bash
# Stable release (PyPI)
pip install pyna-chaos

# Development version
git clone https://github.com/WenyinWei/pyna.git
cd pyna
pip install -e ".[dev]"

# With GPU support (CUDA 12)
pip install "pyna-chaos[cuda]"
```

---

## 🚀 Quick Start

### Field-line tracing

```python
from pyna.toroidal.equilibrium import EquilibriumSolovev as SolovevEquilibrium
from pyna.flt import FieldLineTracer

eq = SolovevEquilibrium.iter_like()
tracer = FieldLineTracer(eq.Bfield)
trajectory = tracer.trace(R0=2.0, Z0=0.0, phi_end=200 * 2 * 3.14159)
```

### Poincaré crossings and island chains

```python
import numpy as np
from pyna.flt import FieldLineTracer
from pyna.topo.poincare import PoincareAccumulator, poincare_from_fieldlines
from pyna.topo.section import ToroidalSection

# Canonical section type for topology APIs.
section = ToroidalSection(0.0)
acc = PoincareAccumulator([section])

# or trace directly from field lines
start_pts = np.array([
    [1.8, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.2, 0.0, 0.0],
])
acc = poincare_from_fieldlines(
    field_func=field_func,
    start_pts=start_pts,
    sections=[section],
    t_max=500 * 2 * np.pi,
    backend=tracer,
)
crossings = acc.crossing_array(0)
```

### EAST tokamak manifold visualization

```python
from pyna.toroidal.visual.tokamak_manifold import (
    plot_equilibrium_cross_section,
    plot_poincare_orbits,
    plot_manifold_bundle,
)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 7))
plot_equilibrium_cross_section(ax, eq)
plot_poincare_orbits(ax, orbits, cmap_by='psi_norm')
plot_manifold_bundle(ax, stable_manifold, unstable_manifold)
plt.savefig("east_manifold.png", dpi=300)
```

### Torus deformation calculation

```python
import numpy as np
from pyna import toroidal

# The package root now exposes the preferred toroidal namespace directly.
non_resonant_deformation_spectrum = toroidal.non_resonant_deformation_spectrum
poincare_section_deformation = toroidal.poincare_section_deformation
mean_radial_displacement = toroidal.mean_radial_displacement

# Define perturbation spectrum (m, n, δBr, δBθ, δBφ in T·m)
m = np.array([1, 2, 3])
n = np.array([1, 1, 1])
dBr = np.array([1e-4+0j, 5e-5+0j, 2e-5+0j])
dBth = np.zeros(3, dtype=complex)
dBph = np.zeros(3, dtype=complex)

spec = non_resonant_deformation_spectrum(
    m, n, dBr, dBth, dBph,
    iota=0.35, Bphi=4.5, Btheta=0.3, r=1.0
)
mean_dr = mean_radial_displacement(spec, iota_prime=-0.1)
print(f"Mean radial displacement: {mean_dr:.4f} m")
```

---

## 📂 Module Overview

### Core dynamical systems

| Module | Description |
|--------|-------------|
| `pyna.system` | Abstract dynamical system hierarchy (`DynamicalSystem`, `VectorField*D`) |
| `pyna.flt` | Field-line tracer: RK4, parallel CPU, CUDA/OpenCL backends |
| `pyna.topo.toroidal_island` | Rational surface location, theoretical island half-width |
| `pyna.topo.poincare` | Multi-section crossing accumulation and section helpers |
| `pyna.topo.manifold` | Stable/unstable manifold computation |
| `pyna.topo.toroidal_cycle` | Periodic orbit (X/O cycle) detection and analysis |
| `pyna.coord` | Flux coordinate transformations |
| `pyna.draw` | High-level plotting utilities |
| `pyna.gc` | Guiding-centre orbit integration |
| `pyna.interact` | Interactive widgets (Jupyter) |
| `pyna.utils` | Miscellaneous helpers |

### Toroidal plasma physics (`pyna.toroidal`)

`pyna.toroidal` is the toroidal-geometry namespace for coordinates, coils,
diagnostics, control helpers, and visualisation.

| Submodule | Description |
|-----------|-------------|
| `toroidal.equilibrium` | Toroidal equilibrium helpers still present in-repo pending extraction |
| `toroidal.coords` | PEST, Boozer, Hamada, Equal-arc magnetic coordinate systems |
| `toroidal.coils` | Coil geometry, Biot-Savart, RMP coil-set models |
| `toroidal.control` | Topology control: gap response, q-profile response |
| `toroidal.diagnostics` | Plasma diagnostic observables |
| `toroidal.visual` | Publication-quality tokamak figures (`tokamak_manifold`) |
| `toroidal.torus_deformation` | Non-resonant torus deformation (BNF spectral theory, Wei 2025) |

---

## 🔬 Theory

### Non-resonant torus deformation
Under an external perturbation δ**B**, each invariant torus (flux surface) deforms analytically. The displacement field δ**r** = (δr, δθ, δφ) is computed in Fourier space via the formula (Theorem 2 of Wei 2025):

```
(δr)_mn = (δBr)_mn / [i · (mι + n) · Bφ]
```

For axisymmetric (n = 0) poloidal-field coil perturbations the mean radial shift reduces to:

```
⟨δr⟩ = −δι / ι'
```

where δι is the first-order rotational-transform variation. These results are implemented in `pyna.toroidal.torus_deformation`.

### Poincaré maps & manifolds
A Poincaré section φ = φ₀ turns the continuous field-line flow into an area-preserving 2-D map. Near an X-point the stable (W^s) and unstable (W^u) manifolds intersect transversally, generating the heteroclinic tangle responsible for chaotic transport. `pyna.topo` provides algorithms to compute, visualize, and measure these structures.

### Grad-Shafranov equilibrium
Toroidal MHD equilibrium satisfies ΔψGS = −μ₀R²dp/dψ − F dF/dψ. `pyna.toroidal.equilibrium` exposes Solov'ev analytic solutions and a numerical GS solver with free-boundary capability.

---

## 📚 Documentation

Full documentation (API reference, tutorials, theory notes) is hosted at:
**https://wenyinwei.github.io/pyna/**

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/WenyinWei/pyna).

---

## 📄 License

[GPL-3.0-or-later](LICENSE) © 2024-2026 Wenyin Wei
