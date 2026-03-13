# pyna — Python DYNAmics

<p align="center">
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/pyversions/pyna-chaos" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License"></a>
  <a href="https://wenyinwei.github.io/pyna/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://github.com/WenyinWei/pyna/actions"><img src="https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg" alt="CI"></a>
</p>

**pyna** (**Py**thon **D**Y**NA**mics) is a research library for **dynamical systems** and **magnetic confinement fusion (MCF)** plasma physics. It covers the full workflow from analytic equilibria and field-line tracing to topological island analysis, manifold visualization, and non-resonant torus deformation theory.

> **Author:** [Wenyin Wei](https://github.com/WenyinWei) · **PyPI:** `pyna-chaos` · **Julia companion:** [Juna.jl](https://github.com/WenyinWei/Juna.jl)

---

## ✨ Highlights

| Feature | Details |
|---------|---------|
| 🔀 **Field-line tracing** | RK4 integrator, parallel CPU, optional CUDA (118× speedup) |
| 🌀 **Poincaré maps** | Multi-section maps, X/O-point detection, island width extraction |
| 🗺️ **Manifold visualization** | Publication-quality stable/unstable manifold plots for tokamaks |
| 🧲 **MCF equilibria** | Solov'ev, Grad-Shafranov, stellarator analytic solutions |
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
from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium
from pyna.flt import FieldLineTracer

eq = SolovevEquilibrium.iter_like()
tracer = FieldLineTracer(eq.Bfield)
trajectory = tracer.trace(R0=2.0, Z0=0.0, phi_end=200 * 2 * 3.14159)
```

### Poincaré map

```python
from pyna.topo.poincare import PoincareMap

pmap = PoincareMap(tracer, sections=[0.0])
orbits = pmap.compute(seeds_R=[1.8, 2.0, 2.2], seeds_Z=[0.0, 0.0, 0.0], n_turns=500)
pmap.plot(orbits)
```

### EAST tokamak manifold visualization

```python
from pyna.MCF.visual.tokamak_manifold import (
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
from pyna.MCF.torus_deformation import (
    non_resonant_deformation_spectrum,
    poincare_section_deformation,
    mean_radial_displacement,
)

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
| `pyna.topo.island` | Rational surface location, theoretical island half-width |
| `pyna.topo.poincare` | Multi-section Poincaré map infrastructure |
| `pyna.topo.manifold` | Stable/unstable manifold computation |
| `pyna.topo.cycle` | Periodic orbit (X/O cycle) detection and analysis |
| `pyna.coord` | Flux coordinate transformations |
| `pyna.draw` | High-level plotting utilities |
| `pyna.gc` | Guiding-centre orbit integration |
| `pyna.interact` | Interactive widgets (Jupyter) |
| `pyna.utils` | Miscellaneous helpers |

### Magnetic Confinement Fusion (`pyna.MCF`)

| Submodule | Description |
|-----------|-------------|
| `MCF.equilibrium` | Solov'ev, Grad-Shafranov, stellarator analytic/numeric equilibria |
| `MCF.coords` | PEST, Boozer, Hamada, Equal-arc magnetic coordinate systems |
| `MCF.coils` | Coil geometry, Biot-Savart, RMP coil-set models |
| `MCF.control` | Topology control: gap response, q-profile response |
| `MCF.diagnostics` | Plasma diagnostic observables |
| `MCF.plasma_response` | Perturbed GS solver for plasma response |
| `MCF.visual` | Publication-quality tokamak figures (`tokamak_manifold`) |
| `MCF.torus_deformation` | Non-resonant torus deformation (BNF spectral theory, Wei 2025) |

---

## 🔬 Theory

### Non-resonant torus deformation
Under an external perturbation δ**B**, each invariant torus (flux surface) deforms analytically. The displacement field δ**r** = (δr, δθ, δφ) is computed in Fourier space via the formula (Theorem 2 of Wei 2025):

```
(δr)_mn = −(δBr)_mn / [(mι + n) · Bφ]
```

For axisymmetric (n = 0) poloidal-field coil perturbations the mean radial shift reduces to:

```
⟨δr⟩ = −δι / ι'
```

where δι is the first-order rotational-transform variation. These results are implemented in `pyna.MCF.torus_deformation`.

### Poincaré maps & manifolds
A Poincaré section φ = φ₀ turns the continuous field-line flow into an area-preserving 2-D map. Near an X-point the stable (W^s) and unstable (W^u) manifolds intersect transversally, generating the heteroclinic tangle responsible for chaotic transport. `pyna.topo` provides algorithms to compute, visualize, and measure these structures.

### Grad-Shafranov equilibrium
Toroidal MHD equilibrium satisfies ΔψGS = −μ₀R²dp/dψ − F dF/dψ. `pyna.MCF.equilibrium` exposes Solov'ev analytic solutions and a numerical GS solver with free-boundary capability.

---

## 📚 Documentation

Full documentation (API reference, tutorials, theory notes) is hosted at:
**https://wenyinwei.github.io/pyna/**

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/WenyinWei/pyna).

---

## 📄 License

[GPL-3.0-or-later](LICENSE) © 2024–2026 Wenyin Wei
