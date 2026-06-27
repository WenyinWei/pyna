# pyna -- Python DYNAmics

<p align="center">
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/pyna-chaos/"><img src="https://img.shields.io/pypi/pyversions/pyna-chaos" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-LGPL--3.0-green" alt="License"></a>
  <a href="https://wenyinwei.github.io/pyna/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://github.com/WenyinWei/pyna/actions"><img src="https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/WenyinWei/pyna/actions/workflows/notebook-tests.yml"><img src="https://github.com/WenyinWei/pyna/actions/workflows/notebook-tests.yml/badge.svg" alt="Notebook Tests"></a>
</p>

**pyna** (**Py**thon **D**Y**NA**mics) is a research library for **dynamical systems** and **magnetic confinement fusion (MCF)** plasma physics. It covers the full workflow from analytic equilibria and field-line tracing to topological island analysis, manifold visualization, and non-resonant torus deformation theory.

> **Author:** [Wenyin Wei](https://github.com/WenyinWei) · **PyPI:** `pyna-chaos` · **Julia companion:** [Juna.jl](https://github.com/WenyinWei/Juna.jl)

Production magnetic-field-line tracing belongs in `pyna`/`cyna`; downstream
projects should call these tracing APIs instead of adding Python RK4 hot loops.
See [FIELD_LINE_TRACING_POLICY.md](FIELD_LINE_TRACING_POLICY.md).

FPT naming convention: `progress_*` means a fixed source phase `phi_s` and a
moving endpoint `phi_e`; `evolve_*_cycle_*` means a cycle-attached object whose
base phase moves, with `phi_s` and `phi_e = phi_s + 2*pi*m` moving together.
This distinction is part of the public API vocabulary.

---

## ✨ Highlights

| Feature | Details |
|---------|---------|
| 🔀 **Field-line tracing** | RK4 integrator with required `cyna` CPU backend; CUDA builds are local opt-in |
| 🌀 **Poincaré sections & island chains** | Multi-section crossing accumulation, island-chain extraction, X/O-point analysis |
| 🗺️ **Manifold visualization** | Publication-quality stable/unstable manifold plots for tokamaks |
| 🧲 **Toroidal geometry** | Coordinates, coils, diagnostics, and field-line tooling |
| 📐 **Magnetic coordinates** | PEST, Boozer, Hamada, Equal-arc transformations |
| 📡 **Torus deformation** | Non-resonant BNF-derived analytic spectral theory (Wei 2025) |
| ⚡ **C++ tracing core** | Required `cyna` backend for production field-line and Poincare tracing |

---

## 📦 Installation

```bash
# Stable release (PyPI)
pip install pyna-chaos

# Development version
git clone https://github.com/WenyinWei/pyna.git
cd pyna
pip install -e ".[dev]"

# Optional Python-side GPU dependencies (CUDA 12 CuPy stack)
pip install "pyna-chaos[cuda]"
```

### cyna C++ tracing core

`pyna._cyna` is part of the supported runtime surface, not an optional extra.
PyPI releases are built as platform wheels for Linux, Windows, and macOS across
CPython 3.9, 3.10, 3.11, 3.12, and 3.13. On those platforms, `pip install
pyna-chaos` should install a wheel with `_cyna_ext` already included.

If no wheel matches your platform, pip falls back to the source distribution and
builds `cyna` locally with xmake:

```bash
pip install pyna-chaos
python -c "import pyna._cyna as c; print(c.is_available())"
```

For development installs:

```bash
git clone https://github.com/WenyinWei/pyna.git
cd pyna
pip install -e ".[dev]"
python -c "import pyna._cyna as c; assert c.is_available(); print(c.get_version())"
```

Source builds require a C++17 compiler, pybind11 headers, and xmake. If xmake or
a compiler is missing, `setup.py` attempts to bootstrap a minimal toolchain on
Windows, macOS, and Linux before building. Set `CYNA_SKIP_TOOL_INSTALL=1` in
controlled CI images where xmake/compiler provisioning is handled externally.
An install that cannot build or load `_cyna_ext` fails; this prevents downstream
projects from silently running without the production tracing backend.

Published PyPI wheels are CPU-only and do not link against CUDA; the wheel CI
sets `CYNA_WITH_CUDA=0` explicitly. For local source builds, leaving
`CYNA_WITH_CUDA` unset attempts to build an optional runtime CUDA backend when
`nvcc` is available. The main `_cyna_ext` module remains CPU-only and falls back
cleanly if that backend is absent. Set `CYNA_WITH_CUDA=0` to force a CPU-only
local build, or `CYNA_WITH_CUDA=1` to require the CUDA backend build. Check
`pyna._cyna.cuda_backend_available()` at runtime when you need to know whether
the optional backend loaded.

---

## 🚀 Quick Start

### Field-line tracing

```python
from pyna.toroidal.equilibrium.Solovev import solovev_iter_like
from pyna.flt import FieldLineTracer
import numpy as np

eq = solovev_iter_like()

# FieldLineTracer expects f([R, Z, phi]) → [BR, BZ, Bphi]
def solovev_field(y):
    R, Z, phi = y
    BR, BZ = eq.BR_BZ(R, Z)
    return np.array([BR, BZ, eq.Bphi(R)])

tracer = FieldLineTracer(solovev_field)
trajectory = tracer.trace(np.array([2.0, 0.0, 0.0]), 200 * 2 * np.pi)
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
    plot_manifold_1d,
)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 7))
plot_equilibrium_cross_section(fig, ax, eq)
plot_poincare_orbits(fig, ax, orbits, cmap_by='psi_norm')
plot_manifold_1d(fig, ax, stable_manifold_RZ, unstable=False)
plot_manifold_1d(fig, ax, unstable_manifold_RZ, unstable=True)
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
split_radial_perturbation_spectrum = toroidal.split_radial_perturbation_spectrum

# Define perturbation spectrum (m, n, δBr, δBθ, δBφ in T·m)
m = np.array([1, 2, 3])
n = np.array([1, 1, 1])
dBr = np.array([1e-4+0j, 5e-5+0j, 2e-5+0j])
dBth = np.zeros(3, dtype=complex)
dBph = np.zeros(3, dtype=complex)

split = split_radial_perturbation_spectrum(m, n, dBr, iota=0.35)
spec = split.nonresonant_deformation(
    iota=0.35, Bphi=4.5, Btheta=0.3,
    dBth_mn=dBth, dBph_mn=dBph,
)
mean_dr = mean_radial_displacement(delta_iota=1e-4, iota_prime=-0.1)
print(f"Mean radial displacement: {mean_dr:.4f} m")
```

---

## 📂 Module Overview

### Core dynamical systems

| Module | Description |
|--------|-------------|
| `pyna.system` | Abstract dynamical system hierarchy (`DynamicalSystem`, `VectorField*D`) |
| `pyna.flt` | Field-line tracer: RK4 and parallel CPU production backend |
| `pyna.topo.toroidal_island` | Rational surface location, theoretical island half-width |
| `pyna.topo.poincare` | Multi-section crossing accumulation and section helpers |
| `pyna.topo.manifold` | Stable/unstable manifold computation |
| `pyna.topo.toroidal_cycle` | Periodic orbit (X/O cycle) detection and analysis |
| `pyna.toroidal.coords` | Flux coordinate transformations (PEST, Boozer, Hamada, Equal-arc) |
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

where δι is the first-order rotational-transform variation. Resonant B^r Fourier
components (`mι+n=0`) are separated for island-width analysis, while
non-resonant components drive smooth flux-surface deformation. These results are
implemented in `pyna.toroidal.torus_deformation`.

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

[LGPL-3.0-or-later](LICENSE) © 2024-2026 Wenyin Wei
