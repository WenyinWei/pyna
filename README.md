# pyna — Python DYNAmics

**pyna** is a Python library for dynamical systems analysis with a focus on magnetic confinement fusion (MCF) physics. It provides tools for building analytic equilibria, tracing magnetic field lines, constructing Poincaré maps, and extracting topological structures such as magnetic islands.

Cross-language companion: [Juna.jl](https://github.com/WenyinWei/Juna.jl) (Julia).

---

## Installation

```bash
# Stable release
pip install pyna-chaos

# Development version
git clone https://github.com/WenyinWei/pyna.git
cd pyna
pip install -e .
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `pyna.system` | Abstract dynamical system hierarchy (`DynamicalSystem`, `VectorField*D`) |
| `pyna.flt` | Field line tracer (RK4, parallel CPU, CUDA/OpenCL stubs) |
| `pyna.mag.equilibrium` | Axisymmetric equilibrium ABC + `SyntheticCircularTokamakEquilibrium` |
| `pyna.mag.solovev` | Solov'ev analytic equilibrium (Cerfon & Freidberg 2010) |
| `pyna.mag.stellarator` | Analytic helical-ripple stellarator |
| `pyna.mag.rmp` | RMP spectrum analysis, island width at rational surfaces |
| `pyna.topo.island` | Rational surface location, theoretical island half-width |
| `pyna.topo.island_extract` | O/X point extraction from Poincaré data (`IslandChain`) |
| `pyna.topo.poincare` | Multi-section Poincaré map infrastructure |
| `pyna.coord.PEST` | PEST straight-field-line coordinate system (if implemented) |

---

## Quick-Start Examples

### 1. Build a Solov'ev equilibrium and plot flux surfaces

```python
from pyna.mag.solovev import SolovevEquilibrium
import matplotlib.pyplot as plt

# Cerfon & Freidberg (2010) ITER-like configuration
eq = SolovevEquilibrium.iter_like()

fig, ax = plt.subplots()
eq.plot_flux_surfaces(ax=ax, levels=20)
ax.set_title("Solov'ev equilibrium — flux surfaces")
plt.show()
```

### 2. Trace field lines and build a Poincaré map

```python
from pyna.mag.solovev import SolovevEquilibrium
from pyna.topo.poincare import PoincareMap

eq = SolovevEquilibrium.iter_like()

pmap = PoincareMap(eq, phi_section=0.0)
result = pmap.trace(
    R0_list=[6.0, 6.5, 7.0],  # starting major radii [m]
    Z0=0.0,
    n_turns=200,
    n_workers=4,               # parallel CPU
)
pmap.plot(result)
```

### 3. Extract island chain from Poincaré data

```python
from pyna.topo.island_extract import IslandChain

# result from PoincareMap.trace()
chain = IslandChain.from_poincare(result, mode=(4, 1))
print(f"Island half-width r = {chain.half_width_r:.4f} m")
chain.plot_oxpoints()
```

---

## Academic Conventions

pyna follows standard physics and fusion-community naming to make the code immediately recognisable to domain experts:

| Symbol / Name | Meaning | Rationale |
|--------------|---------|-----------|
| `RMP` | Resonant Magnetic Perturbation | All-caps acronym (standard in literature) |
| `PEST` | Projection-Equation Straight-field-line coordinates | Original acronym preserved |
| `Boozer` | Boozer coordinates | Proper noun — author's name |
| `B` | Magnetic field magnitude | Standard physics notation |
| `b` | Unit vector along **B** | Standard physics notation |
| `r` | Minor radius | Lowercase distinguishes from major radius `R` |

---

## Roadmap

- [ ] Boozer coordinate system (`pyna.coord.Boozer`)
- [ ] GPU acceleration (CUDA via `cupy`, `pyna.flt` backend)
- [ ] Lyapunov exponent spectrum
- [ ] Bifurcation diagrams and curves
- [ ] Stable/unstable manifold drawing
- [ ] Two-dimensional map flows (eigenvalue/eigenvector analysis)

---

## Citation

If you use pyna in your research, please cite the relevant physics references:

- Cerfon & Freidberg, *Phys. Plasmas* **17**, 032502 (2010) — Solov'ev equilibrium
- Your own paper (pyna has no formal publication yet)

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
