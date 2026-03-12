# Changelog

## [0.1.0] - 2026-03-12

### Added
- Solov'ev analytic equilibrium (`pyna.mag.solovev`)
- Analytic stellarator model with helical ripple (`pyna.mag.stellarator`)
- Multi-section Poincaré map infrastructure (`pyna.topo.poincare`)
- Magnetic island extraction from Poincaré data (`pyna.topo.island_extract`)
- Field line tracer with parallel CPU support, CUDA/OpenCL stubs (`pyna.flt`)
- PEST straight-field-line coordinates (`pyna.coord.PEST`)
- Variational equations for Poincaré map derivatives (`pyna.topo.variational`)
- Tutorial notebooks: RMP island validation, stellarator multi-section analysis

### Changed
- `half_width_R` → `half_width_r` in `IslandChain` (minor radius uses lowercase r)

## [0.0.2] - earlier

- Initial dynamical system hierarchy
- Basic vector field classes
