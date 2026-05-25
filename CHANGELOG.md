# Changelog

## [0.9.0](https://github.com/WenyinWei/pyna/compare/v0.8.5...v0.9.0) (2026-05-25)


### Features

* add beta-scan poincare grid plot ([28ed2f7](https://github.com/WenyinWei/pyna/commit/28ed2f75c0f5a9e1c582f2ad5406fdc82859dcbb))
* overlay walls in beta Poincare grids ([dda270e](https://github.com/WenyinWei/pyna/commit/dda270e46e1a2a348ebd452d44c732f049780980))


### Documentation

* update ARCHITECTURE.md and README.md for topoquest v0.2.2 ([c0d309b](https://github.com/WenyinWei/pyna/commit/c0d309b7b18fddd1a38c931679edb0279a9476a0))

## [Unreleased]

### Added
- `pyna.topo.chaos`: skeleton module for chaotic region diagnostics — Chirikov
  overlap criterion (`chirikov_overlap`), finite-time Lyapunov exponent field
  (`ftle_field`), and chaotic boundary estimation (`chaotic_boundary_estimate`).
- `pyna.toroidal.optimize`: new subpackage for multi-objective stellarator
  optimisation; includes `neoclassical_epsilon_eff`, `xpoint_field_parallelism`,
  `magnetic_axis_position`, `wall_clearance`, and `compute_all_objectives`
  (all stubs with full docstrings and TODO markers ready for implementation).
- Updated `pyna.topo.__init__` to re-export chaos diagnostics.

## [0.4.1] - 2026-03-19
### Changed
- Added Island and IslandChain class hierarchy with parallel execution support
- Removed development/debug temporary files from repository root and scripts/

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
