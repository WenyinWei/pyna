# Changelog

## [Unreleased]

## [0.8.22] - 2026-07-01

### Added
- Added VMEC mgrid IO helpers with cylindrical ``J = curl(B) / mu0`` current
  density evaluation.
- Added smooth-PEST current-component diagnostics for ``J^rho``, ``J^theta``,
  ``J^phi`` and ``J^theta/J^phi`` sign-reversal analysis.
- Added smooth-PEST surface Fourier ripple diagnostics for high-poloidal-mode
  shape content.
- Added reusable mgrid current-density plotting helpers and
  ``scripts/plot_mgrid_current_diagnostics.py`` for reproducible comparison
  figures from mgrid files plus precomputed smooth PEST coordinates.

## [0.8.20] - 2026-06-30

### Changed
- Made Prefect a hard dependency for field-line trajectory and wall post-compute
  flows instead of silently falling back to plain functions.
- Hardened toroidal wall trace cache validation with field signatures and
  explicit handling for legacy unsigned cache files.
- Added Python 3.9 dependency constraints needed by Prefect 3.x flow schema
  generation.

## [0.8.19] - 2026-06-30

### Added
- Boundary-island orbit APIs with map-order labels, same-orbit multi-section
  Poincare backgrounds, and post-compute/cache helpers for reusable orbit data.
- Toroidal geometry wrappers for structured field-period objects and section
  plotting helpers.
- Regression coverage for manifold growth, field-period seams, reverse tracing,
  dense trajectories, checkpoint/resume, and cyna field grid validation.

### Changed
- Renamed discrete-map boundary island cycle terminology to orbit terminology.
- Hardened fixed-point manifold tracing, stable inverse-map anchoring,
  monodromy span checks, and compact section plotting guards.

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
