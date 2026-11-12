# Changelog

## [0.10.0](https://github.com/WenyinWei/pyna/compare/v0.9.11...v0.10.0) (2026-11-12)


### Features

* **cyna:** add CPU coil-field grid builder ([dec2fa9](https://github.com/WenyinWei/pyna/commit/dec2fa93b8227e5effa293b350ec84717cbe5282))
* **cyna:** expose axisymmetric tau closed orbits and analytic tangent responses ([6caa051](https://github.com/WenyinWei/pyna/commit/6caa051433a3ee9427b091f84c5ead04fb6cde95))
* **cyna:** make CPU tracing the default without CUDA ([611dde1](https://github.com/WenyinWei/pyna/commit/611dde14d9c781592f35fbfa7e2600a7e77f46fc))
* **flt:** add differentiable JAX Poincare map ([0003c84](https://github.com/WenyinWei/pyna/commit/0003c84e2c17bfc9e9ba92ba4ea5eb4db2201b68))
* **topo:** add differentiable manifold branches ([a5a1f2e](https://github.com/WenyinWei/pyna/commit/a5a1f2e91886cce55bf043e0fe7d82e7053d34a4))
* **topo:** add differentiable wall intersections ([adb16b1](https://github.com/WenyinWei/pyna/commit/adb16b1d66b2b5f2c56f6f36fa2873d9c0e25dc8))
* **topo:** add implicit periodic-point solver ([4c882f2](https://github.com/WenyinWei/pyna/commit/4c882f2acd3e989daeb9a3a071b6dd2c98e04786))
* **topo:** add production manifold refresh gate ([631a491](https://github.com/WenyinWei/pyna/commit/631a491dabecf5d2b6faecd57a013b4267d2fd5d))
* **topo:** compare labelled strike bundles ([f1a8d21](https://github.com/WenyinWei/pyna/commit/f1a8d21b9496fd1ce0fb80e32ea08b585dca20ea))
* **topo:** distinguish tokamak and 3D strike metrics ([da4c6fc](https://github.com/WenyinWei/pyna/commit/da4c6fc1482d144be4af0d96f300e7ad9fdcd3ef))
* **topo:** freeze local wall planes at strikes ([cff18c7](https://github.com/WenyinWei/pyna/commit/cff18c7f0a917f17987bef3a31b529dfd8da1eb6))
* **topo:** preserve first-wall strike labels ([54156e1](https://github.com/WenyinWei/pyna/commit/54156e1d6799e9a43d541084d9e406eb97f5e0fc))
* **topo:** preserve manifold seed orders on retrace ([a5b51c1](https://github.com/WenyinWei/pyna/commit/a5b51c1a0dec3f910201488b2e0dbfc805be04c2))
* **topo:** trace differentiable periodic orbits ([3520d12](https://github.com/WenyinWei/pyna/commit/3520d122994bff746905b14fd6f2e7d1e6f2531e))
* **topo:** trace full labelled strike legs ([69d0836](https://github.com/WenyinWei/pyna/commit/69d0836f36b88d8646ea8047f342aa4fc5707334))
* **topo:** trace labelled manifold strikes ([9b2f4ab](https://github.com/WenyinWei/pyna/commit/9b2f4abafc64d7f96c3615e63e09b3407c75a5b2))
* **topo:** trace labelled strike bundles ([13032cf](https://github.com/WenyinWei/pyna/commit/13032cfb22de3c3249a427428094a53bf5c50d97))
* **topo:** track manifold sample correspondence ([69ba3f3](https://github.com/WenyinWei/pyna/commit/69ba3f373347628ca3d97e29098f390124a114d7))
* **topo:** validate differentiable strike parity ([b6362e9](https://github.com/WenyinWei/pyna/commit/b6362e9a2ee5c771d169912135626b3268566ce5))
* **topo:** validate manifold heat loading ([3a3ae33](https://github.com/WenyinWei/pyna/commit/3a3ae333fc8b3faf35ca58d084f42abb3c251d57))
* **topo:** validate manifold label refreshes ([1640d66](https://github.com/WenyinWei/pyna/commit/1640d66445a85e5d931549051a305b241d847908))
* **topo:** validate pre-strike manifold leg clearance ([4f032d7](https://github.com/WenyinWei/pyna/commit/4f032d7128cbc5edd62350b42c1015587c0d684c))
* **topo:** validate production X-line clearance ([509492f](https://github.com/WenyinWei/pyna/commit/509492f997e5febfb34bfcb657a55c18d301164d))


### Bug Fixes

* **integration:** preserve Python compatibility and native build contracts ([1ec89fe](https://github.com/WenyinWei/pyna/commit/1ec89febf6e957c277dce8aa1896734b51a1a986))
* **packaging:** build and test supported CPython wheels across platforms ([d9eec72](https://github.com/WenyinWei/pyna/commit/d9eec72ed9e7799b8ae393254ab12b2c69de0508))
* **release:** seed manifest from the current package version ([4f7478c](https://github.com/WenyinWei/pyna/commit/4f7478c2393ea61dcc0a53d8207fb6785fa455eb))
* **topo:** make periodic-root sensitivities explicit ([8ac3b79](https://github.com/WenyinWei/pyna/commit/8ac3b79527869cf4ee8e932301005afb885f32af))
* **topo:** preserve clearance sign across nfp seams ([afdb50f](https://github.com/WenyinWei/pyna/commit/afdb50fb034006e6b48cf1cc1d68a46b7cb72e08))
* **topo:** preserve unwrapped wall-hit phase ([87f57dc](https://github.com/WenyinWei/pyna/commit/87f57dc5ef3538ea639291407fe092c8b37cc649))
* **topo:** separate eigenvector diagnostics from gradients ([ceab1f5](https://github.com/WenyinWei/pyna/commit/ceab1f56ca3d2a8f36cc5ba932b3d2f57cfe9be3))
* **wheels:** resolve binary dependencies for manylinux test images ([cad1481](https://github.com/WenyinWei/pyna/commit/cad1481ad49ebe8f372b1981a75a1575c6490713))

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
