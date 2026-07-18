# Changelog

## [0.9.0](https://github.com/WenyinWei/pyna/compare/v0.8.18...v0.9.0) (2026-07-18)


### Features

* **boundary:** add reusable field and coil bases ([a7f6974](https://github.com/WenyinWei/pyna/commit/a7f6974da646ffd5d80aafea8865ad472053b9e3))
* **boundary:** compose topology cases and heat targets ([2cc65a4](https://github.com/WenyinWei/pyna/commit/2cc65a4e718f73a833ec157db255b0f3e008976c))
* **control:** add modular boundary response design ([7622b48](https://github.com/WenyinWei/pyna/commit/7622b480d08aaa361ca590a89fef7123bfd3aa1f))
* **control:** add optional Topoquest FPT adapter ([c9df70b](https://github.com/WenyinWei/pyna/commit/c9df70b8180a4c4db26fe3a7819f9ef6e31d4b72))
* **field-lines:** preserve signed native-period maps ([583ac66](https://github.com/WenyinWei/pyna/commit/583ac66b58eeba5ccf3f0e42b76dd9a92cb0d9a2))
* **fields:** centralize toroidal periodicity ([7189e90](https://github.com/WenyinWei/pyna/commit/7189e90939d85cb3cfd93f9c151d934d5589a3c3))
* **heat:** add modular strike and FusionSC backends ([b345753](https://github.com/WenyinWei/pyna/commit/b34575364232260f51ae01da99d5e40e21ec216b))
* **pest:** preserve orientation in candidate-field fits ([26093fc](https://github.com/WenyinWei/pyna/commit/26093fca8b8b85705c5499813e6696636616c3b3))
* **plot:** add lazy boundary topology facades ([62ea609](https://github.com/WenyinWei/pyna/commit/62ea60903607b3178b478e95fa5ab26b7f5942a4))
* **plot:** support adaptive PEST seed tracing ([e5f4f38](https://github.com/WenyinWei/pyna/commit/e5f4f3858ebbf1034a8c1ce94acbbe15ca86a17c))
* **reports:** add synthetic topology audit workflows ([c9f58fe](https://github.com/WenyinWei/pyna/commit/c9f58fe201d153750d18bd4ba8b4bb6ec59d9e9f))
* **spectrum:** bind Nardon diagnostics to field provenance ([cf75f55](https://github.com/WenyinWei/pyna/commit/cf75f5513f3696813a11eadf3e0533456b6a3533))
* **streamlines:** trace PEST face-flux events ([75325d4](https://github.com/WenyinWei/pyna/commit/75325d44618517e13ec1460854d02ef124cc2218))
* **topology:** add nonlinear boundary validation ([3e14c76](https://github.com/WenyinWei/pyna/commit/3e14c7690ed03f20e83a13cac928ff716c5486d0))
* **workflows:** add audited W7-X topology reports ([d881bfd](https://github.com/WenyinWei/pyna/commit/d881bfd15f5b0cd0906c394a442fee06fa8753df))


### Bug Fixes

* **coils:** handle current-loop axis limit ([483e63b](https://github.com/WenyinWei/pyna/commit/483e63bcd963b5da3c8524491b1d51fca3c2a040))
* **pest:** enforce explicit native-period Nfp ([f514707](https://github.com/WenyinWei/pyna/commit/f51470703c6e9855d8bb01741261e8cb8eea6daa))
* **pest:** retain per-surface return scales ([6f388d9](https://github.com/WenyinWei/pyna/commit/6f388d90279b7397749ae3f99d4f61633a42245d))
* **plot:** preserve full surface grid in adaptive traces ([365ac1a](https://github.com/WenyinWei/pyna/commit/365ac1acde7d6293eb6fbdb4578983ff99cd5924))
* **streamlines:** preserve adaptive seed diagnostics ([becc548](https://github.com/WenyinWei/pyna/commit/becc548997d748f236f26692197bdc9ebe2c3cc2))


### Performance Improvements

* **pest:** parallelize surface streamline tracing ([9624eae](https://github.com/WenyinWei/pyna/commit/9624eaedb7ae8b582f60fa546803817e916957e0))


### Documentation

* add localized tutorial notebooks ([b425935](https://github.com/WenyinWei/pyna/commit/b425935e02725cdc8d39fc9dd05b2217eeb1a8fe))
* **quickstart:** fix field-line tracing example ([abd36fb](https://github.com/WenyinWei/pyna/commit/abd36fb5276d0fe12e4d1fcc3799a4dfc86ea0e9))
* refine localized tutorials and pages build ([83a60da](https://github.com/WenyinWei/pyna/commit/83a60daf591aeebfb991cb5aadee9204b7ef690e))
* **tutorial:** lock Nardon spectrum conventions ([e315dbe](https://github.com/WenyinWei/pyna/commit/e315dbe91474f8c6e26cf99ed6b847a44b68e3da))

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
