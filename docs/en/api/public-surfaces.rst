Public API Map
==============

Use this page as the short route into pyna's stable interfaces.  The generated
AutoAPI pages remain the complete debug reference; the entries below are the
interfaces intended for notebooks, scripts and downstream packages.

Geometry Vocabulary
-------------------

Start here when the object in phase space matters more than the solver that
created it.

.. list-table::
   :header-rows: 1

   * - Task
     - Public entry points
   * - Sampled continuous motion
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - Sampled map dynamics
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - Toroidal section geometry
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - Explicit promotion and adapters
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

General Dynamics
----------------

Use :mod:`pyna.dynamics` for non-toroidal models that should still return the
same geometry objects.

.. list-table::
   :header-rows: 1

   * - Model family
     - Public entry points
   * - ODE flows
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Hamiltonian systems
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N-body systems
     - :class:`pyna.dynamics.NBodySystem`
   * - Discrete maps
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - SDEs
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

Toroidal and RMP Workflows
--------------------------

Use these modules for magnetic coordinates, field-line tracing, magnetic
spectrum analysis and visual overlays.

.. list-table::
   :header-rows: 1

   * - Need
     - Public entry points
   * - Equilibria and coordinates
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - Field-line tracing and cache-aware workflows
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - Contravariant radial perturbation spectrum
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - RMP/nRMP tutorial diagnostics
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - Magnetic-spectrum figures
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Poincare, X/O and island overlays
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

When to Drop to AutoAPI
-----------------------

Use :doc:`generated/pyna/index` when you need constructor signatures, inherited
members, rarely used diagnostics or private implementation details.  Keep new
tutorials and user-facing examples on the public entries above whenever
possible.
