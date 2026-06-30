Architecture
============

pyna is organised around two ideas:

1. dynamical systems define evolution rules on finite-dimensional phase spaces;
2. topology modules describe geometric objects living in those phase spaces.

This separation lets the same object hierarchy represent toroidal magnetic
field-line structures, Hamiltonian resonance zones, classical maps, N-body
orbits and stochastic sample paths.

Layer 0: Dynamics
-----------------

``pyna.topo.dynamics`` provides the abstract mathematical layer:

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` and ``GeneralPoincareMap``

``pyna.dynamics`` adds ready-to-use finite-dimensional systems:

- ``CallableFlow`` and ``CallableMap``
- ``HamiltonianSystem`` and ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``, ``BrownianMotion`` and ``GeometricBrownianMotion``

These classes use the topology core for sampled outputs.  A deterministic flow
trajectory is a ``pyna.topo.core.Trajectory``; a discrete iterate cloud is a
``pyna.topo.core.Orbit``.

Layer 1: Geometry
-----------------

``pyna.topo.core`` is the domain-agnostic geometry hierarchy:

.. list-table::
   :header-rows: 1

   * - Class
     - Meaning
     - Time type
   * - ``Trajectory``
     - finite sampled curve in phase space
     - continuous
   * - ``Cycle``
     - periodic orbit of a continuous flow
     - continuous
   * - ``Tube``
     - resonance zone around an elliptic cycle
     - continuous
   * - ``TubeChain``
     - family of tubes sharing one resonance
     - continuous
   * - ``Orbit``
     - finite sampled iterates of a map
     - discrete
   * - ``PeriodicOrbit``
     - finite periodic orbit of a map
     - discrete
   * - ``Island``
     - one resonance island on a section
     - discrete
   * - ``IslandChain``
     - periodic chain of islands on a section
     - discrete

The key bridge is ``section_cut``:

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

This mirrors the toroidal workflow where continuous magnetic island tubes are
observed as discrete island chains on a Poincare section.

Layer 2: Toroidal Specialization
--------------------------------

``pyna.topo.toroidal`` subclasses the generic core:

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

The toroidal layer adds:

- ``R``, ``Z`` and ``phi`` coordinates
- winding numbers ``(m, n)``
- ``DPm`` and monodromy classification
- cyna-accelerated section cuts and tracing
- section-view correspondence and reconstruction helpers

Layer 3: Workflow and Extension Helpers
---------------------------------------

``pyna.topo.protocols``, ``adapters``, ``builders``, ``bridges`` and
``factories`` provide the software-engineering extension layer.  The main
notebook-facing entry point is ``TopologyWorkflow``.  These helpers keep
construction policy and backend selection outside the mathematical dataclasses:
external systems can conform by protocol, normalize data with adapters, promote
objects through builders, cut continuous geometry through bridges, and select
runtime implementations through factories.

Layer 4: Acceleration
---------------------

``cyna`` implements the bottlenecks behind high-level pyna APIs.  It should not
own high-level scientific object semantics; it supplies fast kernels for
tracing, interpolation, fixed-point scans, wall hits and perturbation response.

Design Rules
------------

- Prefer generic ``pyna.topo.core`` classes for new finite-dimensional geometry.
- Add toroidal-specific fields only in ``pyna.topo.toroidal`` subclasses.
- A sampled finite trajectory is geometry, not automatically an invariant set.
- Promote objects to ``Cycle``/``PeriodicOrbit`` only when a periodic structure
  is part of the model or has been validated numerically.
- Keep cyna at bridge boundaries; application-level APIs should return pyna
  objects, not raw C++ arrays.
