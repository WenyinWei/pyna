Mini Cases
==========

This page is the short path between the quickstart and the full API reference.
Use it when you already know the kind of system you have and want the smallest
working pyna pattern.

Which Entry Point?
------------------

.. list-table::
   :header-rows: 1

   * - You have
     - Start with
     - Geometry you usually get
   * - An ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` or ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory`` then possibly ``Cycle``
   * - A Hamiltonian ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` or ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - A finite-dimensional map ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit`` then possibly ``PeriodicOrbit``
   * - A toroidal magnetic field
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``, ``Tube``, ``IslandChain``, manifolds
   * - A stochastic teaching model
     - ``BrownianMotion`` or ``GeometricBrownianMotion``
     - sampled ``Trajectory`` plus statistics

Case 1: ODE Sample to Closed Cycle
----------------------------------

``Trajectory`` means sampled data.  ``Cycle`` means you are making the stronger
claim that the sample is closed.

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow

   wf = TopologyWorkflow(closure_tol=2e-2)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(wf.closing_error(traj))
   cycle = wf.closed_cycle(traj)
   print(cycle.period_value, cycle.ambient_dim)

For a production workflow, keep the closing tolerance explicit.  It makes
numerical assumptions reviewable.

Case 2: Map Iterate to Periodic Orbit
-------------------------------------

Maps produce ``Orbit`` objects first.  Promote to ``PeriodicOrbit`` only for
known or numerically verified closed samples.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap
   from pyna.topo import TopologyWorkflow

   flip = CallableMap(lambda x: np.array([-x[0], -x[1]]), dim=2)
   wf = TopologyWorkflow(closure_tol=1e-12)

   orbit = wf.orbit(flip, [1.0, 0.0], n_iter=2)
   periodic = wf.periodic_orbit(
       orbit.states[:-1],
       map_obj=flip,
       coordinate_names=("x", "y"),
   )
   print(periodic.period, periodic.points[0].state)

If your map comes from another package, either wrap it with ``CallableMap`` or
implement ``__call__(x)`` plus a ``phase_space`` attribute.

Case 3: Analytic Stellarator O/X Points
---------------------------------------

For magnetic-confinement work, a field-line flow is cut by a Poincare section.
The executable tutorial :doc:`/notebooks/tutorials/analytic_stellarator_geometry_workflow`
does the complete calculation:

1. build the public analytic stellarator model;
2. integrate the period-4 return map;
3. classify O/X points from the monodromy;
4. promote the result to ``PeriodicOrbit`` and ``IslandChain``.

Use this notebook when testing changes to fixed-point algorithms or section
geometry.  It is small enough to run locally before publishing docs.

Case 4: Custom System Registration
----------------------------------

Factories are optional.  They matter when your downstream project is
configuration-driven.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableFlow
   from pyna.topo.factories import DynamicalSystemFactory

   def make_damped_oscillator(gamma=0.1):
       return CallableFlow(
           lambda x, t: np.array([x[1], -x[0] - gamma*x[1]]),
           dim=2,
           coordinate_names=("q", "p"),
           label="damped oscillator",
       )

   DynamicalSystemFactory.register(
       "damped-oscillator",
       lambda gamma=0.1: make_damped_oscillator(gamma),
       overwrite=True,
   )
   flow = DynamicalSystemFactory.create("damped-oscillator", gamma=0.05)

Use local ``Registry`` instances in tests if global registration would make the
test order-dependent.

Case 5: Where to Customize
--------------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - Extend
     - Keep in mind
   * - New physics model
     - ``CallableFlow``, ``HamiltonianSystem`` or subclass ``ContinuousFlow``
     - return pyna geometry from integration methods
   * - New map family
     - ``CallableMap`` or subclass ``DiscreteMap``
     - expose stable coordinate names
   * - New section
     - ``pyna.topo.section.Section`` style object
     - implement crossing/project semantics clearly
   * - New data format
     - ``pyna.topo.adapters``
     - normalize data; do not silently claim periodicity
   * - New assembly policy
     - ``pyna.topo.builders``
     - centralize validation and metadata
   * - New backend selection
     - factories or workflow facade
     - keep raw backend arrays behind pyna objects

The rule of thumb: use dataclasses for mathematical objects, adapters for input
normalization, builders for validation, and factories only when users need
stable string keys.

Notebook Checklist
------------------

Before publishing documentation:

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/general_dynamics_geometry_patterns.ipynb \
     notebooks/tutorials/analytic_stellarator_geometry_workflow.ipynb

For the same notebook set used by GitHub Pages, run the Sphinx build locally:

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
