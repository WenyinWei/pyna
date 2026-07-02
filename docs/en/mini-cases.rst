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
The executable tutorial :doc:`/notebooks/tutorials/RMP_resonance_analysis`
now carries the complete visual calculation:

1. build the public analytic stellarator model;
2. validate divergence-free ``m=1`` and ``m>1`` RMP templates;
3. trace unperturbed and perturbed Poincare sections;
4. compare analytic resonant X/O phases with ``cyna`` Newton fixed points;
5. compute the total nRMP response from all non-resonant spectrum rows;
6. use contribution tables only as diagnostics for ranking and convergence;
7. visualise nRMP flux-surface deformation and field-line speed modulation;
8. overlay local stable branches and a PEST-style coordinate grid.

Use this notebook when testing changes to fixed-point plotting, section
geometry, RMP/nRMP diagnostics, or tutorial rendering.  It is small enough to
run locally before publishing docs, while still exercising the public helper
APIs used by downstream analysis scripts.

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

Case 5: SDE Distribution Estimate
---------------------------------

Single SDE paths are pyna trajectories.  Monte Carlo ensembles are statistical
estimators; keep them as arrays until pyna grows a dedicated ensemble object.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1)
   print(path.final)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=100_000)
   terminal = 100.0 * np.exp(gbm.expected_log_growth()[0] + gbm.sigma[0] * z)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

For a full executed case with Brownian, Ornstein-Uhlenbeck and geometric
Brownian motion distributions, use :doc:`/en/tutorials/sde-monte-carlo`.

Case 6: Where to Customize
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
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

For heavy notebooks with saved outputs, run them locally and commit the updated
``.ipynb`` file:

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

For the same notebook set used by GitHub Pages, run the Sphinx build locally:

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
