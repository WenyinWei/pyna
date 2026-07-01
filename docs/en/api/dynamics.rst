General Dynamics (``pyna.dynamics``)
====================================

``pyna.dynamics`` is the broad dynamical-systems layer.  It is intentionally
small and interoperable with ``pyna.topo``:

- callable ODE flows with sampled trajectories
- canonical Hamiltonian systems and separable Hamiltonians
- pairwise gravitational/electrostatic N-body systems
- finite-dimensional maps with Jacobians, fixed-point residuals and Lyapunov
  spectrum estimates
- Ito SDEs, Brownian motion and geometric Brownian motion

The classes use a state-first convention: ``rhs(x, t)`` for flows and
``step(x)`` for maps.

Geometry Integration
--------------------

The module returns the same geometry classes used by toroidal topology:

- ``TimeSeriesSolution`` is a ``pyna.topo.core.Trajectory``.
- ``CallableMap.orbit_geometry`` returns ``pyna.topo.core.Orbit``.
- ``CallableMap.periodic_orbit`` returns ``pyna.topo.core.PeriodicOrbit``.
- ``pyna.topo.CoreTube`` and ``pyna.topo.CoreIslandChain`` are the generic
  finite-dimensional roots; ``pyna.topo.Tube`` remains the toroidal
  specialization for backward compatibility.

This lets Hamiltonian systems, N-body flows, maps and SDE sample paths share
the same ``Cycle``/``Tube``/``IslandChain`` vocabulary as magnetic field-line
topology.

For teaching notebooks or extension-heavy workflows, see
:doc:`dynamics-patterns` for ``TopologyWorkflow`` and the lower-level adapter,
builder, bridge and factory helpers.

Continuous Flows
----------------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Hamiltonian Systems
-------------------

Use ``HamiltonianSystem`` when you can provide ``H(q, p, t)`` or its gradient.
Use ``SeparableHamiltonianSystem`` for ``H(q, p) = T(p) + V(q)`` and
velocity-Verlet stepping.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import SeparableHamiltonianSystem

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   x1 = oscillator.step_velocity_verlet(np.array([1.0, 0.0]), dt=0.01)

.. automodule:: pyna.dynamics
   :no-index:
   :members: HamiltonianSystem, SeparableHamiltonianSystem
   :show-inheritance:

N-body Systems
--------------

``NBodySystem`` stores flattened state vectors as
``[positions.ravel(), velocities.ravel()]`` and provides helpers to pack and
unpack structured arrays.  It supports Newtonian gravity and electrostatic
Coulomb interactions.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import NBodySystem

   system = NBodySystem([1.0, 1.0], spatial_dim=2, interaction="gravity")
   y0 = system.pack_state(
       positions=np.array([[-1.0, 0.0], [1.0, 0.0]]),
       velocities=np.zeros((2, 2)),
   )
   dy = system.vector_field(y0)

.. automodule:: pyna.dynamics
   :no-index:
   :members: NBodySystem
   :show-inheritance:

Maps and Local Manifolds
------------------------

``CallableMap`` handles arbitrary finite-dimensional maps.  ``fixed_point_eigenspaces``
classifies stable, unstable and center eigenspaces of a fixed point and is a
useful bridge to local manifold construction.

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

Stochastic Differential Equations
---------------------------------

The SDE layer uses Ito form ``dX = a(X,t) dt + B(X,t) dW`` and a deterministic
Euler-Maruyama implementation for reproducible research and teaching examples.
For distribution-estimation workflows, see
:doc:`/en/tutorials/sde-monte-carlo`.

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

Related Topology Layer
----------------------

The topology package keeps the abstract mathematical hierarchy and Poincare
machinery:

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
