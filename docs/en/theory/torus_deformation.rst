Torus Deformation Theory
========================

``pyna.toroidal.torus_deformation`` contains the analytic torus-deformation
tools used to study how invariant tori and resonant structures respond to
controlled perturbations.

Conceptual Role
---------------

In the geometry hierarchy:

- an invariant torus is an ``InvariantTorus``;
- a resonant elliptic cycle is the core of a ``Tube``;
- a hyperbolic cycle bounds a tube and generates stable/unstable manifolds;
- cutting tubes with a Poincare section produces ``IslandChain`` objects.

Torus-deformation calculations therefore feed directly into topology control:
they predict which spectral perturbations move, split, heal or suppress
resonant structures.

Public API
----------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

Related Modules
---------------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
