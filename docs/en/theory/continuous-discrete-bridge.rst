Continuous and Discrete Geometry
================================

pyna uses separate object families for continuous-time and discrete-time
dynamical systems.

Continuous-time side:

- ``Trajectory`` is sampled finite-time geometry.
- ``Cycle`` is a periodic orbit of a flow.
- ``Tube`` is a resonance zone around an elliptic cycle, possibly bounded by
  hyperbolic cycles.
- ``TubeChain`` groups tubes belonging to one resonance.

Discrete-time side:

- ``Orbit`` is sampled map iteration geometry.
- ``PeriodicOrbit`` is a closed orbit of a map.
- ``Island`` is one reduced resonance island on a section.
- ``IslandChain`` is the section-level chain of islands.

The bridge between the two sides is a section cut.  Cutting a ``Cycle`` by a
Poincare section produces a ``PeriodicOrbit`` of the return map.  Cutting a
``Tube`` produces an ``IslandChain``.  Cutting a ``TubeChain`` merges the
island chains from its tubes.

This separation is intentional.  A numerical trajectory can be useful geometry
without proving invariance.  Builders and adapters therefore make promotion
explicit: users can require closure checks before a sampled trajectory becomes
a ``Cycle`` or before map samples become a ``PeriodicOrbit``.

The same vocabulary is shared by generic finite-dimensional systems and by the
toroidal magnetic-field-line specialization.  Generic roots are available as
``pyna.topo.CoreTube`` and related names; toroidal defaults remain available as
``pyna.topo.Tube``, ``pyna.topo.Cycle`` and ``pyna.topo.IslandChain``.

See Also
--------

- :doc:`/notebooks/tutorials/general_dynamics_geometry_patterns`
- :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/tutorials/island_jacobian_analysis`
