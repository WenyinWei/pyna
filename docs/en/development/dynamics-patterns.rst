Dynamics Workflows and Extension Helpers
========================================

pyna separates mathematical geometry from construction policy.

The core hierarchy stays compact:

- continuous-time geometry: ``Trajectory``, ``Cycle``, ``Tube``,
  ``TubeChain``;
- discrete-time geometry: ``Orbit``, ``PeriodicOrbit``, ``Island``,
  ``IslandChain``;
- toroidal classes remain the default public topology specializations under
  ``pyna.topo.Tube``, ``pyna.topo.Cycle`` and ``pyna.topo.IslandChain``.

The helper layer adds one user-facing workflow facade plus explicit extension
points around that hierarchy.

Workflow Facade
---------------

``TopologyWorkflow`` is the recommended first stop for tutorials and analysis
scripts.  It composes the lower-level helpers into the path users actually
follow:

1. build or receive a flow/map;
2. integrate a ``Trajectory`` or iterate an ``Orbit``;
3. explicitly promote closed samples to ``Cycle`` or ``PeriodicOrbit``;
4. cut ``Cycle``/``Tube``/``TubeChain`` objects by a section.

The facade is intentionally thin.  It does not introduce new mathematics; it
keeps the notebook code readable while still making each promotion explicit.

Worked Tutorial
---------------

For a compact workflow overview, start with :doc:`/en/mini-cases`.  For a
complete visual tutorial that applies the same promotion ideas to a real
toroidal calculation, use :doc:`/notebooks/tutorials/RMP_resonance_analysis`.
It shows sampled Poincare crossings, explicit X/O fixed-point geometry,
coordinate-grid overlays and local manifold branches.

For short copy-paste recipes, use :doc:`/en/mini-cases`.  That page is the
intended bridge between the quickstart and the full API reference.

Protocols
---------

``pyna.topo.protocols`` defines structural contracts such as ``FlowLike``,
``MapLike``, ``SectionLike`` and ``TubeLike``.  Use these when adding a new
domain package that should interoperate with pyna without inheriting every base
class directly.

Adapters
--------

``pyna.topo.adapters`` converts user data into stable core objects:

- arrays or solver outputs to ``Trajectory`` and ``Orbit``;
- points or fixed-point-like objects to ``SectionPoint``;
- verified samples to ``PeriodicOrbit`` or ``Cycle`` when requested.

Adapters normalize representation; they should not hide mathematical claims.
For example, an open sampled trajectory remains a ``Trajectory`` unless a
caller explicitly asks for a ``Cycle`` and accepts or passes the closure check.

Builders
--------

``GeometryBuilder``, ``IslandChainBuilder`` and ``TubeChainBuilder`` capture
construction policy.  Prefer builders when a workflow assembles topology from
several lower-level ingredients, because they centralize validation, metadata
and back-links.

Bridges
-------

``CoreSectionCutBridge`` is the default continuous-to-discrete bridge for core
objects:

- ``Cycle.section_cut(section)`` returns a ``PeriodicOrbit``;
- ``Tube.section_cut(section)`` returns an ``IslandChain``;
- ``TubeChain.section_cut(section)`` merges the resulting islands.

Toroidal objects already own optimized ``section_cut`` methods.  Use them
directly or call ``TopologyWorkflow.section_cut(...)`` and let the object
dispatch its own implementation.

Factories
---------

``DynamicalSystemFactory`` builds ready-to-use systems from stable string keys
such as ``callable-flow``, ``callable-map``, ``hamiltonian``, ``nbody`` and
``geometric-brownian-motion``.

``PoincareMapFactory`` chooses an executable return-map implementation.  The
default ``backend="auto"`` currently selects the portable
``GeneralPoincareMap`` unless cyna field-cache arguments are supplied.

``GeometryFactory`` builds topology geometry through the builder layer.  It is
useful for config-driven examples and downstream packages that need stable
construction keys.

Compatibility Rules
-------------------

- Do not change ``pyna.topo.Tube``, ``Cycle`` or ``IslandChain`` to point at
  the core classes; use ``CoreTube``, ``CoreCycle`` and ``CoreIslandChain`` for
  generic roots.
- Do not use duck-typed fake sections at toroidal-only boundaries.  Use first
  class ``Section`` objects.
- Treat registries as mutable state.  Use local ``Registry`` instances in tests
  and downstream packages when isolation matters.
