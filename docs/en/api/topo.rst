Topology (``pyna.topo``)
========================

The ``pyna.topo`` package provides algorithms for analysing the topological
structure of Poincaré maps: magnetic islands, X/O cycles, stable/unstable
manifolds, and heteroclinic tangles.

The central design is a two-layer hierarchy:

``pyna.topo.core``
   finite-dimensional, domain-agnostic geometry:
   ``Trajectory``, ``Orbit``, ``PeriodicOrbit``, ``Cycle``, ``Island``,
   ``IslandChain``, ``Tube`` and ``TubeChain``.

``pyna.topo.toroidal``
   magnetic-confinement specializations that add ``R/Z/phi``, winding numbers,
   monodromy payloads, cyna-backed section cuts and toroidal diagnostics.

.. contents:: Submodules
   :depth: 2
   :local:

----

Poincaré Maps
-------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Geometry Object Hierarchy
-------------------------

.. automodule:: pyna.topo._base
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.core
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Sections
--------

.. automodule:: pyna.topo.section
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Poincare Accumulators
---------------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Magnetic Islands
----------------

.. automodule:: pyna.topo.toroidal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.toroidal_island
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Manifold Computation
---------------------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Periodic Orbits / Cycles
-------------------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Classical Maps and Chaos Diagnostics
------------------------------------

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.chaos
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
