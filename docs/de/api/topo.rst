Topologie (``pyna.topo``)
=========================

Das Paket ``pyna.topo`` stellt Algorithmen zur Analyse der topologischen
Struktur von Poincaré-Karten bereit: magnetische Inseln, X/O-Zyklen, stabile
und instabile Mannigfaltigkeiten sowie heterokline Verflechtungen.

Das zentrale Design ist eine zweischichtige Hierarchie:

``pyna.topo.core``
   endlichdimensionale, domänenunabhängige Geometrie:
   ``Trajectory``, ``Orbit``, ``PeriodicOrbit``, ``Cycle``, ``Island``,
   ``IslandChain``, ``Tube`` und ``TubeChain``.

``pyna.topo.toroidal``
   Spezialisierungen für magnetische Einschließung, die ``R/Z/phi``,
   Windungszahlen, Monodromy-Nutzdaten, cyna-gestützte Schnitte und toroidale
   Diagnostik hinzufügen.

.. contents:: Submodules
   :depth: 2
   :local:

----

Poincaré-Karten
---------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Geometrieobjekt-Hierarchie
--------------------------

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

Schnitte
--------

.. automodule:: pyna.topo.section
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Poincare-Akkumulatoren
----------------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Magnetische Inseln
------------------

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

Mannigfaltigkeitsberechnung
---------------------------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Periodische Orbits / Zyklen
---------------------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Klassische Abbildungen und Chaosdiagnostik
------------------------------------------

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
