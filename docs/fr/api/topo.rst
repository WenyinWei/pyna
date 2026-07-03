Topologie (``pyna.topo``)
=========================

Le paquet ``pyna.topo`` fournit des algorithmes d'analyse de la structure
topologique des cartes de Poincaré : îlots magnétiques, cycles X/O, variétés
stables/instables et enchevetrements heterocliniques.

La conception centrale est une hierarchie a deux couches :

``pyna.topo.core``
   géométrie de dimension finie independante du domaine :
   ``Trajectory``, ``Orbit``, ``PeriodicOrbit``, ``Cycle``, ``Island``,
   ``IslandChain``, ``Tube`` et ``TubeChain``.

``pyna.topo.toroidal``
   spécialisations pour le confinement magnétique qui ajoutent ``R/Z/phi``, les
   nombres d'enroulement, les charges utiles de monodromie, les coupes de
   section appuyees par cyna et les diagnostics toroidaux.

.. contents:: Sous-modules
   :depth: 2
   :local:

----

Cartes de Poincaré
------------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Hierarchie des objets geometriques
----------------------------------

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

Accumulateurs de Poincaré
-------------------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Îlots magnétiques
-----------------

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

Calcul de variétés
------------------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Orbites periodiques / cycles
----------------------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Cartes classiques et diagnostics du chaos
-----------------------------------------

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
