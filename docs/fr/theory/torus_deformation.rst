Theorie de deformation du tore
==============================

``pyna.toroidal.torus_deformation`` contient les outils analytiques de
deformation du tore utilisés pour etudier la reponse des tores invariants et des
structures resonantes aux perturbations controlees.

Role conceptuel
---------------

Dans la hierarchie géométrique :

- un tore invariant est un ``InvariantTorus`` ;
- un cycle elliptique resonant est le coeur d'un ``Tube`` ;
- un cycle hyperbolique borne un tube et genere des variétés
  stables/instables ;
- couper des tubes par une section de Poincaré produit des objets
  ``IslandChain``.

Les calculs de deformation du tore alimentent donc directement le contrôle
topologique : ils predisent quelles perturbations spectrales deplacent,
scindent, guerissent ou suppriment les structures resonantes.

API publique
------------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

Modules associes
----------------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
