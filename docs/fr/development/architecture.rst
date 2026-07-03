Architecture
============

pyna est organise autour de deux idees :

1. les systèmes dynamiques définissent des règles d'evolution sur des espaces
   des phases de dimension finie ;
2. les modules de topologie decrivent des objets geometriques vivant dans ces
   espaces des phases.

Cette separation permet a la même hierarchie d'objets de representer des
structures de lignes de champ magnétique toroidales, des zones de résonance
hamiltoniennes, des cartes classiques, des orbites à N corps et des chemins
échantillonnés stochastiques.

Couche 0 : dynamique
--------------------

``pyna.topo.dynamics`` fournit la couche mathématique abstraite :

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` et ``GeneralPoincareMap``

``pyna.dynamics`` ajoute des systèmes de dimension finie prêts à l'emploi :

- ``CallableFlow`` et ``CallableMap``
- ``HamiltonianSystem`` et ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``, ``BrownianMotion`` et ``GeometricBrownianMotion``

Ces classes utilisent le coeur topologique pour les sorties échantillonnées. Une
trajectoire de flot deterministe est une ``pyna.topo.core.Trajectory`` ; un
nuage d'itérations discret est une ``pyna.topo.core.Orbit``.

Couche 1 : géométrie
--------------------

``pyna.topo.core`` est la hierarchie géométrique independante du domaine :

.. list-table::
   :header-rows: 1

   * - Classe
     - Signification
     - Type de temps
   * - ``Trajectory``
     - courbe echantillonnee finie dans l'espace des phases
     - continu
   * - ``Cycle``
     - orbite periodique d'un flot continu
     - continu
   * - ``Tube``
     - zone de résonance autour d'un cycle elliptique
     - continu
   * - ``TubeChain``
     - famille de tubes partageant une résonance
     - continu
   * - ``Orbit``
     - iterations echantillonnees finies d'une carte
     - discret
   * - ``PeriodicOrbit``
     - orbite periodique finie d'une carte
     - discret
   * - ``Island``
     - un îlot de résonance sur une section
     - discret
   * - ``IslandChain``
     - chaîne périodique d'îlots sur une section
     - discret

Le bridge cle est ``section_cut`` :

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

Cela reflete le workflow toroidal ou les tubes continus d'îlots magnétiques
sont observes comme des chaines d'îlots discretes sur une section de Poincaré.

Couche 2 : spécialisation toroidale
-----------------------------------

``pyna.topo.toroidal`` sous-classe le coeur generique :

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

La couche toroidale ajoute :

- les coordonnées ``R``, ``Z`` et ``phi``
- les nombres d'enroulement ``(m, n)``
- ``DPm`` et la classification de monodromie
- les coupes de section et le traçage accélérés par cyna
- la correspondance de vues de section et les auxiliaires de reconstruction

Couche 3 : workflow et auxiliaires d'extension
----------------------------------------------

``pyna.topo.protocols``, ``adapters``, ``builders``, ``bridges`` et
``factories`` fournissent la couche d'extension d'ingenierie logicielle. Le
principal point d'entree cote notebooks est ``TopologyWorkflow``. Ces auxiliaires
gardent la politique de construction et la selection de backend en dehors des
dataclasses mathématiques : les systèmes externes peuvent se conformer par
protocole, normaliser les données avec des adaptateurs, promouvoir les objets
via des builders, couper la géométrie continue via des bridges et choisir les
implementations d'exécution via des factories.

Couche 4 : accélération
-----------------------

``cyna`` implemente les goulets d'etranglement derriere les API pyna de haut
niveau. Il ne doit pas posseder la semantique scientifique des objets de haut
niveau ; il fournit des noyaux rapides pour le traçage, l'interpolation, les
balayages de points fixes, les impacts sur paroi et la reponse aux
perturbations.

Regles de conception
--------------------

- Preferer les classes generiques ``pyna.topo.core`` pour toute nouvelle
  géométrie de dimension finie.
- Ajouter les champs spécifiques au toroidal uniquement dans les sous-classes
  ``pyna.topo.toroidal``.
- Une trajectoire finie echantillonnee est une géométrie, pas automatiquement un
  ensemble invariant.
- Promouvoir des objets en ``Cycle``/``PeriodicOrbit`` uniquement lorsqu'une
  structure périodique fait partie du modèle ou a été validée numeriquement.
- Garder cyna aux frontières de bridge ; les API applicatives doivent renvoyer
  des objets pyna, pas des tableaux C++ bruts.
