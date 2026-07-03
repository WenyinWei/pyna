Workflows de dynamique et auxiliaires d'extension
=================================================

``pyna.topo`` expose des auxiliaires de workflow autour des objets
topologiques de base. Le principal point d'entree pour les utilisateurs est
``TopologyWorkflow`` ; les modules Protocol, Adapter, Builder, Bridge et
Factory de plus bas niveau restent disponibles pour les bibliothèques aval qui
ont besoin de points d'extension stables.

Facade de workflow
------------------

``TopologyWorkflow`` est concu pour les notebooks et les scripts quotidiens. Il
combine construction du système, integration/iteration, promotion explicite et
coupes de section sans ajouter de nouveau type d'objet mathématique.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocoles
----------

Les protocoles structurels decrivent les contrats d'extension pour les systèmes
externes. Les objets tiers peuvent participer en implementant les attributs et
methodes requis ; sous-classer les classes pyna est facultatif.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adaptateurs
-----------

Les adaptateurs normalisent tableaux, sorties de solveur et objets pyna
existants vers les representations geometriques de base. Ils ne promeuvent pas
silencieusement les données échantillonnées en objets invariants.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builders
--------

Les builders encodent des règles de promotion explicites. Par exemple, une
trajectoire ne peut etre promue en ``Cycle`` que par un appel de builder ou
d'adaptateur capable d'exiger des échantillons fermés.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridges
-------

Les bridges relient les familles d'objets en temps continu et en temps discret :
``Cycle -> PeriodicOrbit`` et ``Tube/TubeChain -> IslandChain``.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factories et registres
----------------------

Les factories fournissent des points d'entree de construction stables pour les
systèmes, la géométrie et les cartes de Poincaré. Les registres sont explicites
et protégés contre les doublons afin que les tests et bibliothèques aval
puissent isoler leurs propres extensions.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
