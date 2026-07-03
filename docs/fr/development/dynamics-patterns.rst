Workflows de dynamique et auxiliaires d'extension
=================================================

pyna sépare la géométrie mathématique de la politique de construction.

La hierarchie centrale reste compacte :

- géométrie en temps continu : ``Trajectory``, ``Cycle``, ``Tube``,
  ``TubeChain`` ;
- géométrie en temps discret : ``Orbit``, ``PeriodicOrbit``, ``Island``,
  ``IslandChain`` ;
- les classes toroidales restent les spécialisations topologiques publiques par
  défaut sous ``pyna.topo.Tube``, ``pyna.topo.Cycle`` et
  ``pyna.topo.IslandChain``.

La couche d'auxiliaires ajoute une facade de workflow orientee utilisateur et
des points d'extension explicites autour de cette hierarchie.

Facade de workflow
------------------

``TopologyWorkflow`` est le premier point d'entree recommande pour les
tutoriels et les scripts d'analyse. Il compose les auxiliaires de plus bas
niveau selon le chemin que les utilisateurs suivent effectivement :

1. construire ou recevoir un flot/une carte ;
2. integrer une ``Trajectory`` ou iterer une ``Orbit`` ;
3. promouvoir explicitement les échantillons fermés en ``Cycle`` ou
   ``PeriodicOrbit`` ;
4. couper les objets ``Cycle``/``Tube``/``TubeChain`` par une section.

La facade est volontairement mince. Elle n'introduit pas de nouvelle
mathématique ; elle garde le code de notebook lisible tout en rendant chaque
promotion explicite.

Tutoriel travaille
------------------

Pour un apercu compact du workflow, commencez par :doc:`/fr/mini-cases`. Pour
un tutoriel visuel complet appliquant les memes idees de promotion à un calcul
toroidal reel, utilisez :doc:`/notebooks/tutorials/RMP_resonance_analysis`. Il
montre des croisements de Poincaré échantillonnés, une géométrie explicite de
points fixes X/O, des superpositions de grilles de coordonnées et des branches
de variétés locales.

Pour des recettes courtes à copier-coller, utilisez :doc:`/fr/mini-cases`.
Cette page est le bridge prevu entre le démarrage rapide et la référence API
complète.

Protocoles
----------

``pyna.topo.protocols`` definit des contrats structurels tels que ``FlowLike``,
``MapLike``, ``SectionLike`` et ``TubeLike``. Utilisez-les lors de l'ajout d'un
nouveau paquet de domaine qui doit interoperer avec pyna sans heriter
directement de chaque classe de base.

Adaptateurs
-----------

``pyna.topo.adapters`` convertit les données utilisateur en objets de coeur
stables :

- tableaux ou sorties de solveur vers ``Trajectory`` et ``Orbit`` ;
- points ou objets de type point fixe vers ``SectionPoint`` ;
- échantillons vérifiés vers ``PeriodicOrbit`` ou ``Cycle`` sur demande.

Les adaptateurs normalisent la representation ; ils ne doivent pas cacher les
affirmations mathématiques. Par exemple, une trajectoire echantillonnee ouverte
reste une ``Trajectory`` sauf si l'appelant demande explicitement un ``Cycle`` et
accepte ou fournit le contrôle de fermeture.

Builders
--------

``GeometryBuilder``, ``IslandChainBuilder`` et ``TubeChainBuilder`` capturent la
politique de construction. Preferez les builders lorsqu'un workflow assemble la
topologie a partir de plusieurs ingredients de plus bas niveau, car ils
centralisent validation, métadonnées et liens de retour.

Bridges
-------

``CoreSectionCutBridge`` est le bridge continu-vers-discret par défaut pour les
objets du coeur :

- ``Cycle.section_cut(section)`` renvoie une ``PeriodicOrbit`` ;
- ``Tube.section_cut(section)`` renvoie une ``IslandChain`` ;
- ``TubeChain.section_cut(section)`` fusionne les îlots resultants.

Les objets toroidaux possedent déjà des methodes ``section_cut`` optimisees.
Utilisez-les directement ou appelez ``TopologyWorkflow.section_cut(...)`` et
laissez l'objet dispatcher sa propre implementation.

Factories
---------

``DynamicalSystemFactory`` construit des systèmes prêts à l'emploi a partir de
cles de chaine stables telles que ``callable-flow``, ``callable-map``,
``hamiltonian``, ``nbody`` et ``geometric-brownian-motion``.

``PoincareMapFactory`` choisit une implementation executable de carte de retour.
Le ``backend="auto"`` par défaut selectionne actuellement le
``GeneralPoincareMap`` portable sauf si des arguments de cache de champ cyna
sont fournis.

``GeometryFactory`` construit la géométrie topologique via la couche builder.
Elle est utile pour les exemples pilotes par configuration et les paquets aval
qui ont besoin de cles de construction stables.

Règles de compatibilité
-----------------------

- Ne faites pas pointer ``pyna.topo.Tube``, ``Cycle`` ou ``IslandChain`` vers
  les classes du coeur ; utilisez ``CoreTube``, ``CoreCycle`` et
  ``CoreIslandChain`` pour les racines generiques.
- N'utilisez pas de fausses sections par duck typing aux frontières purement
  toroïdales. Utilisez des objets ``Section`` de première classe.
- Traitez les registres comme un etat mutable. Utilisez des instances locales
  de ``Registry`` dans les tests et les paquets aval lorsque l'isolation compte.
