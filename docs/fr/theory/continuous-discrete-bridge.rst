Géométrie continue et discrète
==============================

pyna utilise des familles d'objets séparées pour les systèmes dynamiques en
temps continu et en temps discret.

Cote temps continu :

- ``Trajectory`` est une géométrie echantillonnee sur un temps fini.
- ``Cycle`` est une orbite périodique d'un flot.
- ``Tube`` est une zone de résonance autour d'un cycle elliptique,
  eventuellement bornee par des cycles hyperboliques.
- ``TubeChain`` regroupe des tubes appartenant à une même résonance.

Côté temps discret :

- ``Orbit`` est une géométrie echantillonnee d'iterations de carte.
- ``PeriodicOrbit`` est une orbite fermee d'une carte.
- ``Island`` est un îlot de résonance reduit sur une section.
- ``IslandChain`` est la chaine d'îlots au niveau de la section.

Le bridge entre les deux cotes est une coupe de section. Couper un ``Cycle`` par
une section de Poincaré produit une ``PeriodicOrbit`` de la carte de retour.
Couper un ``Tube`` produit une ``IslandChain``. Couper une ``TubeChain`` fusionne
les chaines d'îlots issues de ses tubes.

Cette separation est intentionnelle. Une trajectoire numérique peut etre une
géométrie utile sans prouver l'invariance. Les builders et adaptateurs rendent
donc la promotion explicite : les utilisateurs peuvent exiger des controles de
fermeture avant qu'une trajectoire echantillonnee devienne un ``Cycle`` ou avant
que des échantillons de carte deviennent une ``PeriodicOrbit``.

Le même vocabulaire est partage par les systèmes generiques de dimension finie
et par la spécialisation toroidale des lignes de champ magnétique. Les racines
generiques sont disponibles sous les noms ``pyna.topo.CoreTube`` et assimiles ;
les valeurs par défaut toroidales restent disponibles comme ``pyna.topo.Tube``,
``pyna.topo.Cycle`` et ``pyna.topo.IslandChain``.

Voir aussi
----------

- :doc:`/fr/mini-cases`
- :doc:`/notebooks/i18n/fr/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/i18n/fr/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/i18n/fr/tutorials/island_jacobian_analysis`
