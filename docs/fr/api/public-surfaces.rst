Carte des API publiques
=======================

Cette page sert de raccourci vers les interfaces stables de pyna. Les pages
AutoAPI générées restent la référence complète pour le débogage ; les entrées
ci-dessous sont celles à privilégier dans les notebooks, scripts de recherche
et paquets aval.

Vocabulaire géométrique
-----------------------

Commencez ici lorsque l'objet dans l'espace des phases compte davantage que le
solveur qui l'a produit.

.. list-table::
   :header-rows: 1

   * - Tâche
     - Entrées publiques
   * - Mouvement continu échantillonné
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - Dynamique de cartes discrètes
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - Géométrie de section toroïdale
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - Promotion explicite et adaptateurs
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

Systèmes dynamiques généraux
----------------------------

Utilisez :mod:`pyna.dynamics` pour les modèles non toroïdaux qui doivent tout
de même produire les mêmes objets géométriques.

.. list-table::
   :header-rows: 1

   * - Famille de modèles
     - Entrées publiques
   * - Flots ODE
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Systèmes hamiltoniens
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - Systèmes à N corps
     - :class:`pyna.dynamics.NBodySystem`
   * - Cartes discrètes
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - EDS
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

Workflows toroïdaux et RMP
--------------------------

Pour les coordonnées magnétiques, le suivi de lignes de champ, l'analyse
spectrale et les overlays graphiques, utilisez ces modules.

.. list-table::
   :header-rows: 1

   * - Besoin
     - Entrées publiques
   * - Équilibres et coordonnées
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - Suivi de lignes et workflows avec cache
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - Spectre radial contravariant de perturbation
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - Diagnostics RMP/nRMP
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - Figures de spectre magnétique
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Overlays Poincare, X/O et îlots
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

Quand utiliser AutoAPI
----------------------

Utilisez :doc:`/en/api/generated/pyna/index` pour les signatures de
constructeur, membres hérités, diagnostics rares ou détails d'implémentation.
Les nouveaux tutoriels et exemples utilisateur doivent rester autant que
possible sur les entrées publiques ci-dessus.
