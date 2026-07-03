Mini-cas
========

Cette page est le chemin court entre le démarrage rapide et la référence API
complète. Utilisez-la lorsque vous connaissez déjà le type de système dont vous
disposez et que vous voulez le plus petit motif pyna operationnel.

Quel point d'entree ?
---------------------

.. list-table::
   :header-rows: 1

   * - Vous avez
     - Commencez avec
     - Geometrie habituellement obtenue
   * - Une ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` ou ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory`` puis eventuellement ``Cycle``
   * - Un hamiltonien ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` ou ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - Une application de dimension finie ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit`` puis eventuellement ``PeriodicOrbit``
   * - Un champ magnétique toroïdal
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``, ``Tube``, ``IslandChain``, varietes
   * - Un modèle stochastique pédagogique
     - ``BrownianMotion`` ou ``GeometricBrownianMotion``
     - ``Trajectory`` echantillonnee plus statistiques

Cas 1 : d'un échantillon ODE à un cycle fermé
---------------------------------------------

``Trajectory`` signifie données échantillonnées. ``Cycle`` signifie que vous
formulez l'affirmation plus forte selon laquelle l'échantillon est fermé.

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow

   wf = TopologyWorkflow(closure_tol=2e-2)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(wf.closing_error(traj))
   cycle = wf.closed_cycle(traj)
   print(cycle.period_value, cycle.ambient_dim)

Pour un workflow de production, gardez la tolerance de fermeture explicite.
Cela rend les hypothèses numériques auditables.

Cas 2 : d'une iteration de carte à une orbite périodique
--------------------------------------------------------

Les cartes produisent d'abord des objets ``Orbit``. Ne promouvez vers
``PeriodicOrbit`` que des échantillons fermés connus ou numeriquement vérifiés.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap
   from pyna.topo import TopologyWorkflow

   flip = CallableMap(lambda x: np.array([-x[0], -x[1]]), dim=2)
   wf = TopologyWorkflow(closure_tol=1e-12)

   orbit = wf.orbit(flip, [1.0, 0.0], n_iter=2)
   periodic = wf.periodic_orbit(
       orbit.states[:-1],
       map_obj=flip,
       coordinate_names=("x", "y"),
   )
   print(periodic.period, periodic.points[0].state)

Si votre carte vient d'un autre paquet, enveloppez-la avec ``CallableMap`` ou
implementez ``__call__(x)`` avec un attribut ``phase_space``.

Cas 3 : points O/X analytiques de stellarator
---------------------------------------------

Pour le confinement magnétique, un flot de lignes de champ est coupe par une
section de Poincaré. Le tutoriel executable
:doc:`/notebooks/i18n/fr/tutorials/RMP_resonance_analysis` contient maintenant le
calcul visuel complet :

1. construire le modèle public de stellarator analytique ;
2. valider les gabarits RMP sans divergence ``m=1`` et ``m>1`` ;
3. tracer les sections de Poincaré non perturbees et perturbees ;
4. comparer les phases resonantes X/O analytiques aux points fixes Newton de
   ``cyna`` ;
5. analyser des spectres RMP multi-composants avec des atlas pcolormesh de
   ``B^r`` contravariant, des cartes de résonance ``q``/``m/n`` avec
   projections de Poincaré facultatives, des barres Plotly 3-D interactives,
   des cartes radiales a ``n`` fixe/``m`` fixe, des courbes de résonance et des
   marqueurs de largeur d'îlot activables ;
6. calculer la reponse nRMP totale a partir de toutes les lignes spectrales non
   resonantes ;
7. utiliser les tableaux de contribution uniquement comme diagnostics de
   classement et de convergence ;
8. visualiser la deformation des surfaces de flux nRMP et la modulation de
   vitesse des lignes de champ ;
9. superposer des branches stables locales et une grille de coordonnées de type
   PEST.

Utilisez ce notebook lorsque vous testez des modifications du traçage des points
fixes, de la géométrie de section, des diagnostics RMP/nRMP ou du rendu de
tutoriel. Il est assez petit pour etre exécuté localement avant publication de
la documentation, tout en exercant les API auxiliaires publiques utilisées par
les scripts d'analyse aval.

Cas 4 : enregistrement d'un système personnalise
------------------------------------------------

Les factories sont facultatives. Elles comptent lorsque votre projet aval est
pilote par configuration.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableFlow
   from pyna.topo.factories import DynamicalSystemFactory

   def make_damped_oscillator(gamma=0.1):
       return CallableFlow(
           lambda x, t: np.array([x[1], -x[0] - gamma*x[1]]),
           dim=2,
           coordinate_names=("q", "p"),
           label="damped oscillator",
       )

   DynamicalSystemFactory.register(
       "damped-oscillator",
       lambda gamma=0.1: make_damped_oscillator(gamma),
       overwrite=True,
   )
   flow = DynamicalSystemFactory.create("damped-oscillator", gamma=0.05)

Utilisez des instances locales de ``Registry`` dans les tests si
l'enregistrement global rendrait l'ordre des tests dependant.

Cas 5 : estimation de distribution de SDE
-----------------------------------------

Les chemins SDE individuels sont des trajectoires pyna. Les ensembles Monte
Carlo sont des estimateurs statistiques ; conservez-les sous forme de tableaux
jusqu'a ce que pyna dispose d'un objet d'ensemble dedie.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1)
   print(path.final)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=100_000)
   terminal = 100.0 * np.exp(gbm.expected_log_growth()[0] + gbm.sigma[0] * z)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

Pour un cas complet exécuté avec des distributions de mouvement brownien,
d'Ornstein-Uhlenbeck et de mouvement brownien géométrique, utilisez
:doc:`/fr/tutorials/sde-monte-carlo`.

Cas 6 : ou personnaliser
------------------------

.. list-table::
   :header-rows: 1

   * - Objectif
     - Etendre
     - A garder en tete
   * - Nouveau modele physique
     - ``CallableFlow``, ``HamiltonianSystem`` ou sous-classe ``ContinuousFlow``
     - retourner de la géométrie pyna depuis les méthodes d'intégration
   * - Nouvelle famille de cartes
     - ``CallableMap`` ou sous-classe ``DiscreteMap``
     - exposer des noms de coordonnées stables
   * - Nouvelle section
     - objet de style ``pyna.topo.section.Section``
     - implementer clairement les semantiques de croisement/projection
   * - Nouveau format de données
     - ``pyna.topo.adapters``
     - normaliser les données ; ne pas revendiquer silencieusement la périodicité
   * - Nouvelle politique d'assemblage
     - ``pyna.topo.builders``
     - centraliser validation et metadonnees
   * - Nouvelle selection de backend
     - factories ou facade de workflow
     - garder les tableaux bruts de backend derriere les objets pyna

Règle pratique : utilisez des dataclasses pour les objets mathématiques, des
adaptateurs pour la normalisation des entrees, des builders pour la validation,
et des factories seulement lorsque les utilisateurs ont besoin de cles de chaine
stables.

Checklist pour notebooks
------------------------

Avant de publier la documentation :

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

Pour les notebooks lourds avec sorties sauvegardees, executez-les localement et
committez le fichier ``.ipynb`` mis a jour :

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

Pour le même ensemble de notebooks que celui utilise par GitHub Pages,
construisez Sphinx localement :

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
