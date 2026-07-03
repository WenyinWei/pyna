Distributions SDE par Monte Carlo
=================================

Ce tutoriel montre le workflow SDE pratique dans pyna :

1. définir un modèle d'Itô avec ``BrownianMotion``,
   ``GeometricBrownianMotion`` ou ``ItoSDE`` ;
2. générer un chemin échantillon reproductible comme ``Trajectory`` ;
3. exécuter un ensemble Monte Carlo vectorisé pour estimer les distributions ;
4. comparer moyenne empirique, variance et quantiles aux formules analytiques
   lorsqu'elles sont disponibles.

Utilisez les classes SDE de pyna pour la frontière du modèle et pour la
géométrie d'un chemin unique. Utilisez des tableaux NumPy vectorisés pour les
grands ensembles jusqu'à ce que pyna dispose d'une classe de géométrie
d'ensemble dédiée. Cela garde le modèle d'objets mathématiques honnête : une
réalisation unique est une trajectoire échantillonnée, tandis qu'un nuage de
réalisations est un estimateur statistique.

.. note::

   Le notebook exécutable ci-dessous est versionné avec ses sorties sauvegardées
   et l'exécution ``nbsphinx`` désactivée. Réexécutez-le localement lorsque vous
   modifiez les paramètres numériques ; le workflow de documentation rendra ces
   sorties sauvegardées sur GitHub Pages.

Notebook exécutable :

- :doc:`/notebooks/i18n/fr/tutorials/sde_monte_carlo_distribution`

Motif à copier-coller
---------------------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution est une Trajectory pyna

   n_paths = 200_000
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=n_paths)
   log_terminal = (
       np.log(100.0)
       + gbm.expected_log_growth()[0] * 1.0
       + gbm.sigma[0] * np.sqrt(1.0) * z
   )
   terminal = np.exp(log_terminal)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

Notes d'extension
-----------------

- ``ItoSDE.diffusion_matrix`` accepte une diffusion scalaire, vectorielle ou
  matricielle.
- ``ItoSDE.euler_maruyama`` accepte des incréments ``dW`` fournis de
  l'extérieur, de sorte que les expériences à nombres aléatoires communs et les
  tests de régression puissent être déterministes.
- Ne promouvez un chemin échantillon unique vers des objets topologiques que
  lorsque l'affirmation géométrique a un sens. Les échantillons Monte Carlo
  estiment des distributions ; ils ne sont pas automatiquement des ensembles
  invariants.
