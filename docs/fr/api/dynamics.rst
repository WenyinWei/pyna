Dynamique générale (``pyna.dynamics``)
======================================

``pyna.dynamics`` est la couche large des systèmes dynamiques. Elle est
volontairement compacte et interoperable avec ``pyna.topo`` :

- flots ODE appelables avec trajectoires échantillonnées
- systèmes hamiltoniens canoniques et hamiltoniens separables
- systèmes à N corps gravitationnels/electrostatiques par paires
- cartes de dimension finie avec jacobiens, residus de points fixes et
  estimations du spectre de Lyapunov
- SDE d'Ito, mouvement brownien et mouvement brownien géométrique

Les classes suivent une convention etat-d'abord : ``rhs(x, t)`` pour les flots
et ``step(x)`` pour les cartes.

Integration géométrique
-----------------------

Le module renvoie les memes classes de géométrie que celles utilisées par la
topologie toroidale :

- ``TimeSeriesSolution`` est une ``pyna.topo.core.Trajectory``.
- ``CallableMap.orbit_geometry`` renvoie ``pyna.topo.core.Orbit``.
- ``CallableMap.periodic_orbit`` renvoie ``pyna.topo.core.PeriodicOrbit``.
- ``pyna.topo.CoreTube`` et ``pyna.topo.CoreIslandChain`` sont les racines
  generiques de dimension finie ; ``pyna.topo.Tube`` reste la spécialisation
  toroïdale pour compatibilité ascendante.

Ainsi les systèmes hamiltoniens, flots à N corps, cartes et chemins
échantillonnés de SDE partagent le même vocabulaire
``Cycle``/``Tube``/``IslandChain`` que la topologie des lignes de champ
magnétique.

Pour les notebooks d'enseignement ou les workflows riches en extensions, voir
:doc:`dynamics-patterns` pour ``TopologyWorkflow`` et les auxiliaires
adaptateur, builder, bridge et factory de plus bas niveau.

Flots continus
--------------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Systemes hamiltoniens
---------------------

Utilisez ``HamiltonianSystem`` lorsque vous pouvez fournir ``H(q, p, t)`` ou
son gradient. Utilisez ``SeparableHamiltonianSystem`` pour
``H(q, p) = T(p) + V(q)`` et l'integration velocity-Verlet.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import SeparableHamiltonianSystem

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   x1 = oscillator.step_velocity_verlet(np.array([1.0, 0.0]), dt=0.01)

.. automodule:: pyna.dynamics
   :no-index:
   :members: HamiltonianSystem, SeparableHamiltonianSystem
   :show-inheritance:

Systemes à N corps
------------------

``NBodySystem`` stocke les vecteurs d'etat aplatis sous la forme
``[positions.ravel(), velocities.ravel()]`` et fournit des auxiliaires pour
empaqueter et depaqueter des tableaux structures. Il prend en charge la gravite
newtonienne et les interactions electrostatiques de Coulomb.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import NBodySystem

   system = NBodySystem([1.0, 1.0], spatial_dim=2, interaction="gravity")
   y0 = system.pack_state(
       positions=np.array([[-1.0, 0.0], [1.0, 0.0]]),
       velocities=np.zeros((2, 2)),
   )
   dy = system.vector_field(y0)

.. automodule:: pyna.dynamics
   :no-index:
   :members: NBodySystem
   :show-inheritance:

Cartes et variétés locales
--------------------------

``CallableMap`` gere des cartes arbitraires de dimension finie.
``fixed_point_eigenspaces`` classe les espaces propres stable, instable et
centre d'un point fixe ; c'est un bridge utile vers la construction de variétés
locales.

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

Équations différentielles stochastiques
---------------------------------------

La couche SDE utilise la forme d'Ito ``dX = a(X,t) dt + B(X,t) dW`` et une
implementation deterministe d'Euler-Maruyama pour la recherche reproductible et
les exemples pedagogiques. Pour les workflows d'estimation de distribution,
voir :doc:`/fr/tutorials/sde-monte-carlo`.

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

Couche topologique associee
---------------------------

Le paquet de topologie conserve la hierarchie mathématique abstraite et la
mecanique de Poincaré :

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
