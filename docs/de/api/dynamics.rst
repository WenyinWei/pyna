Allgemeine Dynamik (``pyna.dynamics``)
======================================

``pyna.dynamics`` ist die breite Schicht für dynamische Systeme.  Sie ist
absichtlich klein gehalten und mit ``pyna.topo`` interoperabel:

- aufrufbare ODE-Flüsse mit abgetasteten Trajektorien
- kanonische Hamiltonsche Systeme und separable Hamiltonians
- paarweise gravitative/elektrostatische N-Körper-Systeme
- endlichdimensionale Abbildungen mit Jacobi-Matrizen, Fixpunktresiduen und
  Schätzungen des Lyapunov-Spektrums
- Ito-SDEs, Brownian motion und geometrische Brownian motion

Die Klassen verwenden eine zustandsorientierte Konvention:
``rhs(x, t)`` für Flüsse und ``step(x)`` für Abbildungen.

Geometrieintegration
--------------------

Das Modul gibt dieselben Geometrieklassen zurück, die auch von der toroidalen
Topologie verwendet werden:

- ``TimeSeriesSolution`` ist eine ``pyna.topo.core.Trajectory``.
- ``CallableMap.orbit_geometry`` gibt ``pyna.topo.core.Orbit`` zurück.
- ``CallableMap.periodic_orbit`` gibt ``pyna.topo.core.PeriodicOrbit`` zurück.
- ``pyna.topo.CoreTube`` und ``pyna.topo.CoreIslandChain`` sind die generischen
  endlichdimensionalen Wurzeln; ``pyna.topo.Tube`` bleibt aus Gründen der
  Rückwärtskompatibilität die toroidale Spezialisierung.

Dadurch können Hamiltonsche Systeme, N-Körper-Flüsse, Abbildungen und
SDE-Abtastpfade dasselbe Vokabular ``Cycle``/``Tube``/``IslandChain`` wie die
Topologie magnetischer Feldlinien teilen.

Für Lehrnotebooks oder erweiterungsreiche Workflows siehe
:doc:`dynamics-patterns` zu ``TopologyWorkflow`` und den Low-Level-Hilfen für
Adapter, Builder, Bridges und Factories.

Kontinuierliche Flüsse
----------------------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Hamiltonsche Systeme
--------------------

Verwenden Sie ``HamiltonianSystem``, wenn Sie ``H(q, p, t)`` oder dessen
Gradienten bereitstellen können.  Verwenden Sie ``SeparableHamiltonianSystem``
für ``H(q, p) = T(p) + V(q)`` und Velocity-Verlet-Schritte.

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

N-Körper-Systeme
----------------

``NBodySystem`` speichert abgeflachte Zustandsvektoren als
``[positions.ravel(), velocities.ravel()]`` und stellt Hilfen zum Packen und
Entpacken strukturierter Arrays bereit.  Es unterstützt Newtonsche Gravitation
und elektrostatische Coulomb-Wechselwirkungen.

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

Abbildungen und lokale Mannigfaltigkeiten
-----------------------------------------

``CallableMap`` verarbeitet beliebige endlichdimensionale Abbildungen.
``fixed_point_eigenspaces`` klassifiziert stabile, instabile und zentrale
Eigenräume eines Fixpunkts und ist eine nützliche Brücke zur Konstruktion
lokaler Mannigfaltigkeiten.

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

Stochastische Differentialgleichungen
-------------------------------------

Die SDE-Schicht verwendet die Ito-Form ``dX = a(X,t) dt + B(X,t) dW`` und eine
deterministische Euler-Maruyama-Implementierung für reproduzierbare Forschung
und Lehrbeispiele.  Für Workflows zur Verteilungsschätzung siehe
:doc:`/en/tutorials/sde-monte-carlo`.

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

Verwandte Topologieschicht
--------------------------

Das Topologiepaket hält die abstrakte mathematische Hierarchie und die
Poincare-Maschinerie:

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
