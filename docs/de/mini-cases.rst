Kurzbeispiele
=============

Diese Seite ist der kurze Weg zwischen dem Schnelleinstieg und der vollständigen
API-Referenz.  Verwenden Sie sie, wenn Sie bereits wissen, welche Art von
System vorliegt, und das kleinste funktionsfähige pyna-Muster suchen.

Welcher Einstiegspunkt?
-----------------------

.. list-table::
   :header-rows: 1

   * - Gegeben ist
     - Beginnen mit
     - Geometrie, die Sie typischerweise erhalten
   * - Eine ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` oder ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory``, danach ggf. ``Cycle``
   * - Ein Hamiltonian ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` oder ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - Eine endlichdimensionale Abbildung ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit``, danach ggf. ``PeriodicOrbit``
   * - Ein toroidales Magnetfeld
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``, ``Tube``, ``IslandChain``, Mannigfaltigkeiten
   * - Ein stochastisches Lehrmodell
     - ``BrownianMotion`` oder ``GeometricBrownianMotion``
     - abgetastete ``Trajectory`` plus Statistiken

Fall 1: ODE-Stichprobe zum geschlossenen Zyklus
-----------------------------------------------

``Trajectory`` bedeutet abgetastete Daten.  ``Cycle`` bedeutet, dass Sie die
stärkere Aussage treffen, dass die Stichprobe geschlossen ist.

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

Halten Sie in Produktions-Workflows die Schließtoleranz explizit.  Dadurch
bleiben numerische Annahmen überprüfbar.

Fall 2: Karteniteration zum periodischen Orbit
----------------------------------------------

Abbildungen erzeugen zunächst ``Orbit``-Objekte.  Stufen Sie nur bekannte oder
numerisch verifizierte geschlossene Stichproben zu ``PeriodicOrbit`` hoch.

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

Wenn Ihre Abbildung aus einem anderen Paket stammt, umhüllen Sie sie entweder
mit ``CallableMap`` oder implementieren Sie ``__call__(x)`` zusammen mit einem
Attribut ``phase_space``.

Fall 3: Analytische Stellarator-O/X-Punkte
------------------------------------------

Für Arbeiten zur magnetischen Einschließung schneidet man den Feldlinienfluss
mit einem Poincaré-Schnitt.  Das ausführbare Tutorial
:doc:`/notebooks/i18n/de/tutorials/RMP_resonance_analysis` enthält jetzt die
vollständige visuelle Rechnung:

1. Aufbau des öffentlichen analytischen Stellarator-Modells;
2. Validierung divergenzfreier ``m=1``- und ``m>1``-RMP-Templates;
3. Verfolgung ungestörter und gestörter Poincaré-Schnitte;
4. Vergleich analytischer resonanter X/O-Phasen mit ``cyna``-Newton-Fixpunkten;
5. Analyse mehrkomponentiger RMP-Spektren mit kontravarianten ``B^r``-
   pcolormesh-Atlanten, ``q``/``m/n``-Resonanzkarten mit optionalen
   Poincaré-Projektionen, interaktiven Plotly-3-D-Balken, radialen Karten mit
   festem ``n``/festem ``m``, Resonanzkurven und umschaltbaren
   Inselbreitenmarkern;
6. Berechnung der gesamten nRMP-Antwort aus allen nichtresonanten Spektralzeilen;
7. Verwendung von Beitragstabellen nur als Diagnostik für Ranking und
   Konvergenz;
8. Visualisierung von nRMP-Flussflächendeformation und Modulation der
   Feldliniengeschwindigkeit;
9. Überlagerung lokaler stabiler Zweige und eines PEST-artigen
   Koordinatengitters.

Verwenden Sie dieses Notebook, wenn Sie Änderungen an Fixpunktplots,
Schnittgeometrie, RMP/nRMP-Diagnostik oder Tutorial-Rendering testen.  Es ist
klein genug, um es vor der Veröffentlichung der Dokumentation lokal auszuführen,
deckt aber dennoch die öffentlichen Hilfs-APIs ab, die von nachgelagerten
Analyseskripten verwendet werden.

Fall 4: Registrierung eigener Systeme
-------------------------------------

Factories sind optional.  Sie werden wichtig, wenn Ihr nachgelagertes Projekt
konfigurationsgetrieben ist.

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

Verwenden Sie in Tests lokale ``Registry``-Instanzen, wenn globale
Registrierung die Testreihenfolge abhängig machen würde.

Fall 5: SDE-Verteilungsschätzung
--------------------------------

Einzelne SDE-Pfade sind pyna-Trajektorien.  Monte-Carlo-Ensembles sind
statistische Schätzer; halten Sie sie als Arrays, bis pyna ein eigenes
Ensemble-Objekt erhält.

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

Ein vollständig ausgeführtes Beispiel mit Verteilungen für brownsche Bewegung,
Ornstein-Uhlenbeck und geometrische brownsche Bewegung finden Sie unter
:doc:`/de/tutorials/sde-monte-carlo`.

Fall 6: Wo anpassen?
--------------------

.. list-table::
   :header-rows: 1

   * - Ziel
     - Erweitern
     - Beachten
   * - Neues physikalisches Modell
     - ``CallableFlow``, ``HamiltonianSystem`` oder Unterklasse von ``ContinuousFlow``
     - pyna-Geometrie aus Integrationsmethoden zurückgeben
   * - Neue Abbildungsfamilie
     - ``CallableMap`` oder Unterklasse von ``DiscreteMap``
     - stabile Koordinatennamen bereitstellen
   * - Neuer Schnitt
     - Objekt im Stil von ``pyna.topo.section.Section``
     - Kreuzungs- und Projektionssemantik klar implementieren
   * - Neues Datenformat
     - ``pyna.topo.adapters``
     - Daten normalisieren; Periodizität nicht stillschweigend behaupten
   * - Neue Assemblierungsregel
     - ``pyna.topo.builders``
     - Validierung und Metadaten zentralisieren
   * - Neue Backend-Auswahl
     - Factories oder Workflow-Fassade
     - rohe Backend-Arrays hinter pyna-Objekten halten

Faustregel: Verwenden Sie Dataclasses für mathematische Objekte, Adapter für
Eingabenormalisierung, Builder für Validierung und Factories nur dann, wenn
Benutzer stabile Zeichenkettenschlüssel benötigen.

Notebook-Checkliste
-------------------

Vor der Veröffentlichung der Dokumentation:

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/i18n/de/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/i18n/de/tutorials/island_jacobian_analysis.ipynb

Für schwere Notebooks mit gespeicherten Ausgaben führen Sie sie lokal aus und
committen Sie die aktualisierte ``.ipynb``-Datei:

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/i18n/de/tutorials/sde_monte_carlo_distribution.ipynb

Für denselben Notebook-Satz, der von GitHub Pages verwendet wird, bauen Sie die
Sphinx-Dokumentation lokal:

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
