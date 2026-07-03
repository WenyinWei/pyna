SDE-Monte-Carlo-Verteilungen
============================

Dieses Tutorial zeigt den praktischen SDE-Workflow in pyna:

1. ein Itô-Modell mit ``BrownianMotion``, ``GeometricBrownianMotion`` oder
   ``ItoSDE`` definieren;
2. einen reproduzierbaren Abtastpfad als ``Trajectory`` erzeugen;
3. ein vektorisiertes Monte-Carlo-Ensemble für Verteilungsschätzungen
   ausführen;
4. empirischen Mittelwert, Varianz und Quantile mit analytischen Formeln
   vergleichen, wenn diese verfügbar sind.

Verwenden Sie pyna-SDE-Klassen für die Modellgrenze und für Geometrie einzelner
Pfade.  Verwenden Sie vektorisierte NumPy-Arrays für große Ensembles, bis pyna
eine eigene Ensemble-Geometrieklasse erhält.  So bleibt das mathematische
Objektmodell ehrlich: Eine einzelne Realisierung ist eine abgetastete
Trajektorie, während eine Wolke von Realisierungen ein statistischer Schätzer
ist.

.. note::

   Das ausführbare Notebook unten ist mit gespeicherten Ausgaben committed und
   hat die Ausführung durch ``nbsphinx`` deaktiviert.  Führen Sie es lokal
   erneut aus, wenn Sie numerische Parameter ändern; der Docs-Workflow rendert
   diese gespeicherten Ausgaben auf GitHub Pages.

Ausführbares Notebook:

- :doc:`/notebooks/i18n/de/tutorials/sde_monte_carlo_distribution`

Copy-Paste-Muster
-----------------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution is a pyna Trajectory

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

Erweiterungshinweise
--------------------

- ``ItoSDE.diffusion_matrix`` akzeptiert skalare, vektorielle oder
  matrixwertige Diffusion.
- ``ItoSDE.euler_maruyama`` akzeptiert extern bereitgestellte ``dW``-
  Inkremente, sodass Experimente mit common random numbers und Regressionstests
  deterministisch sein können.
- Stufen Sie einen einzelnen Abtastpfad nur dann über Topologieobjekte hoch,
  wenn die geometrische Aussage sinnvoll ist.  Monte-Carlo-Stichproben schätzen
  Verteilungen; sie sind nicht automatisch invariante Mengen.
