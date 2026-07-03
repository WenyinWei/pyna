Tutorials und Beispiele
=======================

Die öffentlichen Notebooks sind nach Workflow gruppiert.  Der
Dokumentations-Build kopiert ``notebooks/`` in den Sphinx-Quellbaum, daher
spiegeln die folgenden Pfade die Repository-Struktur wider.

Empfohlener Lernpfad
--------------------

Beginnen Sie mit :doc:`/en/quickstart`, arbeiten Sie dann den generischen
Geometrie-Workflow und stochastische Modelle durch und wechseln Sie anschließend
zu toroidalen Monodromy-/RMP-Beispielen:

1. :doc:`/en/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

Allgemeine dynamische Systeme
-----------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

Der generische Geometrie-Workflow und der analytische
Stellarator-Fixpunkt-Workflow sind jetzt in das RMP-Resonanz-Tutorial
integriert, statt als eigenständige textbasierte Notebooks veröffentlicht zu
werden.  Dieses Tutorial zeigt dieselbe Hochstufungskette: abgetastete
Kreuzungen -> Fixpunktgeometrie -> X/O-Klassifikation -> Überlagerungen von
Mannigfaltigkeiten und Koordinatengittern.

Stochastische Differentialgleichungen
-------------------------------------

Das SDE-Tutorial wird lokal vorab ausgeführt, weil Verteilungsschätzungen oft
zehn- oder hunderttausende Monte-Carlo-Pfade verwenden.  GitHub Pages rendert
die gespeicherten Ausgaben, statt CI-Zeit für die schweren Sampling-Zellen zu
verwenden.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

Magnetische Koordinaten und Gleichgewichte
------------------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMPs, Inseln und Poincare-Analyse
---------------------------------

Beginnen Sie mit dem Resonanzanalyse-Notebook, wenn Sie magnetische Topologie
untersuchen.  Es behandelt jetzt divergenzfreie RMP-Templates, den wichtigen
``m=1``-Zweig, ``cyna``-Fixpunktvalidierung, mehrkomponentige kontravariante
``B^r``-Magnetfeldspektren als Atlanten, modulare ``q``/``m/n``-Resonanzkarten
mit optionalen Poincare- und Insel-Overlays, gemischte RMP/nRMP-Spektren, die
gesamte nRMP-Antwort aus allen nichtresonanten Moden,
Feldliniengeschwindigkeitsmodulation und Prüfungen der Störungsordnung.

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` bleibt als Ausführungs-/Cache-Variante des
Resonanzanalyse-Workflows im Repository, aber die öffentliche Dokumentation
verlinkt auf die erklärende Version oben.

Monodromy und Mannigfaltigkeiten
--------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

Klassische und allgemeine dynamische Systeme
--------------------------------------------

Das Repository enthält außerdem leichtgewichtige Notebooks unter
``notebooks/examples``: ``Lorenz_attractor.ipynb``,
``resonance_1_1_map.ipynb``, ``Mobiusian_saddle_cycle.ipynb``,
``Xcycle_construction.ipynb`` und ``FPT_DX_to_DP_sympy.ipynb``.  Sie bleiben
Quellbeispiele statt ausgeführter Dokumentationsseiten, weil mehrere davon
Scratch-artige Notebooks ohne Abschnittstitel sind.

Statische Tutorial-Abbildungen
------------------------------

Mehrere längere Workflows sind im Repository als statische Abbildungen und
generierte Ausgaben unter ``notebooks/tutorials`` vertreten.  Sie behandeln
q-Profil-Diagnostik, PEST/Boozer/Hamada/Equal-Arc-Koordinaten,
Inselunterdrückungsscans, Phasensteuerung, Poincare-Mannigfaltigkeiten und
Solov'ev-Single-Null-Beispiele.
