Anleitungen und Beispiele
=========================

Die öffentlichen Notebooks sind nach Arbeitsablauf gruppiert.  Der
Dokumentationsaufbau kopiert ``notebooks/`` in den Sphinx-Quellbaum, daher
spiegeln die folgenden Pfade die Repository-Struktur wider.

Empfohlener Lernpfad
--------------------

Beginnen Sie mit :doc:`/de/quickstart`, arbeiten Sie dann den generischen
Geometrie-Arbeitsablauf und stochastische Modelle durch und wechseln Sie anschließend
zu toroidalen Monodromie-/RMP-Beispielen:

1. :doc:`/de/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/de/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/de/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/de/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/de/tutorials/RMP_island_validation_solovev`

Allgemeine dynamische Systeme
-----------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

Der generische Geometrie-Arbeitsablauf und der analytische
Stellarator-Fixpunkt-Arbeitsablauf sind jetzt in das RMP-Resonanz-Tutorial
integriert, statt als eigenständige textbasierte Notebooks veröffentlicht zu
werden.  Dieses Tutorial zeigt dieselbe Hochstufungskette: von abgetasteten
Kreuzungen über Fixpunktgeometrie und X/O-Klassifikation bis zu
Überlagerungen von Mannigfaltigkeiten und Koordinatengittern.

Stochastische Differentialgleichungen
-------------------------------------

Die SDE-Anleitung wird lokal vorab ausgeführt, weil Verteilungsschätzungen oft
zehn- oder hunderttausende Monte-Carlo-Pfade verwenden.  GitHub Pages rendert
die gespeicherten Ausgaben, statt CI-Zeit für die schweren Abtastzellen zu
verwenden.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/de/tutorials/sde_monte_carlo_distribution

Magnetische Koordinaten und Gleichgewichte
------------------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/de/tutorials/magnetic_coordinates_comparison

RMPs, Inseln und Poincaré-Analyse
---------------------------------

Beginnen Sie mit dem Resonanzanalyse-Notebook, wenn Sie magnetische Topologie
untersuchen.  Es behandelt jetzt divergenzfreie RMP-Vorlagen, den wichtigen
``m=1``-Zweig, ``cyna``-Fixpunktvalidierung, mehrkomponentige kontravariante
``B^r``-Magnetfeldspektren als Atlanten, modulare ``q``/``m/n``-Resonanzkarten
mit optionalen Poincaré- und Inselüberlagerungen, gemischte RMP/nRMP-Spektren, die
gesamte nRMP-Antwort aus allen nichtresonanten Moden,
Feldliniengeschwindigkeitsmodulation und Prüfungen der Störungsordnung.

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/de/tutorials/RMP_resonance_analysis
   /notebooks/i18n/de/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/de/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` bleibt als Ausführungs-/Cache-Variante des
Resonanzanalyse-Ablaufs im Repository, aber die öffentliche Dokumentation
verlinkt auf die erklärende Version oben.

Monodromie und Mannigfaltigkeiten
---------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/de/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/de/tutorials/monodromy_xcycle_analytic

Klassische und allgemeine dynamische Systeme
--------------------------------------------

Das Repository enthält außerdem leichtgewichtige Notebooks unter
``notebooks/examples``: ``Lorenz_attractor.ipynb``,
``resonance_1_1_map.ipynb``, ``Mobiusian_saddle_cycle.ipynb``,
``Xcycle_construction.ipynb`` und ``FPT_DX_to_DP_sympy.ipynb``.  Sie bleiben
Quellbeispiele statt ausgeführter Dokumentationsseiten, weil mehrere davon
Entwurfsnotebooks ohne Abschnittstitel sind.

Statische Tutorial-Abbildungen
------------------------------

Mehrere längere Arbeitsabläufe sind im Repository als statische Abbildungen und
generierte Ausgaben unter ``notebooks/tutorials`` vertreten.  Sie behandeln
q-Profil-Diagnostik, PEST/Boozer/Hamada/Equal-Arc-Koordinaten,
Inselunterdrückungsscans, Phasensteuerung, Poincaré-Mannigfaltigkeiten und
Solov'ev-Beispiele mit einer einzelnen Null.
