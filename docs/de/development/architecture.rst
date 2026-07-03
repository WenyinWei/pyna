Architektur
===========

pyna ist um zwei Ideen herum organisiert:

1. dynamische Systeme definieren Evolutionsregeln auf endlichdimensionalen
   Phasenräumen;
2. Topologiemodule beschreiben geometrische Objekte, die in diesen
   Phasenräumen leben.

Diese Trennung erlaubt derselben Objekthierarchie, toroidale
Magnetfeldlinienstrukturen, Hamiltonsche Resonanzzonen, klassische
Abbildungen, N-Körper-Orbits und stochastische Abtastpfade darzustellen.

Schicht 0: Dynamik
------------------

``pyna.topo.dynamics`` stellt die abstrakte mathematische Schicht bereit:

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` und ``GeneralPoincareMap``

``pyna.dynamics`` ergänzt einsatzbereite endlichdimensionale Systeme:

- ``CallableFlow`` und ``CallableMap``
- ``HamiltonianSystem`` und ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``, ``BrownianMotion`` und ``GeometricBrownianMotion``

Diese Klassen verwenden den Topologiekern für abgetastete Ausgaben.  Eine
deterministische Flusstrajektorie ist eine ``pyna.topo.core.Trajectory``; eine
diskrete Wolke von Iterierten ist ein ``pyna.topo.core.Orbit``.

Schicht 1: Geometrie
--------------------

``pyna.topo.core`` ist die domänenunabhängige Geometriehierarchie:

.. list-table::
   :header-rows: 1

   * - Klasse
     - Bedeutung
     - Zeittyp
   * - ``Trajectory``
     - endliche abgetastete Kurve im Phasenraum
     - kontinuierlich
   * - ``Cycle``
     - periodischer Orbit eines kontinuierlichen Flusses
     - kontinuierlich
   * - ``Tube``
     - Resonanzzone um einen elliptischen Zyklus
     - kontinuierlich
   * - ``TubeChain``
     - Familie von Tubes, die eine Resonanz teilen
     - kontinuierlich
   * - ``Orbit``
     - endliche abgetastete Iterierte einer Abbildung
     - diskret
   * - ``PeriodicOrbit``
     - endlicher periodischer Orbit einer Abbildung
     - diskret
   * - ``Island``
     - eine reduzierte Resonanzinsel auf einem Schnitt
     - diskret
   * - ``IslandChain``
     - periodische Inselkette auf einem Schnitt
     - diskret

Die zentrale Brücke ist ``section_cut``:

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

Dies spiegelt den toroidalen Workflow wider, in dem kontinuierliche magnetische
Insel-Tubes auf einem Poincare-Schnitt als diskrete Inselketten beobachtet
werden.

Schicht 2: Toroidale Spezialisierung
------------------------------------

``pyna.topo.toroidal`` unterklassifiziert den generischen Kern:

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

Die toroidale Schicht ergänzt:

- Koordinaten ``R``, ``Z`` und ``phi``
- Windungszahlen ``(m, n)``
- ``DPm`` und Monodromy-Klassifikation
- cyna-beschleunigte Schnitte und Tracing
- Korrespondenz zur Schnittansicht und Rekonstruktionshilfen

Schicht 3: Workflow- und Erweiterungshilfen
-------------------------------------------

``pyna.topo.protocols``, ``adapters``, ``builders``, ``bridges`` und
``factories`` bilden die softwaretechnische Erweiterungsschicht.  Der wichtigste
notebookseitige Einstiegspunkt ist ``TopologyWorkflow``.  Diese Hilfen halten
Konstruktionspolitik und Backend-Auswahl außerhalb der mathematischen
Dataclasses: Externe Systeme können per Protocol konform sein, Daten mit
Adaptern normalisieren, Objekte über Builder hochstufen, kontinuierliche
Geometrie über Bridges schneiden und Laufzeitimplementierungen über Factories
auswählen.

Schicht 4: Beschleunigung
-------------------------

``cyna`` implementiert die Engpässe hinter den High-Level-pyna-APIs.  Es sollte
keine High-Level-Semantik wissenschaftlicher Objekte besitzen; es liefert
schnelle Kernel für Tracing, Interpolation, Fixpunktscans, Wandtreffer und
Störungsantwort.

Designregeln
------------

- Bevorzugen Sie generische Klassen aus ``pyna.topo.core`` für neue
  endlichdimensionale Geometrie.
- Fügen Sie toroidal-spezifische Felder nur in Unterklassen von
  ``pyna.topo.toroidal`` hinzu.
- Eine abgetastete endliche Trajektorie ist Geometrie, nicht automatisch eine
  invariante Menge.
- Stufen Sie Objekte nur dann zu ``Cycle``/``PeriodicOrbit`` hoch, wenn eine
  periodische Struktur Teil des Modells ist oder numerisch validiert wurde.
- Halten Sie cyna an Bridge-Grenzen; anwendungsnahe APIs sollten pyna-Objekte
  zurückgeben, keine rohen C++-Arrays.
