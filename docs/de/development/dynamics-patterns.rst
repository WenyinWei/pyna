Dynamik-Workflows und Erweiterungshilfen
========================================

pyna trennt mathematische Geometrie von Konstruktionspolitik.

Die Kernhierarchie bleibt kompakt:

- kontinuierliche Geometrie: ``Trajectory``, ``Cycle``, ``Tube``,
  ``TubeChain``;
- diskrete Geometrie: ``Orbit``, ``PeriodicOrbit``, ``Island``,
  ``IslandChain``;
- toroidale Klassen bleiben die öffentlichen Standard-Spezialisierungen der
  Topologie unter ``pyna.topo.Tube``, ``pyna.topo.Cycle`` und
  ``pyna.topo.IslandChain``.

Die Hilfsschicht ergänzt eine benutzerseitige Workflow-Fassade sowie explizite
Erweiterungspunkte um diese Hierarchie.

Workflow-Fassade
----------------

``TopologyWorkflow`` ist der empfohlene erste Einstieg für Tutorials und
Analyseskripte.  Es komponiert die Low-Level-Hilfen zu dem Pfad, dem Benutzer
tatsächlich folgen:

1. einen Fluss/eine Abbildung aufbauen oder entgegennehmen;
2. eine ``Trajectory`` integrieren oder einen ``Orbit`` iterieren;
3. geschlossene Stichproben explizit zu ``Cycle`` oder ``PeriodicOrbit``
   hochstufen;
4. ``Cycle``/``Tube``/``TubeChain``-Objekte mit einem Schnitt schneiden.

Die Fassade ist absichtlich dünn.  Sie führt keine neue Mathematik ein; sie
hält Notebook-Code lesbar und macht zugleich jede Hochstufung explizit.

Ausgearbeitetes Tutorial
------------------------

Für einen kompakten Workflow-Überblick beginnen Sie mit :doc:`/en/mini-cases`.
Für ein vollständiges visuelles Tutorial, das dieselben Hochstufungsideen auf
eine reale toroidale Rechnung anwendet, verwenden Sie
:doc:`/notebooks/tutorials/RMP_resonance_analysis`.  Es zeigt abgetastete
Poincare-Kreuzungen, explizite X/O-Fixpunktgeometrie, Überlagerungen von
Koordinatengittern und lokale Mannigfaltigkeitszweige.

Für kurze Copy-Paste-Rezepte verwenden Sie :doc:`/en/mini-cases`.  Diese Seite
ist als Brücke zwischen dem Schnelleinstieg und der vollständigen API-Referenz
gedacht.

Protocols
---------

``pyna.topo.protocols`` definiert strukturelle Verträge wie ``FlowLike``,
``MapLike``, ``SectionLike`` und ``TubeLike``.  Verwenden Sie diese beim
Hinzufügen eines neuen Domänenpakets, das mit pyna interoperieren soll, ohne
direkt von jeder Basisklasse zu erben.

Adapter
-------

``pyna.topo.adapters`` wandelt Benutzerdaten in stabile Kernobjekte um:

- Arrays oder Solver-Ausgaben zu ``Trajectory`` und ``Orbit``;
- Punkte oder fixpunktartige Objekte zu ``SectionPoint``;
- verifizierte Stichproben bei Bedarf zu ``PeriodicOrbit`` oder ``Cycle``.

Adapter normalisieren die Darstellung; sie sollten mathematische Aussagen
nicht verstecken.  Eine offene abgetastete Trajektorie bleibt zum Beispiel eine
``Trajectory``, sofern ein Aufrufer nicht explizit einen ``Cycle`` verlangt und
die Schließprüfung akzeptiert oder übergibt.

Builder
-------

``GeometryBuilder``, ``IslandChainBuilder`` und ``TubeChainBuilder`` erfassen
Konstruktionspolitik.  Bevorzugen Sie Builder, wenn ein Workflow Topologie aus
mehreren Low-Level-Bestandteilen zusammensetzt, weil sie Validierung, Metadaten
und Rückverweise zentralisieren.

Bridges
-------

``CoreSectionCutBridge`` ist die standardmäßige Brücke von kontinuierlichen zu
diskreten Kernobjekten:

- ``Cycle.section_cut(section)`` gibt einen ``PeriodicOrbit`` zurück;
- ``Tube.section_cut(section)`` gibt eine ``IslandChain`` zurück;
- ``TubeChain.section_cut(section)`` führt die entstehenden Inseln zusammen.

Toroidale Objekte besitzen bereits optimierte ``section_cut``-Methoden.
Verwenden Sie diese direkt oder rufen Sie ``TopologyWorkflow.section_cut(...)``
auf und lassen Sie das Objekt seine eigene Implementierung dispatchen.

Factories
---------

``DynamicalSystemFactory`` baut einsatzbereite Systeme aus stabilen
Zeichenkettenschlüsseln wie ``callable-flow``, ``callable-map``,
``hamiltonian``, ``nbody`` und ``geometric-brownian-motion``.

``PoincareMapFactory`` wählt eine ausführbare Return-Map-Implementierung.  Der
Standard ``backend="auto"`` wählt derzeit die portable ``GeneralPoincareMap``,
sofern keine cyna-Feld-Cache-Argumente bereitgestellt werden.

``GeometryFactory`` baut Topologiegeometrie über die Builder-Schicht.  Sie ist
nützlich für konfigurationsgetriebene Beispiele und nachgelagerte Pakete, die
stabile Konstruktionsschlüssel benötigen.

Kompatibilitätsregeln
---------------------

- Ändern Sie ``pyna.topo.Tube``, ``Cycle`` oder ``IslandChain`` nicht so, dass
  sie auf die Kernklassen zeigen; verwenden Sie ``CoreTube``, ``CoreCycle`` und
  ``CoreIslandChain`` für generische Wurzeln.
- Verwenden Sie an toroidal-spezifischen Grenzen keine duck-typed
  Pseudo-Schnitte.  Verwenden Sie erstklassige ``Section``-Objekte.
- Behandeln Sie Registries als veränderlichen Zustand.  Verwenden Sie lokale
  ``Registry``-Instanzen in Tests und nachgelagerten Paketen, wenn Isolation
  wichtig ist.
