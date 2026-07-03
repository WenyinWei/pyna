Dynamik-Workflows und Erweiterungshilfen
========================================

``pyna.topo`` stellt Workflow-Hilfen um die zentralen Topologieobjekte bereit.
Der wichtigste benutzerseitige Einstiegspunkt ist ``TopologyWorkflow``; die
Low-Level-Module für Protocols, Adapter, Builder, Bridges und Factories bleiben
für nachgelagerte Bibliotheken verfügbar, die stabile Erweiterungspunkte
benötigen.

Workflow-Fassade
----------------

``TopologyWorkflow`` ist für Notebooks und tägliche Skripte gedacht.  Es
kombiniert Systemkonstruktion, Integration/Iteration, explizite Hochstufung
und Schnitte, ohne einen neuen mathematischen Objekttyp einzuführen.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocols
---------

Strukturelle Protocols beschreiben die Erweiterungsverträge für externe
Systeme.  Objekte von Drittanbietern können teilnehmen, indem sie die
erforderlichen Attribute und Methoden implementieren; das Ableiten von
pyna-Klassen ist optional.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adapter
-------

Adapter normalisieren Arrays, Solver-Ausgaben und vorhandene pyna-Objekte zu
zentralen Geometriedarstellungen.  Sie stufen abgetastete Daten nicht
stillschweigend zu invarianten Objekten hoch.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builder
-------

Builder kodieren explizite Hochstufungsregeln.  Zum Beispiel kann eine
Trajektorie nur über einen Builder- oder Adapter-Aufruf zu einem ``Cycle``
hochgestuft werden, der geschlossene Stichproben verlangen kann.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridges
-------

Bridges verbinden kontinuierliche und diskrete Objektfamilien:
``Cycle -> PeriodicOrbit`` und ``Tube/TubeChain -> IslandChain``.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factories und Registries
------------------------

Factories stellen stabile Konstruktionseinstiege für Systeme, Geometrie und
Poincare-Karten bereit.  Registries sind explizit und doppelt-sicher, sodass
Tests und nachgelagerte Bibliotheken ihre eigenen Erweiterungen isolieren
können.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
