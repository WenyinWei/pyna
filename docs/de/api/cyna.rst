cyna-Beschleunigungsschicht
===========================

``cyna`` ist die mit pyna ausgelieferte C++-Beschleunigungsschicht.  Sie wird
dort eingesetzt, wo Python-Hot-Loops nicht akzeptabel sind:
Feldlinienverfolgung, Poincare-Batches, Fixpunktscans,
Connection-Length- und Wandtreffer, Spulenfelder sowie Kernel der funktionalen
Störungstheorie.

Build-Vertrag
-------------

``pyna._cyna`` erwartet ein kompiliertes ``_cyna_ext``-Binary im Paket.
Quellinstallationen bauen es über xmake; PyPI-Wheels enthalten es.  Siehe
:doc:`../installation` für Plattform-Setup und CUDA-Flags.

Die kanonische Reihenfolge für zylindrische Feld-Caches lautet:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Verwenden Sie :func:`pyna._cyna.prepare_field_cache`, um ein
``pyna.fields.VectorFieldCylind`` oder ein Legacy-Dict in C-kontigue Arrays
umzuwandeln.

High-Level- und Low-Level-APIs
------------------------------

Bevorzugen Sie High-Level-Wrapper für Anwendungscode:

- ``pyna.flt`` und ``pyna.toroidal.flt`` für Tracing
- ``pyna.topo`` für Poincare-Karten, Zyklen, Inseln, Mannigfaltigkeiten und
  FPT-Antwort
- ``pyna.toroidal.coils`` für den Aufbau von Spulenfeldern

Verwenden Sie ``pyna._cyna`` direkt nur an Bridge-Grenzen, für Diagnostik oder
beim Schreiben eines neuen High-Level-Wrappers.

Python-Wrapper-Referenz
-----------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

Hilfsfunktionen
---------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
