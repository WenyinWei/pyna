Theorie der Torusdeformation
============================

``pyna.toroidal.torus_deformation`` enthält die analytischen Werkzeuge zur
Torusdeformation, mit denen untersucht wird, wie invariante Tori und resonante
Strukturen auf kontrollierte Störungen reagieren.

Konzeptionelle Rolle
--------------------

In der Geometriehierarchie gilt:

- ein invarianter Torus ist ein ``InvariantTorus``;
- ein resonanter elliptischer Zyklus ist der Kern eines ``Tube``;
- ein hyperbolischer Zyklus begrenzt einen Tube und erzeugt stabile/instabile
  Mannigfaltigkeiten;
- das Schneiden von Tubes mit einem Poincare-Schnitt erzeugt
  ``IslandChain``-Objekte.

Berechnungen zur Torusdeformation speisen daher direkt die Topologiesteuerung:
Sie sagen voraus, welche spektralen Störungen resonante Strukturen bewegen,
aufspalten, heilen oder unterdrücken.

Öffentliche API
---------------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

Verwandte Module
----------------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
