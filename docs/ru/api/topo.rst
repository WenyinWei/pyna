Топология (``pyna.topo``)
=========================

Пакет ``pyna.topo`` предоставляет алгоритмы анализа топологической структуры
отображений Пуанкаре: магнитные острова, X/O-циклы, устойчивые/неустойчивые
многообразия и гетероклинические спутывания.

Центральная конструкция - двухслойная иерархия:

``pyna.topo.core``
   конечномерная, domain-agnostic геометрия:
   ``Trajectory``, ``Orbit``, ``PeriodicOrbit``, ``Cycle``, ``Island``,
   ``IslandChain``, ``Tube`` и ``TubeChain``.

``pyna.topo.toroidal``
   специализации магнитного удержания, добавляющие ``R/Z/phi``, winding numbers,
   payloads монодромии, cyna-backed section cuts и тороидальные diagnostics.

.. contents:: Подмодули
   :depth: 2
   :local:

----

Отображения Пуанкаре
--------------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Иерархия геометрических объектов
--------------------------------

.. automodule:: pyna.topo._base
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.core
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Сечения
-------

.. automodule:: pyna.topo.section
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Накопители Пуанкаре
-------------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Магнитные острова
-----------------

.. automodule:: pyna.topo.toroidal
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.toroidal_island
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Вычисление многообразий
-----------------------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Периодические орбиты / циклы
----------------------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Классические отображения и диагностика хаоса
--------------------------------------------

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.topo.chaos
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
