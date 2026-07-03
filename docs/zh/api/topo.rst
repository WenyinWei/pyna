拓扑（``pyna.topo``）
====================

``pyna.topo`` 包提供用于分析 Poincare 映射拓扑结构的算法：磁岛、X/O cycle、
稳定/不稳定流形，以及异宿缠结。

中心设计是两层层次：

``pyna.topo.core``
   有限维、与领域无关的几何对象：
   ``Trajectory``、``Orbit``、``PeriodicOrbit``、``Cycle``、``Island``、
   ``IslandChain``、``Tube`` 和 ``TubeChain``。

``pyna.topo.toroidal``
   面向磁约束的专门化，加入 ``R/Z/phi``、绕数、monodromy payload、cyna 支持的
   截面切割和环形诊断。

.. contents:: 子模块
   :depth: 2
   :local:

----

Poincare 映射
-------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

几何对象层次
------------

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

截面
----

.. automodule:: pyna.topo.section
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Poincare Accumulator
--------------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

磁岛
----

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

流形计算
--------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

周期轨道 / Cycle
----------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

经典映射和混沌诊断
------------------

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
