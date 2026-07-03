トポロジー（``pyna.topo``）
==========================

``pyna.topo`` パッケージは、Poincare 写像のトポロジー構造を解析するアルゴリズムを
提供します。磁島、X/O cycle、安定/不安定多様体、異宿 tangles が含まれます。

中心となる設計は 2 層の階層です。

``pyna.topo.core``
   有限次元で領域に依存しない幾何:
   ``Trajectory``、``Orbit``、``PeriodicOrbit``、``Cycle``、``Island``、
   ``IslandChain``、``Tube``、``TubeChain``。

``pyna.topo.toroidal``
   磁場閉じ込め向けの特殊化で、``R/Z/phi``、巻き数、monodromy payload、
   cyna ベースの断面切断、トロイダル診断を追加します。

.. contents:: サブモジュール
   :depth: 2
   :local:

----

Poincare 写像
-------------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

幾何オブジェクト階層
--------------------

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

断面
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

磁島
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

多様体計算
----------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

周期 orbit / cycle
------------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

古典写像とカオス診断
--------------------

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
