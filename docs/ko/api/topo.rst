위상 구조 (``pyna.topo``)
=========================

``pyna.topo`` package는 푸앵카레 맵의 위상 구조를 분석하는 알고리즘을
제공합니다. 여기에는 자기섬, X/O cycle, stable/unstable manifold,
heteroclinic tangle이 포함됩니다.

중심 설계는 두 계층 hierarchy입니다.

``pyna.topo.core``
   유한 차원, domain-agnostic geometry:
   ``Trajectory``, ``Orbit``, ``PeriodicOrbit``, ``Cycle``, ``Island``,
   ``IslandChain``, ``Tube`` 및 ``TubeChain``.

``pyna.topo.toroidal``
   ``R/Z/phi``, winding number, monodromy payload, cyna-backed section cut,
   toroidal diagnostics를 추가하는 자기 구속 specialization입니다.

.. contents:: Submodules
   :depth: 2
   :local:

----

푸앵카레 맵
-----------

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

기하 객체 계층
--------------

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

단면
----

.. automodule:: pyna.topo.section
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

푸앵카레 누적기
---------------

.. automodule:: pyna.topo.poincare
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

자기섬
------

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

다양체 계산
-----------

.. automodule:: pyna.topo.manifold
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

주기 궤도 / Cycle
-----------------

.. automodule:: pyna.topo.toroidal_cycle
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

고전 맵과 Chaos 진단
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
