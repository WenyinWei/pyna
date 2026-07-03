一般力学系（``pyna.dynamics``）
===============================

``pyna.dynamics`` は広い力学系レイヤーです。意図的に小さく保たれており、
``pyna.topo`` と相互運用できます。

- callable ODE flow とサンプル trajectory
- 正準 Hamiltonian 系と可分 Hamiltonian
- 対ごとの重力/静電 N-body 系
- Jacobian、固定点残差、Lyapunov スペクトル推定を備えた有限次元写像
- Ito SDE、Brownian motion、geometric Brownian motion

クラスは state-first の規約を使います。flow は ``rhs(x, t)``、写像は ``step(x)`` です。

幾何との統合
------------

このモジュールは、トロイダル topology と同じ幾何クラスを返します。

- ``TimeSeriesSolution`` は ``pyna.topo.core.Trajectory`` です。
- ``CallableMap.orbit_geometry`` は ``pyna.topo.core.Orbit`` を返します。
- ``CallableMap.periodic_orbit`` は ``pyna.topo.core.PeriodicOrbit`` を返します。
- ``pyna.topo.CoreTube`` と ``pyna.topo.CoreIslandChain`` は汎用の有限次元ルートです。
  ``pyna.topo.Tube`` は後方互換性のためトロイダル特殊化として残ります。

これにより Hamiltonian 系、N-body flow、写像、SDE サンプルパスは、磁力線トポロジーと
同じ ``Cycle``/``Tube``/``IslandChain`` 語彙を共有できます。

教育用 notebook や拡張の多い workflow では、``TopologyWorkflow`` と低レベルの
adapter、builder、bridge、factory helper を扱う :doc:`dynamics-patterns` を参照して
ください。

連続 Flow
---------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Hamiltonian 系
--------------

``H(q, p, t)`` またはその勾配を与えられる場合は ``HamiltonianSystem`` を使います。
``H(q, p) = T(p) + V(q)`` と velocity-Verlet step には
``SeparableHamiltonianSystem`` を使います。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import SeparableHamiltonianSystem

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   x1 = oscillator.step_velocity_verlet(np.array([1.0, 0.0]), dt=0.01)

.. automodule:: pyna.dynamics
   :no-index:
   :members: HamiltonianSystem, SeparableHamiltonianSystem
   :show-inheritance:

N-body 系
---------

``NBodySystem`` はフラット化した状態ベクトルを
``[positions.ravel(), velocities.ravel()]`` として保存し、構造化配列を pack/unpack
する helper を提供します。Newton 重力と静電 Coulomb 相互作用をサポートします。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import NBodySystem

   system = NBodySystem([1.0, 1.0], spatial_dim=2, interaction="gravity")
   y0 = system.pack_state(
       positions=np.array([[-1.0, 0.0], [1.0, 0.0]]),
       velocities=np.zeros((2, 2)),
   )
   dy = system.vector_field(y0)

.. automodule:: pyna.dynamics
   :no-index:
   :members: NBodySystem
   :show-inheritance:

写像と局所多様体
----------------

``CallableMap`` は任意の有限次元写像を扱います。``fixed_point_eigenspaces`` は固定点の
安定、不安定、中心固有空間を分類し、局所多様体構成への有用な bridge になります。

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

確率微分方程式
--------------

SDE レイヤーは Ito 形式 ``dX = a(X,t) dt + B(X,t) dW`` を使い、再現可能な研究と教育例の
ために決定論的な Euler-Maruyama 実装を提供します。分布推定 workflow については
:doc:`/ja/tutorials/sde-monte-carlo` を参照してください。

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

関連する Topology レイヤー
--------------------------

topology パッケージは抽象的な数学階層と Poincare 機構を保持します。

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
