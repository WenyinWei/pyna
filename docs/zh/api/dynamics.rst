一般动力系统（``pyna.dynamics``）
=================================

``pyna.dynamics`` 是宽泛的动力系统层。它刻意保持小而清晰，并能与
``pyna.topo`` 互操作：

- 以 callable 表示的 ODE flow 和采样轨迹
- 正则 Hamiltonian 系统和可分 Hamiltonian
- 成对引力/静电 N-body 系统
- 带 Jacobian、固定点残差和 Lyapunov 谱估计的有限维映射
- Ito SDE、Brownian motion 和 geometric Brownian motion

这些类使用 state-first 约定：flow 使用 ``rhs(x, t)``，映射使用 ``step(x)``。

几何集成
--------

该模块返回与环形拓扑相同的几何类：

- ``TimeSeriesSolution`` 是 ``pyna.topo.core.Trajectory``。
- ``CallableMap.orbit_geometry`` 返回 ``pyna.topo.core.Orbit``。
- ``CallableMap.periodic_orbit`` 返回 ``pyna.topo.core.PeriodicOrbit``。
- ``pyna.topo.CoreTube`` 和 ``pyna.topo.CoreIslandChain`` 是通用有限维根类；
  ``pyna.topo.Tube`` 仍然是保持向后兼容的环形专门化。

这样 Hamiltonian 系统、N-body flow、映射和 SDE 样本路径可以与磁场线拓扑共享同一套
``Cycle``/``Tube``/``IslandChain`` 词汇。

对于教学 notebook 或扩展较多的工作流，请参见 :doc:`dynamics-patterns`，其中介绍
``TopologyWorkflow`` 以及低层 adapter、builder、bridge 和 factory helper。

连续 Flow
---------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Hamiltonian 系统
----------------

当你可以提供 ``H(q, p, t)`` 或其梯度时，使用 ``HamiltonianSystem``。对于
``H(q, p) = T(p) + V(q)``，以及 velocity-Verlet 步进，使用
``SeparableHamiltonianSystem``。

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

N-body 系统
-----------

``NBodySystem`` 把扁平状态向量存为 ``[positions.ravel(), velocities.ravel()]``，
并提供打包和解包结构化数组的 helper。它支持 Newton 引力和静电 Coulomb 相互作用。

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

映射和局部流形
--------------

``CallableMap`` 处理任意有限维映射。``fixed_point_eigenspaces`` 会对固定点的稳定、
不稳定和中心特征空间分类，是连接局部流形构造的有用 bridge。

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

随机微分方程
------------

SDE 层使用 Ito 形式 ``dX = a(X,t) dt + B(X,t) dW``，并提供确定性的
Euler-Maruyama 实现，便于可重复研究和教学示例。分布估计工作流见
:doc:`/zh/tutorials/sde-monte-carlo`。

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

相关拓扑层
----------

topology 包保留抽象数学层次和 Poincare 机制：

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
