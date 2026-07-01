.. _zh-quickstart:

快速入门
========

安装验证
--------

.. code-block:: python

   import pyna
   from pyna._cyna import is_available, get_version

   print(pyna.__version__)
   print(is_available(), get_version())

``is_available()`` 应为 ``True``。如果失败，请先检查 :doc:`installation`
中的 cyna/xmake 构建说明。

通用动力系统
------------

Hamiltonian 系统、N-body、有限维映射和 SDE 都使用同一套相空间几何对象。
例如一维谐振子：

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

   traj = oscillator.trajectory([1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(traj.final)

``traj`` 是 ``pyna.topo.core.Trajectory`` 的子类，因此可以和
``Cycle``、``Tube``、``IslandChain`` 等几何对象协同使用。

有限维映射：

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap

   m = CallableMap(lambda x: np.array([2*x[0], 0.5*x[1]]), dim=2)
   orbit = m.orbit_geometry([1.0, 1.0], n_iter=5)
   fixed = m.periodic_orbit([[0.0, 0.0]], section_label="origin")

随机微分方程：

.. code-block:: python

   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1234)

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
   print(stock.expected_log_growth())

工作流辅助层
------------

大型研究代码和教学 notebook 建议先使用 ``TopologyWorkflow``，把“构造系统、
积分/迭代、显式提升、截面切割”写成清晰的线性流程：

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow
   from pyna.topo.section import HyperplaneSection

   wf = TopologyWorkflow(closure_tol=1e-3)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
   pmap = wf.poincare_map(flow, section, dt=0.02)

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   cycle = wf.closed_cycle(traj)

这里的规则是显式的：``Trajectory`` 只是采样曲线，只有满足闭合检查后才提升为
``Cycle``；``Tube/TubeChain`` 的截面切割会得到 ``IslandChain``。底层
Adapter、Builder、Bridge、Factory 仍然保留给库作者和复杂扩展使用。

环形三维向量场与 Poincare 拓扑
------------------------------

环形磁约束工作流仍然是 pyna 的强项。典型路径是：

1. 构造或读取柱坐标磁场 ``BR, BZ, BPhi``；
2. 使用 ``pyna.flt`` 或 ``pyna.toroidal.flt`` 追踪场线；
3. 在 ``ToroidalSection`` 上累积 Poincare 点；
4. 用 ``pyna.topo`` 查找固定点、周期轨道、岛链和流形；
5. 对生产级批量追踪使用 ``cyna`` 加速。

最小截面对象：

.. code-block:: python

   from pyna.topo.section import ToroidalSection

   section = ToroidalSection(0.0)

通用与环形类层次的关系：

- ``pyna.topo.CoreCycle``、``CoreTube``、``CoreIslandChain`` 是任意有限维
  相空间的通用几何对象；
- ``pyna.topo.Cycle``、``Tube``、``IslandChain`` 默认仍是环形/MCF 专用类，
  带有 ``R/Z/phi``、winding、monodromy 和 cyna-backed section cut；
- 新的非 MCF 代码应优先用 ``pyna.dynamics`` 和 ``pyna.topo.core``，只有
  真实处在环形柱坐标场线问题中才使用 toroidal 专用类。

下一步
------

- 英文快速开始: :doc:`../en/quickstart`
- 中英文迷你案例: :doc:`mini-cases` / :doc:`../en/mini-cases`
- SDE Monte Carlo 教程: :doc:`tutorials/sde-monte-carlo`
- 通用动力系统 API: :doc:`../en/api/dynamics`
- 工作流与扩展 API: :doc:`../en/api/dynamics-patterns`
- 拓扑对象 API: :doc:`../en/api/topo`
- cyna 加速层: :doc:`../en/api/cyna`
