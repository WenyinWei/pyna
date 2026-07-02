迷你案例
========

这一页放在快速入门和完整 API 之间：知道自己手里是什么系统时，直接复制最小
模式；需要深度定制时，看该改哪一层。

入口怎么选
----------

.. list-table::
   :header-rows: 1

   * - 你手里有
     - 从这里开始
     - 通常得到的几何对象
   * - ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` 或 ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory``，必要时提升为 ``Cycle``
   * - Hamiltonian ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` 或 ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - 有限维映射 ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit``，必要时提升为 ``PeriodicOrbit``
   * - 环形磁场线问题
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``、``Tube``、``IslandChain``、流形
   * - 教学用随机模型
     - ``BrownianMotion`` 或 ``GeometricBrownianMotion``
     - 采样 ``Trajectory`` 和统计量

案例 1：ODE 采样到闭合 Cycle
----------------------------

``Trajectory`` 只是采样曲线；``Cycle`` 表示你明确声称它闭合。

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow

   wf = TopologyWorkflow(closure_tol=2e-2)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(wf.closing_error(traj))
   cycle = wf.closed_cycle(traj)
   print(cycle.period_value, cycle.ambient_dim)

生产代码里要把闭合误差容差写清楚，这样数值假设可以被审查。

案例 2：映射迭代到 PeriodicOrbit
-------------------------------

映射先产生 ``Orbit``。只有已知闭合或数值验证闭合时，才提升为
``PeriodicOrbit``。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap
   from pyna.topo import TopologyWorkflow

   flip = CallableMap(lambda x: np.array([-x[0], -x[1]]), dim=2)
   wf = TopologyWorkflow(closure_tol=1e-12)

   orbit = wf.orbit(flip, [1.0, 0.0], n_iter=2)
   periodic = wf.periodic_orbit(
       orbit.states[:-1],
       map_obj=flip,
       coordinate_names=("x", "y"),
   )
   print(periodic.period, periodic.points[0].state)

外部库里的映射可以先用 ``CallableMap`` 包一层；更深度集成时，实现
``__call__(x)`` 和 ``phase_space`` 属性。

案例 3：解析 Stellarator 的 O/X 点
----------------------------------

磁约束问题里，连续场线流在 Poincare 截面上变成离散返回映射。可执行教程
:doc:`/notebooks/tutorials/RMP_resonance_analysis` 现在承担完整图文案例：

1. 构造公开解析 stellarator 模型；
2. 验证 ``m=1`` 和 ``m>1`` RMP 模板的无散度性质；
3. 追踪未扰动和 RMP 扰动后的 Poincare 截面；
4. 用 ``cyna`` Newton 定点和解析 RMP 相位互相校验；
5. 对所有非共振谱行求和，得到总 nRMP 响应；
6. 只把贡献表作为排序和收敛诊断，而不是把单个分量当成模型；
7. 可视化 nRMP 引起的磁面形变和场线流速调制；
8. 叠加局部稳定分支和 PEST-style 坐标网格。

修改固定点绘图、截面几何、RMP/nRMP 诊断或教程渲染前，建议先在本地跑这个
notebook。它的代码路径尽量使用公开 helper，方便用户把案例直接迁移到自己的
分析脚本。

案例 4：注册自定义系统
----------------------

Factory 不是必须的。它适合配置驱动的下游项目。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableFlow
   from pyna.topo.factories import DynamicalSystemFactory

   def make_damped_oscillator(gamma=0.1):
       return CallableFlow(
           lambda x, t: np.array([x[1], -x[0] - gamma*x[1]]),
           dim=2,
           coordinate_names=("q", "p"),
           label="damped oscillator",
       )

   DynamicalSystemFactory.register(
       "damped-oscillator",
       lambda gamma=0.1: make_damped_oscillator(gamma),
       overwrite=True,
   )
   flow = DynamicalSystemFactory.create("damped-oscillator", gamma=0.05)

测试里如果担心全局注册影响顺序，使用局部 ``Registry`` 实例。

案例 5：SDE 分布估计
--------------------

单条 SDE 样本路径是 pyna 的 ``Trajectory``；大量路径的 Monte Carlo ensemble
是统计估计，暂时应保持为数组，不要自动提升为拓扑不变对象。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1)
   print(path.final)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=100_000)
   terminal = 100.0 * np.exp(gbm.expected_log_growth()[0] + gbm.sigma[0] * z)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

完整的本地预执行 notebook 见 :doc:`/zh/tutorials/sde-monte-carlo`。

高手应该改哪一层
----------------

.. list-table::
   :header-rows: 1

   * - 目标
     - 扩展点
     - 注意
   * - 新物理模型
     - ``CallableFlow``、``HamiltonianSystem`` 或 ``ContinuousFlow`` 子类
     - 积分方法返回 pyna 几何对象
   * - 新映射族
     - ``CallableMap`` 或 ``DiscreteMap`` 子类
     - 暴露稳定的坐标名
   * - 新截面
     - ``pyna.topo.section.Section`` 风格对象
     - 明确 crossing/project 语义
   * - 新数据格式
     - ``pyna.topo.adapters``
     - 只做规范化，不要偷偷声称周期性
   * - 新装配策略
     - ``pyna.topo.builders``
     - 集中验证和 metadata
   * - 新后端选择
     - factories 或 workflow facade
     - 原始后端数组应留在 pyna 对象边界内

经验法则：数学对象用 dataclass；输入格式用 adapter；验证和回链用 builder；
只有需要稳定字符串 key 时才引入 factory。

Notebook 上线前检查
-------------------

先跑关键教程：

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

重计算 notebook 在本地执行并提交输出：

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

按 GitHub Pages 的 notebook 集合本地构建：

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
