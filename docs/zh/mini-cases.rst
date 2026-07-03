迷你案例
========

本页是快速开始和完整 API 参考之间的短路径。已经知道自己处理的是哪类系统，并希望
获得最小可运行 pyna 模式时，请从这里开始。

从哪个入口开始？
----------------

.. list-table::
   :header-rows: 1

   * - 你已有
     - 从这里开始
     - 通常得到的几何对象
   * - ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` 或 ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory``，然后可能是 ``Cycle``
   * - Hamiltonian ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` 或 ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - 有限维映射 ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit``，然后可能是 ``PeriodicOrbit``
   * - 环形磁场
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``、``Tube``、``IslandChain``、流形
   * - 随机教学模型
     - ``BrownianMotion`` 或 ``GeometricBrownianMotion``
     - 采样 ``Trajectory`` 加统计量

案例 1：从 ODE 样本到闭合 Cycle
--------------------------------

``Trajectory`` 表示采样数据。``Cycle`` 表示你进一步声称该样本是闭合的。

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

在生产工作流中，应显式保留闭合容差。这样数值假设才便于审查。

案例 2：从映射迭代到周期轨道
------------------------------

映射首先产生 ``Orbit`` 对象。只有对已知闭合或已通过数值验证的样本，才提升为
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

如果你的映射来自其他包，可以用 ``CallableMap`` 包装它，或实现 ``__call__(x)``
并提供 ``phase_space`` 属性。

案例 3：解析 stellarator 的 O/X 点
----------------------------------

在磁约束工作中，场线流会被 Poincare 截面切开。可执行教程
:doc:`/notebooks/tutorials/RMP_resonance_analysis` 现在包含完整的可视化计算：

1. 构建公开的解析 stellarator 模型；
2. 验证无散度的 ``m=1`` 和 ``m>1`` RMP 模板；
3. 追踪未扰动和受扰动的 Poincare 截面；
4. 将解析共振 X/O 相位与 ``cyna`` Newton 固定点比较；
5. 用逆变 ``B^r`` pcolormesh 图册、带可选 Poincare 投影的 ``q``/``m/n``
   共振图、交互式 Plotly 3-D 柱状图、径向固定 ``n``/固定 ``m`` 图、共振曲线和可切换
   磁岛宽度标记，分析多分量 RMP 谱；
6. 计算所有非共振谱行产生的总 nRMP 响应；
7. 仅把贡献表作为排序和收敛诊断；
8. 可视化 nRMP 磁通面形变和场线速度调制；
9. 叠加局部稳定分支和 PEST 风格坐标网格。

当你测试固定点绘图、截面几何、RMP/nRMP 诊断或教程渲染的改动时，请使用这个
notebook。它足够小，可以在发布文档前本地运行，同时仍会覆盖下游分析脚本使用的
公共 helper API。

案例 4：自定义系统注册
-----------------------

factory 是可选的。只有当下游项目由配置驱动时，它们才重要。

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

如果全局注册会让测试依赖执行顺序，请在测试中使用局部 ``Registry`` 实例。

案例 5：SDE 分布估计
--------------------

单条 SDE 路径是 pyna 轨迹。Monte Carlo ensemble 是统计估计器；在 pyna 增加专用
ensemble 对象之前，请把它们保留为数组。

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

完整的已执行示例包含 Brownian、Ornstein-Uhlenbeck 和 geometric Brownian motion
分布，可参见 :doc:`/zh/tutorials/sde-monte-carlo`。

案例 6：在哪里定制
------------------

.. list-table::
   :header-rows: 1

   * - 目标
     - 扩展点
     - 注意事项
   * - 新物理模型
     - ``CallableFlow``、``HamiltonianSystem`` 或 ``ContinuousFlow`` 子类
     - 从积分方法返回 pyna 几何对象
   * - 新映射族
     - ``CallableMap`` 或 ``DiscreteMap`` 子类
     - 暴露稳定的坐标名
   * - 新截面
     - ``pyna.topo.section.Section`` 风格对象
     - 清晰实现 crossing/project 语义
   * - 新数据格式
     - ``pyna.topo.adapters``
     - 规范化数据；不要静默声称周期性
   * - 新装配策略
     - ``pyna.topo.builders``
     - 集中验证和 metadata
   * - 新后端选择
     - factory 或 workflow facade
     - 把原始后端数组保持在 pyna 对象之后

经验法则：数学对象使用 dataclass，输入规范化使用 adapter，验证使用 builder；
只有当用户需要稳定字符串键时才使用 factory。

Notebook 检查清单
-----------------

发布文档前：

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

对于带保存输出的重型 notebook，请在本地运行并提交更新后的 ``.ipynb`` 文件：

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

若要运行 GitHub Pages 使用的同一组 notebook，请本地构建 Sphinx：

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
