公共 API 导航
=============

本页用于快速找到 pyna 的稳定公共接口。自动生成的 AutoAPI 仍然是完整的调试参考；
下面列出的入口更适合 notebook、科研脚本和下游包直接使用。

几何对象词汇
------------

当你关心的是相空间中的对象，而不是产生它的具体求解器时，优先从这里开始。

.. list-table::
   :header-rows: 1

   * - 任务
     - 公共入口
   * - 连续时间采样运动
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - 离散映射动力学
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - 环形截面几何
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - 显式提升与适配
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

一般动力系统
------------

非环形模型也应尽量返回同一套几何对象，此时使用 :mod:`pyna.dynamics`。

.. list-table::
   :header-rows: 1

   * - 模型族
     - 公共入口
   * - ODE 流
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Hamiltonian 系统
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N 体系统
     - :class:`pyna.dynamics.NBodySystem`
   * - 离散映射
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - 随机微分方程
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

环形与 RMP 工作流
----------------

磁坐标、磁力线追踪、磁谱分析和可视化叠加优先使用这些模块。

.. list-table::
   :header-rows: 1

   * - 需求
     - 公共入口
   * - 平衡与坐标
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - 磁力线追踪和带缓存的工作流
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - 逆变径向扰动磁谱
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - RMP/nRMP 教程诊断
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - 磁谱科研制图
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Poincare、X/O 点和磁岛叠加
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

何时查看 AutoAPI
----------------

需要构造函数签名、继承成员、低频诊断或实现细节时，再进入
:doc:`/en/api/generated/pyna/index`。新的教程和面向用户的示例应尽量停留在
上面的公共入口层。
