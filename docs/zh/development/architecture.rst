架构
====

pyna 围绕两个想法组织：

1. 动力系统在有限维相空间上定义演化规则；
2. 拓扑模块描述生活在这些相空间中的几何对象。

这种分离让同一套对象层次可以表示环形磁场线结构、Hamiltonian 共振区、经典映射、
N-body 轨道和随机样本路径。

第 0 层：Dynamics
-----------------

``pyna.topo.dynamics`` 提供抽象数学层：

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` 和 ``GeneralPoincareMap``

``pyna.dynamics`` 增加可直接使用的有限维系统：

- ``CallableFlow`` 和 ``CallableMap``
- ``HamiltonianSystem`` 和 ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``、``BrownianMotion`` 和 ``GeometricBrownianMotion``

这些类使用 topology core 表示采样输出。确定性 flow 轨迹是
``pyna.topo.core.Trajectory``；离散迭代点云是 ``pyna.topo.core.Orbit``。

第 1 层：Geometry
-----------------

``pyna.topo.core`` 是与领域无关的几何层次：

.. list-table::
   :header-rows: 1

   * - 类
     - 含义
     - 时间类型
   * - ``Trajectory``
     - 相空间中的有限采样曲线
     - continuous
   * - ``Cycle``
     - 连续 flow 的周期轨道
     - continuous
   * - ``Tube``
     - 围绕椭圆 cycle 的共振区
     - continuous
   * - ``TubeChain``
     - 共享同一共振的一族 tube
     - continuous
   * - ``Orbit``
     - 映射的有限采样迭代
     - discrete
   * - ``PeriodicOrbit``
     - 映射的有限周期轨道
     - discrete
   * - ``Island``
     - 截面上的一个共振岛
     - discrete
   * - ``IslandChain``
     - 截面上的周期磁岛链
     - discrete

关键 bridge 是 ``section_cut``：

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

这对应环形工作流：连续磁岛 tube 会在 Poincare 截面上被观察为离散磁岛链。

第 2 层：环形专门化
-------------------

``pyna.topo.toroidal`` 继承通用 core：

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

环形层增加：

- ``R``、``Z`` 和 ``phi`` 坐标
- 绕数 ``(m, n)``
- ``DPm`` 和 monodromy 分类
- cyna 加速的截面切割和追踪
- 截面视图对应关系和重构 helper

第 3 层：Workflow 和扩展 Helper
-------------------------------

``pyna.topo.protocols``、``adapters``、``builders``、``bridges`` 和
``factories`` 提供软件工程扩展层。主要面向 notebook 的入口是
``TopologyWorkflow``。这些 helper 把构造策略和后端选择放在数学 dataclass 之外：
外部系统可以通过 protocol 适配，用 adapter 规范化数据，通过 builder 提升对象，
通过 bridge 切割连续几何，并通过 factory 选择运行时实现。

第 4 层：Acceleration
--------------------

``cyna`` 实现高层 pyna API 背后的瓶颈。它不应拥有高层科学对象语义；它提供快速核，
用于追踪、插值、固定点扫描、壁面命中和扰动响应。

设计规则
--------

- 新的有限维几何优先使用通用 ``pyna.topo.core`` 类。
- 只有在 ``pyna.topo.toroidal`` 子类中添加环形专用字段。
- 采样的有限轨迹是几何对象，不会自动成为不变集。
- 只有当周期结构属于模型或已通过数值验证时，才把对象提升为
  ``Cycle``/``PeriodicOrbit``。
- 把 cyna 保持在 bridge 边界；应用层 API 应返回 pyna 对象，而不是原始 C++ 数组。
