动力系统工作流和扩展辅助层
============================

pyna 将数学几何与构造策略分离。

核心层次保持紧凑：

- 连续时间几何：``Trajectory``、``Cycle``、``Tube``、``TubeChain``；
- 离散时间几何：``Orbit``、``PeriodicOrbit``、``Island``、``IslandChain``；
- 环形类仍然是 ``pyna.topo.Tube``、``pyna.topo.Cycle`` 和
  ``pyna.topo.IslandChain`` 下默认的公共拓扑专门化。

辅助层在这个层次周围增加一个面向用户的工作流门面，以及显式扩展点。

工作流门面
----------

``TopologyWorkflow`` 是教程和分析脚本推荐的第一入口。它把低层 helper 组合成用户实际
遵循的路径：

1. 构建或接收 flow/map；
2. 积分得到 ``Trajectory`` 或迭代得到 ``Orbit``；
3. 显式把闭合样本提升为 ``Cycle`` 或 ``PeriodicOrbit``；
4. 用截面切割 ``Cycle``/``Tube``/``TubeChain`` 对象。

这个门面刻意保持很薄。它不引入新的数学；它让 notebook 代码更可读，同时仍保持
每一次提升都显式可见。

完整教程
--------

紧凑的工作流概览可从 :doc:`/zh/mini-cases` 开始。若需要把同样的提升思想用于真实
环形计算的完整可视化教程，请使用
:doc:`/notebooks/i18n/zh/tutorials/RMP_resonance_analysis`。它展示了采样 Poincare 截面点、
显式 X/O 固定点几何、坐标网格叠加和局部流形分支。

短小的可复制配方见 :doc:`/zh/mini-cases`。该页面是快速开始和完整 API 参考之间的
预期桥梁。

扩展协议
--------

``pyna.topo.protocols`` 定义结构化契约，例如 ``FlowLike``、``MapLike``、
``SectionLike`` 和 ``TubeLike``。当你增加一个希望与 pyna 互操作的新领域包，而又不想
直接继承每个基类时，请使用这些契约。

数据适配
--------

``pyna.topo.adapters`` 将用户数据转换为稳定的核心对象：

- 数组或 solver 输出到 ``Trajectory`` 和 ``Orbit``；
- 点或固定点风格对象到 ``SectionPoint``；
- 在需要时把已验证样本转换为 ``PeriodicOrbit`` 或 ``Cycle``。

适配器规范化表示；它们不应隐藏数学声明。例如，开放采样轨迹仍是
``Trajectory``，除非调用方显式请求 ``Cycle`` 并接受或传入闭合检查。

构建器
------

``GeometryBuilder``、``IslandChainBuilder`` 和 ``TubeChainBuilder`` 捕获构造策略。
当工作流从多个低层材料装配拓扑对象时，优先使用构建器，因为它们集中处理验证、
metadata 和 back-link。

连续-离散桥接
--------------

``CoreSectionCutBridge`` 是 core 对象默认的连续到离散桥接实现：

- ``Cycle.section_cut(section)`` 返回 ``PeriodicOrbit``；
- ``Tube.section_cut(section)`` 返回 ``IslandChain``；
- ``TubeChain.section_cut(section)`` 合并得到的 island。

环形对象已经拥有优化过的 ``section_cut`` 方法。可以直接使用它们，或调用
``TopologyWorkflow.section_cut(...)``，让对象分派到自己的实现。

工厂
----

``DynamicalSystemFactory`` 通过稳定字符串键构建可直接使用的系统，例如
``callable-flow``、``callable-map``、``hamiltonian``、``nbody`` 和
``geometric-brownian-motion``。

``PoincareMapFactory`` 选择可执行的返回映射实现。默认 ``backend="auto"`` 当前会选择
可移植的 ``GeneralPoincareMap``，除非提供了 cyna field-cache 参数。

``GeometryFactory`` 通过构建器层构建拓扑几何。它适合配置驱动示例，以及需要稳定
构造键的下游包。

兼容性规则
----------

- 不要把 ``pyna.topo.Tube``、``Cycle`` 或 ``IslandChain`` 改成指向 core 类；
  通用根类使用 ``CoreTube``、``CoreCycle`` 和 ``CoreIslandChain``。
- 不要在仅限环形对象的边界使用 duck-typed 假截面。使用一等 ``Section`` 对象。
- 把注册表视为可变状态。在测试和下游包中需要隔离时，使用局部 ``Registry``
  实例。
