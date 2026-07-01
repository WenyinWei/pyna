架构概览
========

pyna 围绕两条主线组织：

1. 动力系统定义有限维相空间中的演化规则；
2. 拓扑模块描述这些相空间里的几何对象。

分层结构
--------

``pyna.topo.dynamics`` 提供抽象数学层：``PhaseSpace``、
``ContinuousFlow``、``HamiltonianFlow``、``DiscreteMap`` 和
``PoincareMap``。

``pyna.dynamics`` 提供即用系统：``CallableFlow``、Hamiltonian、N-body、
``CallableMap`` 和 Ito SDE。

``pyna.topo.core`` 是领域无关几何层：

- 连续时间：``Trajectory``、``Cycle``、``Tube``、``TubeChain``；
- 离散时间：``Orbit``、``PeriodicOrbit``、``Island``、``IslandChain``。

``pyna.topo.toroidal`` 在通用层上加入环形场线特有的 ``R/Z/phi``、
winding、monodromy、cyna 加速和截面重构。

设计规则
--------

- 新的有限维问题优先接入 ``pyna.dynamics`` 和 ``pyna.topo.core``；
- 只有真实依赖环形柱坐标几何时才使用 toroidal 专用类；
- 采样数据先是 ``Trajectory`` 或 ``Orbit``，闭合结构要显式提升；
- cyna 保持在高性能边界，用户侧 API 返回 pyna 对象而不是裸 C++ 数组。

英文完整版本：

- :doc:`../../en/development/architecture`
