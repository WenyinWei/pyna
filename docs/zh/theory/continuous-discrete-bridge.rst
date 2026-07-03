连续与离散几何
==============

pyna 为连续时间和离散时间动力系统使用彼此独立的对象族。

连续时间侧：

- ``Trajectory`` 是采样的有限时间几何。
- ``Cycle`` 是 flow 的周期轨道。
- ``Tube`` 是围绕椭圆 cycle 的共振区，边界可能由双曲 cycle 给出。
- ``TubeChain`` 将属于同一共振的 tube 分组。

离散时间侧：

- ``Orbit`` 是采样的映射迭代几何。
- ``PeriodicOrbit`` 是映射的闭合轨道。
- ``Island`` 是截面上的一个约化共振岛。
- ``IslandChain`` 是截面层面的磁岛链。

两侧之间的 bridge 是截面切割。用 Poincare 截面切割 ``Cycle`` 会得到返回映射的
``PeriodicOrbit``。切割 ``Tube`` 会得到 ``IslandChain``。切割 ``TubeChain`` 会合并
其 tube 产生的磁岛链。

这种分离是有意为之。数值轨迹即使没有证明不变性，也可以是有用的几何对象。因此
builder 和 adapter 会让提升过程保持显式：用户可以要求闭合检查通过后，才把采样轨迹
变成 ``Cycle``，或把映射样本变成 ``PeriodicOrbit``。

同一套词汇同时供通用有限维系统和环形磁场线专门化使用。通用根类可通过
``pyna.topo.CoreTube`` 等名称访问；环形默认类仍可通过 ``pyna.topo.Tube``、
``pyna.topo.Cycle`` 和 ``pyna.topo.IslandChain`` 访问。

另见
----

- :doc:`/zh/mini-cases`
- :doc:`/notebooks/i18n/zh/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/i18n/zh/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/i18n/zh/tutorials/island_jacobian_analysis`
