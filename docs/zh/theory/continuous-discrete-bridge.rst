连续与离散几何
==============

pyna 有意区分连续时间和离散时间对象。

连续时间：

- ``Trajectory``：有限时间采样曲线；
- ``Cycle``：流的周期轨道；
- ``Tube``：椭圆周期轨道附近的共振管状区域；
- ``TubeChain``：属于同一共振结构的一组 tube。

离散时间：

- ``Orbit``：映射迭代样本；
- ``PeriodicOrbit``：映射的周期轨道；
- ``Island``：截面上的一个共振岛；
- ``IslandChain``：截面上的周期岛链。

桥梁是 section cut：切 ``Cycle`` 得到返回映射的 ``PeriodicOrbit``，切
``Tube`` 得到 ``IslandChain``。这与环形磁场线工作流一致：连续的磁岛 tube
在 Poincare 截面上表现为离散岛链。

英文完整版本：

- :doc:`../../en/theory/continuous-discrete-bridge`
