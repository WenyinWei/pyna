环面形变理论
============

``pyna.toroidal.torus_deformation`` 包含解析环面形变工具，用于研究不变环面和共振结构
如何响应受控扰动。

概念角色
--------

在几何层次中：

- 不变环面是 ``InvariantTorus``；
- 共振椭圆 cycle 是 ``Tube`` 的核心；
- 双曲 cycle 约束 tube，并生成稳定/不稳定流形；
- 用 Poincare 截面切割 tube 会产生 ``IslandChain`` 对象。

因此，环面形变计算会直接进入拓扑控制：它们预测哪些谱扰动会移动、分裂、修复或抑制
共振结构。

公共 API
--------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

相关模块
--------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
