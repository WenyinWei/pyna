Torus Deformation 理论
======================

``pyna.toroidal.torus_deformation`` 用于研究不变环面和共振结构在受控扰动下
的响应。它的输出应回到拓扑对象体系中理解：

- 不变环面对应 ``InvariantTorus``；
- 共振椭圆周期轨道是 ``Tube`` 的核心；
- 双曲周期轨道给出 tube 边界和稳定/不稳定流形；
- 切 Poincare 截面后得到 ``IslandChain``。

因此 torus deformation 计算可以直接服务于拓扑控制：预测哪些谱扰动会移动、
分裂、修复或抑制共振结构。

英文完整版本：

- :doc:`../../en/theory/torus_deformation`
