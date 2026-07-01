拓扑对象与 Poincare 分析 (``pyna.topo``)
========================================

``pyna.topo`` 同时包含通用有限维拓扑对象和环形场线专用工具。

核心对象：

- 连续时间：``Trajectory``、``Cycle``、``Tube``、``TubeChain``；
- 离散时间：``Orbit``、``PeriodicOrbit``、``Island``、``IslandChain``；
- 截面与稳定性：``SectionPoint``、``LinearStabilityData``、``ToroidalSection``。

重要语义：采样曲线不是自动闭合轨道，点云不是自动岛链。只有通过显式检查或
模型先验后，才把采样数据提升为拓扑对象。

英文详细 API：

- :doc:`../../en/api/topo`
- :doc:`../../en/theory/continuous-discrete-bridge`
