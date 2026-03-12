API 参考
========

模块组织
--------

pyna 按功能域分为以下子模块：

**通用动力系统（``pyna`` 顶层）**

- :mod:`pyna.flt` — 场线追踪（CPU/CUDA）
- :mod:`pyna.topo` — 拓扑分析（岛链、庞加莱映射、周期轨道）
- :mod:`pyna.control` — FPT 拓扑控制框架（通用）

**磁约束聚变（``pyna.MCF``）**

- :mod:`pyna.MCF.equilibrium` — MHD 平衡（Solov'ev、GradShafranov）
- :mod:`pyna.MCF.coords` — 磁坐标系（PEST、Boozer、Hamada、Equal-arc）
- :mod:`pyna.MCF.coils` — 线圈几何和 Biot-Savart 场
- :mod:`pyna.MCF.control` — MCF 特定拓扑控制（间隙响应、q 剖面）
- :mod:`pyna.MCF.plasma_response` — 扰动 GS 求解器

详细 API 请参见 `英文 API 文档 <../en/api/index.html>`_。
