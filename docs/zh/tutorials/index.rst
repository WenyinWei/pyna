教程和示例
==========

公共 notebook 按工作流分组。文档构建会把 ``notebooks/`` 复制到 Sphinx 源树，
因此下面的路径与仓库布局一致。

推荐学习路径
------------

先读 :doc:`/zh/quickstart`，再依次学习通用几何工作流、随机模型，以及环形
Monodromy/RMP 示例：

1. :doc:`/zh/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/zh/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/zh/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/zh/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/zh/tutorials/RMP_island_validation_solovev`

按目标选择入口
--------------

.. list-table::
   :header-rows: 1

   * - 目标
     - 先读
     - 继续阅读
   * - 掌握 pyna 基本工作流
     - :doc:`/zh/quickstart`
     - :doc:`/zh/mini-cases`
   * - 估计 SDE 分布
     - :doc:`sde-monte-carlo`
     - :doc:`/notebooks/i18n/zh/tutorials/sde_monte_carlo_distribution`
   * - 比较磁坐标
     - :doc:`/notebooks/i18n/zh/tutorials/magnetic_coordinates_comparison`
     - :doc:`/zh/api/public-surfaces`
   * - 研究 RMP/nRMP 磁拓扑
     - :doc:`/notebooks/i18n/zh/tutorials/RMP_resonance_analysis`
     - :doc:`/notebooks/i18n/zh/tutorials/RMP_island_validation_solovev`,
       :doc:`/notebooks/i18n/zh/tutorials/island_jacobian_analysis`
   * - 分析 Monodromy 和流形
     - :doc:`/notebooks/i18n/zh/tutorials/monodromy_xcycle_analytic`
     - :doc:`/notebooks/i18n/zh/tutorials/monodromy_mobius_saddle`

一般动力系统
------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

通用几何工作流和解析仿星器固定点工作流现在已经并入 RMP 共振教程，
而不是作为独立的纯文本 notebook 发布。该教程展示同一条提升链：
采样交点 -> 不动点几何 -> X/O 分类 -> 流形与坐标网格叠加。

随机微分方程
------------

SDE 教程会在本地预执行，因为分布估计通常使用数万或数十万条蒙特卡洛路径。
GitHub Pages 渲染保存的输出，而不是在 CI 上消耗时间运行重型采样 cell。

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/zh/tutorials/sde_monte_carlo_distribution

磁坐标和平衡
------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/zh/tutorials/magnetic_coordinates_comparison

RMP、磁岛和 Poincaré 分析
--------------------------

研究磁拓扑时，请从共振分析 notebook 开始。它首先用可执行的 convention lock
复现 Nardon 论文式 (3.3)-(3.17)，固定有符号 ``(m,n)`` 指标、实场共轭、共振支选择、
磁岛半宽和 Chirikov 重叠定义。随后覆盖无散度 RMP 模板、重要的 ``m=1`` 分支、
``cyna`` 固定点验证、多分量逆变 ``B^r`` 磁谱图册、带可选
Poincaré 和磁岛叠加的模块化 ``q``/``m/n`` 共振图、混合 RMP/nRMP 谱、来自所有
非共振模的总 nRMP 响应、磁力线速度调制，以及扰动阶数检查。

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/zh/tutorials/RMP_resonance_analysis
   /notebooks/i18n/zh/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/zh/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` 保留在仓库中，作为共振分析工作流的
执行/cache 变体，但公开文档链接到上面的解释版本。

Monodromy 和流形
----------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/zh/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/zh/tutorials/monodromy_xcycle_analytic

经典和一般动力系统
------------------

仓库还在 ``notebooks/examples`` 下包含轻量 notebook：
``Lorenz_attractor.ipynb``、``resonance_1_1_map.ipynb``、
``Mobiusian_saddle_cycle.ipynb``、``Xcycle_construction.ipynb`` 和
``FPT_DX_to_DP_sympy.ipynb``。它们作为源示例保留，而不是作为已执行文档页面发布，
因为其中几个是没有章节标题的草稿式 notebook。

静态教程图
----------

若干较长工作流在仓库中以 ``notebooks/tutorials`` 下的静态图和生成输出表示。
它们覆盖 ``q`` 剖面诊断、PEST/Boozer/Hamada/等弧长坐标、磁岛抑制扫描、
相位控制、Poincaré 流形和 Solov'ev 单零点示例。
