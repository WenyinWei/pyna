教程和示例
==========

公共 notebook 按 workflow 分组。文档构建会把 ``notebooks/`` 复制到 Sphinx 源树，
因此下面的路径与仓库布局一致。

推荐学习路径
------------

先读 :doc:`/zh/quickstart`，再依次学习通用几何 workflow、随机模型，以及环形
monodromy/RMP 示例：

1. :doc:`/zh/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

一般动力系统
------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

通用几何 workflow 和解析 stellarator 固定点 workflow 现在已经并入 RMP resonance
tutorial，而不是作为独立的纯文本 notebook 发布。该教程展示同一条提升链：
sampled crossings -> fixed-point geometry -> X/O classification -> manifold
and coordinate-grid overlays。

随机微分方程
------------

SDE 教程会在本地预执行，因为分布估计通常使用数万或数十万条 Monte Carlo 路径。
GitHub Pages 渲染保存的输出，而不是在 CI 上消耗时间运行重型采样 cell。

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

磁坐标和平衡
------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMP、磁岛和 Poincare 分析
--------------------------

研究磁拓扑时，请从 resonance analysis notebook 开始。它现在覆盖无散度 RMP 模板、
重要的 ``m=1`` 分支、``cyna`` 固定点验证、多分量逆变 ``B^r`` 磁谱图册、带可选
Poincare 和磁岛叠加的模块化 ``q``/``m/n`` 共振图、混合 RMP/nRMP 谱、来自所有
非共振模式的总 nRMP 响应、场线速度调制，以及扰动阶数检查。

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` 保留在仓库中，作为 resonance analysis workflow 的
执行/cache 变体，但公开文档链接到上面的解释版本。

Monodromy 和流形
----------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

经典和一般动力系统
------------------

仓库还在 ``notebooks/examples`` 下包含轻量 notebook：
``Lorenz_attractor.ipynb``、``resonance_1_1_map.ipynb``、
``Mobiusian_saddle_cycle.ipynb``、``Xcycle_construction.ipynb`` 和
``FPT_DX_to_DP_sympy.ipynb``。它们作为源示例保留，而不是作为已执行文档页面发布，
因为其中几个是没有章节标题的 scratch-style notebook。

静态教程图
----------

若干较长 workflow 在仓库中以 ``notebooks/tutorials`` 下的静态图和生成输出表示。
它们覆盖 q-profile 诊断、PEST/Boozer/Hamada/equal-arc 坐标、磁岛抑制扫描、
相位控制、Poincare 流形和 Solov'ev single-null 示例。
