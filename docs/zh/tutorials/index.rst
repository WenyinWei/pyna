教程与案例
==========

公开教程按工作流组织。文档构建会把 ``notebooks/`` 复制到 Sphinx 源目录，因此
下面的路径与仓库布局一致。大计算量 notebook 在本地预执行并提交输出，GitHub
Pages 只负责渲染，避免在 CI 里重复做 Monte Carlo 或长时间场线追踪。

推荐学习路径
------------

先读 :doc:`/zh/quickstart`，再依次看通用几何模式、随机模型，以及环形
monodromy/RMP 案例：

1. :doc:`/zh/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

通用动力系统
------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

通用几何工作流和解析三维磁场定点工作流不再作为单独的纯文本 notebook 发布；
它们的核心链路已经并入 RMP resonance tutorial：采样交点 -> 定点几何 ->
X/O 分类 -> 流形与坐标网格叠加。

随机微分方程
------------

SDE 教程在本地预执行，因为分布估计通常要用数万到数十万条 Monte Carlo
路径。GitHub Pages 渲染保存的输出，而不是在 CI 里重新执行重采样单元。

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

磁坐标与平衡
------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMP、岛链与 Poincare 分析
-------------------------

学习磁拓扑时，建议从 resonance analysis notebook 开始。它覆盖无散度 RMP
模板、重要的 ``m=1`` 分支、``cyna`` 定点校验、多分量逆变 ``B^r`` 磁谱
atlas、可选叠加 Poincare 点迹与岛宽条的 ``q``/``m/n`` 共振图、混合
RMP/nRMP 谱、所有非共振模式求和得到的总 nRMP 响应、场线流速调制和扰动
阶次检查。

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` 保留在仓库中，作为 resonance analysis 工作流的
执行/缓存变体；公开文档链接到上面的解释性版本。

Monodromy 与流形
----------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

经典与通用动力系统
------------------

仓库还包含 ``notebooks/examples`` 下的轻量 notebook：
``Lorenz_attractor.ipynb``、``resonance_1_1_map.ipynb``、
``Mobiusian_saddle_cycle.ipynb``、``Xcycle_construction.ipynb`` 和
``FPT_DX_to_DP_sympy.ipynb``。它们作为源码示例保留，而不是作为已执行文档
页面发布，因为其中若干 notebook 偏 scratch 风格且缺少稳定章节标题。

静态教程图
----------

几个更长的工作流以静态图和生成输出的形式保存在 ``notebooks/tutorials`` 下。
它们覆盖 q-profile 诊断、PEST/Boozer/Hamada/equal-arc 坐标、岛链抑制扫描、
相位控制、Poincare 流形和 Solov'ev single-null 示例。

这些 notebook 暂时以英文为主；右上角语言下拉框在没有目标语言页面时会回退
到英文版本，并在英文 fallback 页面顶部显示缺失翻译提示。
