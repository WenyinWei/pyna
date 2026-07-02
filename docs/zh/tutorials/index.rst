教程与案例
==========

公开教程按工作流组织。大计算量 notebook 会在本地预执行并提交输出，GitHub
Pages 只负责渲染，避免在 CPU 数量有限的 workflow 里重复做 Monte Carlo。

推荐学习路径
------------

1. :doc:`/zh/quickstart`
2. :doc:`/zh/mini-cases`
3. :doc:`sde-monte-carlo`
4. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
5. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
6. :doc:`/notebooks/tutorials/island_jacobian_analysis`
7. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

通用动力系统与随机微分方程
--------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

共享的英文可执行 notebook：

- :doc:`/notebooks/tutorials/sde_monte_carlo_distribution`

环形场线、岛链和流形
--------------------

磁拓扑学习建议从 RMP resonance analysis 开始。它现在覆盖无散度 RMP 模板、
重要的 ``m=1`` 分支、``cyna`` 定点校验、混合 RMP/nRMP 谱、所有非共振
模式求和得到的总 nRMP 响应、场线流速调制和扰动阶次检查。

- :doc:`/notebooks/tutorials/magnetic_coordinates_comparison`
- :doc:`/notebooks/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/tutorials/RMP_island_validation_solovev`
- :doc:`/notebooks/tutorials/island_jacobian_analysis`
- :doc:`/notebooks/tutorials/monodromy_mobius_saddle`
- :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`

通用几何工作流和解析 stellarator 定点工作流不再作为单独无图页面发布；它们
的核心思想已并入 RMP resonance tutorial：从 Poincare 交点、解析 X/O 点、
局部流形到 PEST 网格叠加，形成一个完整可视化案例。

这些 notebook 暂时以英文为主；右上角语言下拉框在没有目标语言页面时会回退
到英文版本。中文页面负责说明学习路径、核心概念和扩展点。
