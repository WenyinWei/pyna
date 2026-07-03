pyna - Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-LGPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** 是一个面向 **动力系统分析** 和 **磁约束聚变物理** 的 Python
库。它覆盖场线追踪、Poincare 映射、Hamiltonian 系统、N-body 相互作用、
有限维映射、Ito 随机微分方程，以及把采样数据提升为相空间几何对象时共用的
拓扑词汇。

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 快速开始
      :link: quickstart
      :link-type: doc

      安装、验证 cyna，并运行第一个环形场线和通用动力系统示例。

   .. grid-item-card:: 迷你案例
      :link: mini-cases
      :link-type: doc

      ODE、Hamiltonian 系统、映射、SDE 和拓扑对象提升的简短配方。

   .. grid-item-card:: 教程
      :link: tutorials/index
      :link-type: doc

      已执行 notebook 和叙述式指南，包括 Monte Carlo SDE 分布估计。

   .. grid-item-card:: API 参考
      :link: api/index
      :link-type: doc

      手写模块指南，以及从源码生成的完整参考入口。

.. toctree::
   :maxdepth: 2
   :caption: 文档

   installation
   quickstart
   mini-cases
   tutorials/index
   api/index
   theory/index
   development/index
   straight_theta_surface_coordinates
