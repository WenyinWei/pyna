快速入门
========

场线追踪
--------
.. code-block:: python

   import numpy as np
   from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium
   from pyna.flt import trace_fieldlines

   # 构建 Solov'ev 解析平衡（Cerfon & Freidberg 2010）
   eq = SolovevEquilibrium(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)

   # 追踪场线
   starts = np.array([[1.86 + 0.3*i, 0.0] for i in range(5)])
   result = trace_fieldlines(eq.field_line_rhs, starts, n_turns=100)

庞加莱映射
----------
.. code-block:: python

   from pyna.topo.poincare import poincare_from_fieldlines
   
   section_R = np.linspace(1.2, 2.4, 200)
   points = poincare_from_fieldlines(eq.field_line_rhs, section_R, n_turns=500)

泛函扰动论（FPT）拓扑控制
--------------------------
.. code-block:: python

   from pyna.control.fpt import A_matrix, cycle_shift
   
   # 计算 X 点处的 FPT A 矩阵
   A = A_matrix(eq.field_line_rhs, R_xpt=1.55, Z_xpt=-0.85)
   
   # 预测线圈扰动下的 X 点位移
   shift = cycle_shift(A, delta_g)
   print(f"X 点位移：δR={shift[0]*1000:.2f} mm，δZ={shift[1]*1000:.2f} mm")
