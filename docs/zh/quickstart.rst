.. _zh-quickstart:

快速入门
========

这一页用一个不需要外部数据文件的简单解析 tokamak 平衡，串起 **pyna** 的三项
核心能力：场线追踪、Poincare 映射和岛链拓扑。

.. note::

   示例使用 **Solov'ev 解析平衡**（Cerfon & Freidberg 2010），并缩放到
   EAST-like 参数（R0 约 1.86 m，B0 = 5.3 T）。它适合作为通用测试床：
   Grad-Shafranov 方程有解析解，磁场分量闭式可得，形状参数也可调。

----

0. 安装验证
-----------

.. code-block:: python

   import pyna
   from pyna._cyna import is_available, get_version

   print(pyna.__version__)
   print(is_available(), get_version())

``is_available()`` 应为 ``True``。如果失败，请先检查 :doc:`installation`
中的 cyna/xmake 构建说明。

----

1. 构造解析平衡
---------------

先导入平衡并查看基本参数：

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"kappa = {eq.kappa:.2f}  delta = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

返回的 ``eq`` 对象提供 ``eq.BR_BZ(R, Z)``、``eq.Bphi(R)``、
``eq.psi(R, Z)``（归一化磁通）和 ``eq.q_profile(psi)``。

----

2. 追踪场线并累积 Poincare 截面点
---------------------------------

Poincare 截面记录每次场线穿过指定环向截面时的 ``(R, Z)`` 坐标。经过很多
环向圈后，嵌套磁面表现为闭合曲线；磁岛则表现为一串离散截面点。

.. code-block:: python

   from pyna.flt import FieldLineTracer, get_backend
   from pyna.topo.poincare import poincare_from_fieldlines
   from pyna.topo.section import ToroidalSection

   section = ToroidalSection(0.0)

   def field_rhs(phi, RZ):
       R, Z = RZ
       BR, BZ = eq.BR_BZ(R, Z)
       Bphi = eq.Bphi(R)
       return [R * BR / Bphi, R * BZ / Bphi]

   R_starts = np.linspace(Rmaxis + 0.05, Rmaxis + 0.45, 8)
   Z_starts = np.zeros(8)

   backend = get_backend("cpu")
   flt = FieldLineTracer(field_rhs, backend=backend)
   pacc = poincare_from_fieldlines(
       field_func=field_rhs,
       start_pts=np.column_stack([R_starts, Z_starts, np.zeros_like(R_starts)]),
       sections=[section],
       t_max=300 * 2 * np.pi,
       backend=flt,
   )
   poincare_pts = [pacc.crossing_array(0)[:, :2]]

   fig, ax = plt.subplots(figsize=(6, 6))
   for Rs, Zs in poincare_pts:
       ax.scatter(Rs, Zs, s=0.8, color="steelblue")
   ax.set_xlabel("R (m)")
   ax.set_ylabel("Z (m)")
   ax.set_aspect("equal")
   ax.set_title("Poincare map -- Solov'ev equilibrium")
   plt.tight_layout()
   plt.show()

.. figure:: /_static/quickstart_poincare.png
   :align: center
   :width: 80%
   :alt: Solov'ev 解析平衡的 Poincare 图，显示嵌套磁面

   **图 1.** Solov'ev 解析平衡的 Poincare 图。每种颜色对应一条场线；
   嵌套闭合曲线对应磁面。红色叉号标出磁轴，黑色曲线是最后闭合磁面
   （LCFS, psi = 1）。

每一圈同心曲线对应一条绕磁面缠绕的场线。``q = m/n`` 有理面是共振扰动
（例如 RMP 线圈）可能打开磁岛的位置。

----

3. 定位有理面并测量岛宽
-----------------------

加入小的共振扰动后，``q = 2/1`` 面上会打开磁岛。pyna 可以定位该面，并在同一
工作流中测量岛半宽：

.. code-block:: python

   from pyna.topo.toroidal_island import locate_rational_surface, island_halfwidth
   from pyna.toroidal.coords import build_PEST_mesh

   nR, nZ = 100, 100
   R_grid = np.linspace(0.3 * eq.R0, 1.5 * eq.R0, nR)
   Z_grid = np.linspace(-eq.a * eq.kappa * 1.3, eq.a * eq.kappa * 1.3, nZ)
   Rg, Zg = np.meshgrid(R_grid, Z_grid, indexing="ij")

   BR, BZ = eq.BR_BZ(Rg, Zg)
   Bphi = eq.Bphi(Rg)
   psi_norm = eq.psi(Rg, Zg)

   S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
       R_grid, Z_grid, BR, BZ, Bphi, psi_norm,
       Rmaxis, Zmaxis, ns=40, ntheta=181
   )
   S_values = S[1:]
   q_values = q_iS[1:]
   print(f"q range: {q_values[0]:.2f} -> {q_values[-1]:.2f}")

   res = locate_rational_surface(S_values, q_values, m=2, n=1)
   print(f"q=2/1 surface at S = {res[0]:.4f}  (psi_norm = {res[0]**2:.4f})")

返回的 ``S_res``（``S = sqrt(psi_norm)``）给出共振层位置。把它和扰动后的
Poincare 图一起传给 ``island_halfwidth``，即可得到以米为单位的岛宽。

----

4. 通用有限维动力系统
---------------------

pyna 不限于环形场线。同一套拓扑对象模型也可用于 Hamiltonian 系统、N-body
流、映射和 SDE 样本路径。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import (
       SeparableHamiltonianSystem,
       CallableMap,
       GeometricBrownianMotion,
   )

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   traj = oscillator.trajectory([1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(traj.final)

   linear_map = CallableMap(lambda x: np.array([2*x[0], 0.5*x[1]]), dim=2)
   orbit = linear_map.orbit_geometry([1.0, 1.0], n_iter=5)
   print(orbit.period_guess)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
   print(gbm.expected_log_growth())

当轨道或映射迭代样本被提升为几何/拓扑对象时，可使用
:mod:`pyna.topo.core` 中的 ``Cycle``、``PeriodicOrbit``、``Tube`` 和
``IslandChain`` 等对象。

----

5. 使用工作流门面构造对象
-------------------------

大型项目和教学 notebook 建议使用 ``TopologyWorkflow``，把分析顺序保持为显式
线性流程，避免在代码里散落临时构造函数。

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow
   from pyna.topo.section import HyperplaneSection

   wf = TopologyWorkflow(closure_tol=1e-3)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
   pmap = wf.poincare_map(flow, section, dt=0.02)

   closed_traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   cycle = wf.closed_cycle(closed_traj)

底层 adapter、builder、bridge 和 factory 仍然面向库作者开放；多数 notebook
应优先从 workflow facade 开始。

----

6. 下一步
---------

- **教程与案例**：
  :doc:`/zh/mini-cases`、
  :doc:`/zh/tutorials/sde-monte-carlo`、
  :doc:`/notebooks/tutorials/RMP_resonance_analysis`、
  :doc:`/notebooks/tutorials/magnetic_coordinates_comparison`、
  :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

- **API 参考**：
  :doc:`/zh/api/index` 和 :doc:`/en/api/index`

- **CUDA 加速**：
  安装 ``cupy-cuda12x``，并在追踪器中传入 ``backend=get_backend("cuda")``，
  用于大批量岛宽扫描。
