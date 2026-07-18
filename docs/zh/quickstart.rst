快速开始
========

本页使用一个不需要外部数据文件的简单解析 tokamak 平衡，带你走过 **pyna** 的三个
核心能力：场线追踪、Poincare 映射和磁岛拓扑。

.. note::

   所有示例都使用 **Solov'ev 解析平衡**（Cerfon & Freidberg 2010），并缩放到接近
   EAST 的参数（R₀ ≈ 1.86 m，B₀ = 5.3 T）。它是一个通用测试平台：精确的
   Grad-Shafranov 解、闭式场分量，以及可调形状。

----

1. 构造解析平衡
----------------

先导入平衡对象并查看它的基本参数：

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"κ  = {eq.kappa:.2f}  δ = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

返回的 ``eq`` 对象提供 ``eq.BR_BZ(R, Z)``、``eq.Bphi(R)``、
``eq.psi(R, Z)``（归一化磁通）和 ``eq.q_profile(psi)``。

----

2. 追踪场线并累积 Poincare 截面点
----------------------------------

Poincare 截面记录场线每次穿过选定环向截面（这里为 φ = 0）时的 (R, Z) 坐标。
经过许多环向圈后，嵌套磁通面会表现为闭合曲线；磁岛则表现为一串离散截面点。

.. code-block:: python

   from pyna.flt import get_backend
   from pyna.topo.poincare import PoincareToroidalSection, poincare_from_fieldlines

   # The accumulator section detects crossings between sampled 3-D points.
   section = PoincareToroidalSection(0.0)

   # --- unit tangent in cylindrical coordinates: dR/dl, dZ/dl, dφ/dl ---
   def field_rhs(rzphi):
       R, Z, _phi = rzphi
       BR, BZ = eq.BR_BZ(R, Z)
       Bphi   = eq.Bphi(R)
       Bnorm  = np.sqrt(BR**2 + BZ**2 + Bphi**2)
       return [BR / Bnorm, BZ / Bnorm, Bphi / (R * Bnorm)]

   # --- seed 8 field lines radially outward from the axis ---
   R_starts = np.linspace(Rmaxis + 0.05, Rmaxis + 0.45, 8)
   Z_starts = np.zeros(8)

   # --- integrate about 80 toroidal turns per line ---
   n_turns = 80
   flt = get_backend('cpu', field_func=field_rhs, dt=0.08)
   pacc = poincare_from_fieldlines(
       field_func=field_rhs,
       start_pts=np.column_stack([R_starts, Z_starts, np.zeros_like(R_starts)]),
       sections=[section],
       t_max=n_turns * 2 * np.pi * Rmaxis,
       backend=flt,
   )
   poincare_pts = pacc.crossing_array(0)[:, :2]

   # --- plot ---
   fig, ax = plt.subplots(figsize=(6, 6))
   ax.scatter(poincare_pts[:, 0], poincare_pts[:, 1], s=0.8, color='steelblue')
   ax.set_xlabel('R (m)')
   ax.set_ylabel('Z (m)')
   ax.set_aspect('equal')
   ax.set_title('Poincaré map — Solov\'ev equilibrium')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/quickstart_poincare.png
   :align: center
   :width: 80%
   :alt: Poincaré map of a Solov'ev analytic equilibrium showing nested flux surfaces

   **图 1.** Solov'ev 解析平衡的 Poincare 映射（接近 EAST 的参数，每条场线
   250 次环向穿越）。每种颜色对应一条场线；嵌套闭合曲线是磁通面。红色叉号标出
   磁轴；黑色曲线是最后闭合磁通面（LCFS, ψ = 1）。

每个同心环对应一条绕磁通面缠绕的场线。q = m/n 有理面是共振扰动（例如 RMP
线圈）可以打开磁岛的位置。

----

3. 定位有理面并测量磁岛
------------------------

加入一个小的共振扰动后，q = 2/1 面上会打开磁岛。pyna 可以在一次调用中定位该面并
测量磁岛半宽：

.. code-block:: python

   from pyna.topo.toroidal_island import locate_rational_surface, island_halfwidth

   # Build q(S) from PEST mesh
   from pyna.toroidal.coords import build_PEST_mesh

   nR, nZ = 100, 100
   R_grid = np.linspace(0.3*eq.R0, 1.5*eq.R0, nR)
   Z_grid = np.linspace(-eq.a*eq.kappa*1.3, eq.a*eq.kappa*1.3, nZ)
   Rg, Zg  = np.meshgrid(R_grid, Z_grid, indexing='ij')

   BR, BZ   = eq.BR_BZ(Rg, Zg)
   Bphi     = eq.Bphi(Rg)
   psi_norm = eq.psi(Rg, Zg)

   S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
       R_grid, Z_grid, BR, BZ, Bphi, psi_norm,
       Rmaxis, Zmaxis, ns=40, ntheta=181
   )
   S_values = S[1:]
   q_values = q_iS[1:]
   print(f"q range: {q_values[0]:.2f} → {q_values[-1]:.2f}")

   # Locate q = 2/1 surface
   res = locate_rational_surface(S_values, q_values, m=2, n=1)
   print(f"q=2/1 surface at S = {res[0]:.4f}  (ψ_norm = {res[0]**2:.4f})")

返回的 ``S_res`` 值（S = √ψ_norm）会准确给出共振层位置。把它和受扰动的
Poincare 映射一起传给 ``island_halfwidth``，即可得到以米为单位的磁岛宽度。

----

4. 一般有限维动力系统
----------------------

pyna 不局限于环形场线。同一套拓扑对象模型也可用于 Hamiltonian 系统、N-body
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
   print(traj.final)  # TimeSeriesSolution is a pyna.topo.core.Trajectory

   linear_map = CallableMap(lambda x: np.array([2*x[0], 0.5*x[1]]), dim=2)
   orbit = linear_map.orbit_geometry([1.0, 1.0], n_iter=5)
   print(orbit.period_guess)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
   print(gbm.expected_log_growth())

当一条轨迹或映射轨道已经从采样数据提升为几何/拓扑对象时，可使用
:mod:`pyna.topo.core` 中的 ``Cycle``、``PeriodicOrbit``、``Tube`` 和
``IslandChain`` 等对象。

----

5. 基于工作流的构造
--------------------

在较大的项目和教学 notebook 中，使用 ``TopologyWorkflow`` 可以让分析序列保持
显式，而不用在代码中散布临时构造器。

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

低层 adapter、builder、bridge 和 factory 仍然供库作者使用，但大多数 notebook
应从 workflow facade 开始。

----

6. 下一步
---------

- **教程** -- 带图的完整示例：
  :doc:`/zh/mini-cases`,
  :doc:`/zh/tutorials/sde-monte-carlo`,
  :doc:`/notebooks/i18n/zh/tutorials/RMP_resonance_analysis`,
  :doc:`/notebooks/i18n/zh/tutorials/magnetic_coordinates_comparison`,
  :doc:`/notebooks/i18n/zh/tutorials/RMP_island_validation_solovev`

- **API 参考** -- 完整 docstring：
  :doc:`/zh/api/index`

- **CUDA 加速** -- 安装 ``cupy-cuda12x``，并把
  ``backend=get_backend('cuda')`` 传给 tracer，即可在磁岛宽度扫描中获得最高约
  100× 的加速。
