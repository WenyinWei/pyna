.. _zh-straight-theta-surface-coordinates:

基于 Poincare 迹线构建 straight-theta 磁面坐标
================================================

本教程介绍 pyna 中当前推荐的嵌套闭合磁面构建流程。目标是从真实三维磁场的
Poincare 迹线构造可靠的 straight-theta / PEST-like 磁面网格。Boozer 坐标不在
本文范围内；它可以在已有可靠磁面网格上作为后处理继续构造。

核心原则
--------

这个流程有三个基本判断：

1. 场线追踪和 Poincare 点采样应使用 cyna/pyna 的批量接口，不应在 Python 层写慢
   field-line tracing。
2. 磁面几何主要来自同一条磁力线多圈 Poincare 点的粘合；不要用 Python Fourier
   拟合或曲线拟合去补救错误排序。
3. 近磁轴处如果 Poincare 点排序不稳定，就不要把这些点作为源磁面。用磁轴到最内
   可靠磁面的插值填充芯部，并可用轴上的 ``DX_pol`` 特征值给 ``iota_axis`` 提供
   约束。

相关 API
--------

通用数组工具位于 ``pyna.toroidal.surface_coordinates``：

.. code-block:: python

   from pyna.toroidal.surface_coordinates import (
       circle_map_lift_iota,
       insert_axis_core_surfaces,
       periodic_shift_theta,
       rank_phase_from_axis,
       stitch_periodic,
       theta_coverage,
   )

其中最重要的是：

``rank_phase_from_axis(R, Z, axis_R, axis_Z)``
   对单个截面上一条闭合 Poincare 曲线按几何角排序，返回该闭合曲线上的 rank phase。
   几何角只用于排序，不直接作为最终 straight theta。

``circle_map_lift_iota(phase)``
   用 Poincare 点的 turn index 拟合
   ``phase_k = phase0 + 2*pi*iota*k (mod 2*pi)``。这一步保留了圈数信息，避免直接
   对局部相邻角做 unwrap 时遇到的模数分支问题。

``stitch_periodic(theta, values, target_theta)``
   把散落在周期 theta 上的 ``R`` 或 ``Z`` 点粘合到统一 theta 网格。

``insert_axis_core_surfaces(...)``
   在磁轴和最内可靠磁面之间插入若干芯部磁面，避免使用近轴不稳定 Poincare 源面。

``progress_DX_pol_along_orbit``
   位于 ``pyna.toroidal.flt``。它沿一条已经采样好的轨道推进
   ``DX_pol(phi_e, phi_s)``，不要求轨道是周期轨道，也不会重新追踪磁力线。

最小构建流程
------------

假设你已经用 cyna/pyna 得到了若干可靠磁面的 Poincare 数据：

- ``R_hits[s, k]``、``Z_hits[s, k]``：第 ``s`` 个环向截面上，第 ``k`` 圈的点。
- ``phi_sections[s]``：每个截面的环向角。
- ``axis_R[s]``、``axis_Z[s]``：每个截面上的磁轴位置。
- ``theta_grid``：希望输出的统一 straight-theta 网格。

单个磁面的核心步骤如下：

.. code-block:: python

   import numpy as np

   from pyna.toroidal.surface_coordinates import (
       circle_map_lift_iota,
       rank_phase_from_axis,
       stitch_periodic,
       theta_coverage,
   )

   TWOPI = 2.0 * np.pi

   iota_estimates = []
   per_section = []

   for s, phi in enumerate(phi_sections):
       rr = np.asarray(R_hits[s], dtype=float)
       zz = np.asarray(Z_hits[s], dtype=float)
       keep = np.isfinite(rr) & np.isfinite(zz)
       rr = rr[keep]
       zz = zz[keep]

       phase = rank_phase_from_axis(rr, zz, axis_R[s], axis_Z[s])
       iota, rms = circle_map_lift_iota(phase)
       if np.isfinite(iota):
           iota_estimates.append(iota)
       per_section.append((rr, zz))

   iota = float(np.nanmedian(iota_estimates))

   R_surface = np.full((len(phi_sections), theta_grid.size), np.nan)
   Z_surface = np.full_like(R_surface, np.nan)

   for s, phi in enumerate(phi_sections):
       rr, zz = per_section[s]
       turns = np.arange(rr.size, dtype=float)
       theta_straight = iota * (float(phi) + turns * TWOPI)

       if theta_coverage(theta_straight, bins=48) < 0.75:
           continue

       R_surface[s] = stitch_periodic(theta_straight, rr, theta_grid)
       Z_surface[s] = stitch_periodic(theta_straight, zz, theta_grid)

最终 ``R_surface[s, j]`` 和 ``Z_surface[s, j]`` 就是该磁面在第 ``s`` 个截面、第
``j`` 个 straight theta 上的坐标。

加入磁轴到最内可靠磁面的芯部插值
--------------------------------

真实数据中最靠近磁轴的 Poincare 点经常因为半径太小、磁轴定位误差或几何角退化而
排序不稳定。推荐做法是选择稍靠外的第一个可靠磁面作为源面，然后用线性插值填充
磁轴到该源面之间的芯部。

.. code-block:: python

   from pyna.toroidal.surface_coordinates import insert_axis_core_surfaces

   # R_surf, Z_surf shape: (n_phi, n_reliable_surfaces, n_theta)
   # radial_labels shape: (n_reliable_surfaces,)
   core = insert_axis_core_surfaces(
       R_surf,
       Z_surf,
       radial_labels,
       axis_R,
       axis_Z,
       fractions=[0.25, 0.5, 0.75],
   )

   R_surf = core.R_surf
   Z_surf = core.Z_surf
   radial_labels = core.radial_labels

这样插入的芯部磁面严格连接磁轴和第一个可靠磁面，不依赖近轴 Poincare 点的极角
排序。

用 DX_pol 估计轴上 iota
----------------------

如果需要对芯部 iota 加端点约束，可以沿磁轴轨道推进 ``DX_pol``，观察其特征值相位：

.. code-block:: python

   import numpy as np
   from pyna.toroidal.flt import progress_DX_pol_along_orbit

   DX = progress_DX_pol_along_orbit(
       R_axis_traj,
       Z_axis_traj,
       phi_traj,
       R_grid,
       Z_grid,
       Phi_grid,
       BR_flat,
       BZ_flat,
       BPhi_flat,
       max_step=0.005,
   )

   eig = np.linalg.eigvals(DX[-1])
   phase = abs(np.angle(eig[np.argmax(np.imag(eig))]))
   iota_axis = phase / (phi_traj[-1] - phi_traj[0])

``progress_DX_pol_along_orbit`` 的输入是一条已经采样好的轨道，因此它可用于周期轨道、
非周期轨道，以及一般的 ``DX_pol`` 动态诊断。

实际使用建议
------------

- 先筛掉近轴排序不稳定的 Poincare 源面。
- 对每个候选源面检查 ``theta_coverage``、相邻径向连接、二阶差分和等 theta 线图。
- 对低阶有理面附近的源面保持谨慎；必要时只把它作为目标插值位置，不作为源面。
- 芯部几何优先用磁轴到最内可靠磁面的插值，不要强行追踪极靠近磁轴的面。
- 如果某个装置的 Poincare 点有天然顺序，也仍应保留 turn index，并使用
  ``circle_map_lift_iota`` 解决模数分支问题。

在 NCSX 和 W7X 真空场测试中，这个流程已经能稳定生成四截面的嵌套磁面和等
theta 网格。topoquest 中的装置专用脚本可以作为真实数据 IO 和可视化的参考；pyna
中保留的是可复用的数组构建核心。
