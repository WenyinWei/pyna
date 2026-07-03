クイックスタート
================

このページでは、外部データファイルを必要としない単純な解析的 tokamak 平衡を使い、
**pyna** の 3 つの中核機能である磁力線追跡、Poincare 写像、磁島トポロジーを順に
確認します。

.. note::

   すべての例では **Solov'ev 解析平衡**（Cerfon & Freidberg 2010）を使い、
   EAST に近いパラメータ（R₀ ≈ 1.86 m、B₀ = 5.3 T）にスケールしています。これは
   汎用的なテストベッドとして有用です。厳密な Grad-Shafranov 解、閉形式の場成分、
   調整可能な形状を備えています。

----

1. 解析平衡を構築する
---------------------

まず平衡を import し、基本パラメータを確認します。

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"κ  = {eq.kappa:.2f}  δ = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

返される ``eq`` オブジェクトは ``eq.BR_BZ(R, Z)``、``eq.Bphi(R)``、
``eq.psi(R, Z)``（規格化磁束）、``eq.q_profile(psi)`` を公開します。

----

2. 磁力線を追跡し Poincare 交差を蓄積する
------------------------------------------

Poincare 断面は、磁力線が選んだトロイダル断面（ここでは φ = 0）を横切るたびに
(R, Z) 座標を記録します。多くのトロイダル周回の後、入れ子状の磁束面は閉曲線として
現れ、磁島は離散的な断面点の列として現れます。

.. code-block:: python

   from pyna.flt import FieldLineTracer, get_backend
   from pyna.topo.poincare import PoincareAccumulator, poincare_from_fieldlines
   from pyna.topo.section import ToroidalSection

   # Use the canonical topology section type; ``pyna.topo.poincare`` keeps
   # backward-compatible aliases for accumulator-only workflows.
   section = ToroidalSection(0.0)

   # --- define the ODE right-hand side: dR/dφ, dZ/dφ ---
   def field_rhs(phi, RZ):
       R, Z = RZ
       BR, BZ = eq.BR_BZ(R, Z)
       Bphi   = eq.Bphi(R)
       return [R * BR / Bphi, R * BZ / Bphi]

   # --- seed 8 field lines radially outward from the axis ---
   R_starts = np.linspace(Rmaxis + 0.05, Rmaxis + 0.45, 8)
   Z_starts = np.zeros(8)

   # --- integrate 300 toroidal turns per line ---
   backend = get_backend('cpu')
   flt = FieldLineTracer(field_rhs, backend=backend)
   pacc = poincare_from_fieldlines(
       field_func=field_rhs,
       start_pts=np.column_stack([R_starts, Z_starts, np.zeros_like(R_starts)]),
       sections=[section],
       t_max=300 * 2 * np.pi,
       backend=flt,
   )
   poincare_pts = [pacc.crossing_array(0)[:, :2]]

   # --- plot ---
   fig, ax = plt.subplots(figsize=(6, 6))
   for Rs, Zs in poincare_pts:
       ax.scatter(Rs, Zs, s=0.8, color='steelblue')
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

   **図 1.** Solov'ev 解析平衡の Poincare 写像（EAST に近いパラメータ、各磁力線
   250 回のトロイダル通過）。各色は 1 本の磁力線に対応します。入れ子状の閉曲線は
   磁束面です。赤い十字は磁気軸、黒い曲線は最後の閉じた磁束面（LCFS, ψ = 1）を
   示します。

各同心リングは、1 本の磁力線が磁束面の周りを巻いていることに対応します。q = m/n の
有理面は、RMP コイルなどの共鳴摂動によって磁島が開く場所です。

----

3. 有理面を見つけ磁島を測る
---------------------------

小さな共鳴摂動を加えると、q = 2/1 面に磁島が開きます。pyna はその面を特定し、
磁島半幅を 1 回の呼び出しで測定できます。

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

返される ``S_res`` 値（S = √ψ_norm）は、共鳴層がどこにあるかを正確に示します。
これを摂動後の Poincare 写像とともに ``island_halfwidth`` に渡すと、磁島幅をメートル
単位で得られます。

----

4. 一般の有限次元力学
---------------------

pyna はトロイダル磁力線に限られません。同じトポロジーオブジェクトモデルは
Hamiltonian 系、N-body flow、写像、SDE サンプルパスにも使えます。

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

軌道や写像 orbit がサンプルデータから幾何/トポロジーオブジェクトへ持ち上げられた
ときは、:mod:`pyna.topo.core` の ``Cycle``、``PeriodicOrbit``、``Tube``、
``IslandChain`` などを使います。

----

5. Workflow ベースの構成
------------------------

大きなプロジェクトや教育用 notebook では、``TopologyWorkflow`` を使うことで、場当たり的な
constructor をコード中に散らさず、解析手順を明示的に保てます。

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

低レベルの adapter、builder、bridge、factory はライブラリ作者向けに引き続き利用
できますが、多くの notebook は workflow facade から始めるべきです。

----

6. 次のステップ
---------------

- **チュートリアル** -- プロット付きの実例:
  :doc:`/ja/mini-cases`,
  :doc:`/ja/tutorials/sde-monte-carlo`,
  :doc:`/notebooks/tutorials/RMP_resonance_analysis`,
  :doc:`/notebooks/tutorials/magnetic_coordinates_comparison`,
  :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

- **API リファレンス** -- 完全な docstring:
  :doc:`/ja/api/index`

- **CUDA 加速** -- ``cupy-cuda12x`` をインストールし、島幅スキャンで
  ``backend=get_backend('cuda')`` を tracer に渡すと、最大 100× の高速化が得られます。
