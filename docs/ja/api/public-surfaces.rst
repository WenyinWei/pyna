Public API マップ
=================

このページは、pyna の安定した公開インターフェースへの短い導線です。完全な
デバッグ用リファレンスは AutoAPI に残し、ここでは notebook、研究スクリプト、
下流パッケージで優先して使う入口だけをまとめます。

幾何オブジェクト
----------------

ソルバーそのものより相空間内の対象を扱いたいときは、ここから始めます。

.. list-table::
   :header-rows: 1

   * - 目的
     - 公開入口
   * - 連続時間のサンプル軌道
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - 離散写像の力学
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - トロイダル断面幾何
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - 明示的な昇格と adapter
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

一般力学系
----------

トロイダルでないモデルでも同じ幾何オブジェクトを返したい場合は
:mod:`pyna.dynamics` を使います。

.. list-table::
   :header-rows: 1

   * - モデル
     - 公開入口
   * - ODE flow
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Hamiltonian 系
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N 体問題
     - :class:`pyna.dynamics.NBodySystem`
   * - 離散写像
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - SDE
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

トロイダル/RMP ワークフロー
---------------------------

磁気座標、磁力線追跡、磁気スペクトル解析、図の overlay には以下を使います。

.. list-table::
   :header-rows: 1

   * - 必要なもの
     - 公開入口
   * - 平衡と座標
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - 磁力線追跡と cache-aware workflow
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - 反変径方向摂動スペクトル
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - RMP/nRMP 診断
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - 磁気スペクトル図
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Poincare、X/O 点、磁島 overlay
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

AutoAPI を見る場合
------------------

constructor signature、継承 member、低頻度の診断、実装詳細が必要なときは
:doc:`/en/api/generated/pyna/index` を参照してください。新しい tutorial や
ユーザー向け例は、できるだけ上の公開入口だけで書くのが推奨です。
