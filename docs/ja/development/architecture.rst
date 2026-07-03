アーキテクチャ
==============

pyna は 2 つの考え方を中心に構成されています。

1. 力学系は有限次元相空間上の発展規則を定義する。
2. topology モジュールは、その相空間に存在する幾何オブジェクトを記述する。

この分離により、同じオブジェクト階層で、トロイダル磁力線構造、Hamiltonian 共鳴領域、
古典写像、N-body orbit、確率サンプルパスを表せます。

Layer 0: Dynamics
-----------------

``pyna.topo.dynamics`` は抽象的な数学レイヤーを提供します。

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` と ``GeneralPoincareMap``

``pyna.dynamics`` はすぐに使える有限次元系を追加します。

- ``CallableFlow`` と ``CallableMap``
- ``HamiltonianSystem`` と ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``、``BrownianMotion``、``GeometricBrownianMotion``

これらのクラスは、サンプル出力に topology core を使います。決定論的 flow trajectory は
``pyna.topo.core.Trajectory`` であり、離散反復の点群は ``pyna.topo.core.Orbit`` です。

Layer 1: Geometry
-----------------

``pyna.topo.core`` は領域に依存しない幾何階層です。

.. list-table::
   :header-rows: 1

   * - クラス
     - 意味
     - 時間タイプ
   * - ``Trajectory``
     - 相空間内の有限サンプル曲線
     - continuous
   * - ``Cycle``
     - 連続 flow の周期軌道
     - continuous
   * - ``Tube``
     - 楕円 cycle 周りの共鳴領域
     - continuous
   * - ``TubeChain``
     - 1 つの共鳴を共有する tube の族
     - continuous
   * - ``Orbit``
     - 写像の有限サンプル反復
     - discrete
   * - ``PeriodicOrbit``
     - 写像の有限周期 orbit
     - discrete
   * - ``Island``
     - 断面上の 1 つの共鳴島
     - discrete
   * - ``IslandChain``
     - 断面上の周期的な島列
     - discrete

重要な bridge は ``section_cut`` です。

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

これは、連続的な磁島 tube が Poincare 断面上で離散的な island chain として観測される
トロイダル workflow を反映しています。

Layer 2: Toroidal Specialization
--------------------------------

``pyna.topo.toroidal`` は汎用 core を subclass します。

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

トロイダル層は次を追加します。

- ``R``、``Z``、``phi`` 座標
- 巻き数 ``(m, n)``
- ``DPm`` と monodromy 分類
- cyna で加速された断面切断と追跡
- 断面ビューの対応と再構成 helper

Layer 3: Workflow and Extension Helpers
---------------------------------------

``pyna.topo.protocols``、``adapters``、``builders``、``bridges``、``factories`` は
ソフトウェア工学上の拡張レイヤーを提供します。notebook 向けの主な入口は
``TopologyWorkflow`` です。これらの helper は、構成方針とバックエンド選択を数学的
dataclass の外に保ちます。外部システムは protocol に従い、adapter でデータを正規化し、
builder でオブジェクトを持ち上げ、bridge で連続幾何を切り、factory で実行時実装を
選択できます。

Layer 4: Acceleration
---------------------

``cyna`` は高レベル pyna API の背後にあるボトルネックを実装します。高レベルな科学的
オブジェクトの意味論を持つべきではありません。追跡、補間、固定点スキャン、壁 hit、
摂動応答の高速カーネルを提供します。

設計ルール
----------

- 新しい有限次元幾何には、汎用の ``pyna.topo.core`` クラスを優先する。
- トロイダル専用フィールドは ``pyna.topo.toroidal`` subclass にのみ追加する。
- サンプルされた有限 trajectory は幾何であり、自動的に不変集合になるわけではない。
- 周期構造がモデルの一部であるか、数値的に検証されている場合だけ
  ``Cycle``/``PeriodicOrbit`` に持ち上げる。
- cyna は bridge 境界に保つ。アプリケーションレベル API は生の C++ 配列ではなく
  pyna オブジェクトを返すべきである。
