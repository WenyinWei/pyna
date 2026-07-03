アーキテクチャ
==============

pyna は 2 つの考え方を中心に構成されています。

1. 力学系は有限次元相空間上の発展規則を定義する。
2. トポロジーモジュールは、その相空間に存在する幾何オブジェクトを記述する。

この分離により、同じオブジェクト階層で、トロイダル磁力線構造、ハミルトン共鳴領域、
古典写像、N 体軌道、確率サンプルパスを表せます。

レイヤー 0: 力学系
------------------

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

これらのクラスは、サンプル出力にトポロジーの中心層を使います。決定論的フロー軌道は
``pyna.topo.core.Trajectory`` であり、離散反復の点群は ``pyna.topo.core.Orbit`` です。

レイヤー 1: 幾何
----------------

``pyna.topo.core`` は領域に依存しない幾何階層です。

.. list-table::
   :header-rows: 1

   * - クラス
     - 意味
     - 時間種別
   * - ``Trajectory``
     - 相空間内の有限サンプル曲線
     - 連続
   * - ``Cycle``
     - 連続フローの周期軌道
     - 連続
   * - ``Tube``
     - 楕円型 cycle 周りの共鳴領域
     - 連続
   * - ``TubeChain``
     - 1 つの共鳴を共有する tube の族
     - 連続
   * - ``Orbit``
     - 写像の有限サンプル反復
     - 離散
   * - ``PeriodicOrbit``
     - 写像の有限周期軌道
     - 離散
   * - ``Island``
     - 断面上の 1 つの共鳴島
     - 離散
   * - ``IslandChain``
     - 断面上の周期的な島列
     - 離散

重要なブリッジは ``section_cut`` です。

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

これは、連続的な磁島 tube が Poincare 断面上で離散的な島列として観測される
トロイダルワークフローを反映しています。

レイヤー 2: トロイダル特殊化
----------------------------

``pyna.topo.toroidal`` は汎用の中心層をサブクラス化します。

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
- 断面ビューの対応と再構成ヘルパー

レイヤー 3: ワークフローと拡張ヘルパー
--------------------------------------

``pyna.topo.protocols``、``adapters``、``builders``、``bridges``、``factories`` は
ソフトウェア工学上の拡張レイヤーを提供します。notebook 向けの主な入口は
``TopologyWorkflow`` です。これらのヘルパーは、構成方針とバックエンド選択を数学的
データクラスの外に保ちます。外部システムはプロトコルに従い、アダプターでデータを正規化し、
ビルダーでオブジェクトを持ち上げ、ブリッジで連続幾何を切り、ファクトリーで実行時実装を
選択できます。

レイヤー 4: 高速化
------------------

``cyna`` は高レベル pyna API の背後にあるボトルネックを実装します。高レベルな科学的
オブジェクトの意味論を持つべきではありません。追跡、補間、固定点スキャン、壁衝突、
摂動応答の高速カーネルを提供します。

設計ルール
----------

- 新しい有限次元幾何には、汎用の ``pyna.topo.core`` クラスを優先する。
- トロイダル専用フィールドは ``pyna.topo.toroidal`` サブクラスにのみ追加する。
- サンプルされた有限軌道は幾何であり、自動的に不変集合になるわけではない。
- 周期構造がモデルの一部であるか、数値的に検証されている場合だけ
  ``Cycle``/``PeriodicOrbit`` に持ち上げる。
- cyna はブリッジ境界に保つ。アプリケーションレベル API は生の C++ 配列ではなく
  pyna オブジェクトを返すべきである。
