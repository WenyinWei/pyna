連続幾何と離散幾何
==================

pyna は、連続時間と離散時間の力学系に別々のオブジェクト族を使います。

連続時間側:

- ``Trajectory`` はサンプルされた有限時間幾何です。
- ``Cycle`` は flow の周期軌道です。
- ``Tube`` は楕円 cycle の周りの共鳴領域で、双曲 cycle によって境界づけられる場合が
  あります。
- ``TubeChain`` は同じ共鳴に属する tube をまとめます。

離散時間側:

- ``Orbit`` はサンプルされた写像反復の幾何です。
- ``PeriodicOrbit`` は写像の閉じた orbit です。
- ``Island`` は断面上の 1 つの縮約共鳴島です。
- ``IslandChain`` は断面レベルの島列です。

両側をつなぐ bridge は断面切断です。``Cycle`` を Poincare 断面で切ると、return map の
``PeriodicOrbit`` が得られます。``Tube`` を切ると ``IslandChain`` が得られます。
``TubeChain`` を切ると、その tube から得られる island chain がマージされます。

この分離は意図的です。数値 trajectory は、不変性を証明しなくても有用な幾何であり得ます。
そのため builder と adapter は持ち上げを明示的にします。ユーザーは、サンプル trajectory
が ``Cycle`` になる前、また写像サンプルが ``PeriodicOrbit`` になる前に、閉合チェックを
要求できます。

同じ語彙は、汎用の有限次元系とトロイダル磁力線特殊化で共有されます。汎用ルートは
``pyna.topo.CoreTube`` などの名前で利用できます。トロイダルの既定値は
``pyna.topo.Tube``、``pyna.topo.Cycle``、``pyna.topo.IslandChain`` として残ります。

関連項目
--------

- :doc:`/ja/mini-cases`
- :doc:`/notebooks/i18n/ja/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/i18n/ja/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/i18n/ja/tutorials/island_jacobian_analysis`
