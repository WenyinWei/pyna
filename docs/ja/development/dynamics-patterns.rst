力学系 Workflow と拡張 Helper
=============================

pyna は数学的幾何と構成方針を分離します。

核心階層は小さく保たれます。

- 連続時間幾何: ``Trajectory``、``Cycle``、``Tube``、``TubeChain``。
- 離散時間幾何: ``Orbit``、``PeriodicOrbit``、``Island``、``IslandChain``。
- トロイダルクラスは、``pyna.topo.Tube``、``pyna.topo.Cycle``、
  ``pyna.topo.IslandChain`` の下で既定の公開 topology 特殊化のままです。

helper レイヤーは、この階層の周囲に 1 つのユーザー向け workflow facade と明示的な
拡張点を追加します。

Workflow Facade
---------------

``TopologyWorkflow`` はチュートリアルと解析スクリプトの最初の入口として推奨されます。
ユーザーが実際にたどる経路へ、低レベル helper を合成します。

1. flow/map を構築するか受け取る。
2. ``Trajectory`` を積分するか、``Orbit`` を反復する。
3. 閉じたサンプルを明示的に ``Cycle`` または ``PeriodicOrbit`` へ持ち上げる。
4. ``Cycle``/``Tube``/``TubeChain`` オブジェクトを断面で切る。

この facade は意図的に薄く作られています。新しい数学を導入せず、notebook コードを
読みやすく保ちながら、各持ち上げを明示的にします。

実例チュートリアル
------------------

コンパクトな workflow の概観は :doc:`/ja/mini-cases` から始めてください。同じ持ち上げの
考え方を実際のトロイダル計算へ適用する完全な可視化チュートリアルには
:doc:`/notebooks/tutorials/RMP_resonance_analysis` を使います。サンプルされた Poincare
交差、明示的な X/O 固定点幾何、座標格子 overlay、局所多様体枝が示されています。

短いコピー&ペースト用レシピには :doc:`/ja/mini-cases` を使います。このページは
クイックスタートと完全な API リファレンスの間をつなぐものです。

Protocol
--------

``pyna.topo.protocols`` は ``FlowLike``、``MapLike``、``SectionLike``、``TubeLike`` などの
構造的契約を定義します。すべての基底クラスを直接継承せずに pyna と相互運用したい
新しい領域パッケージを追加するときに使います。

Adapter
-------

``pyna.topo.adapters`` はユーザーデータを安定した core オブジェクトへ変換します。

- 配列または solver 出力から ``Trajectory`` と ``Orbit`` へ。
- 点または固定点に似たオブジェクトから ``SectionPoint`` へ。
- 要求されたとき、検証済みサンプルから ``PeriodicOrbit`` または ``Cycle`` へ。

adapter は表現を正規化します。数学的主張を隠すべきではありません。たとえば、開いた
サンプル trajectory は、呼び出し側が明示的に ``Cycle`` を求め、閉合チェックを受け入れる
か渡さない限り、``Trajectory`` のままです。

Builder
-------

``GeometryBuilder``、``IslandChainBuilder``、``TubeChainBuilder`` は構成方針を捕捉します。
workflow が複数の低レベル材料から topology を組み立てる場合、builder を優先して
ください。検証、metadata、back-link を集中できます。

Bridge
------

``CoreSectionCutBridge`` は core オブジェクトの既定の連続から離散への bridge です。

- ``Cycle.section_cut(section)`` は ``PeriodicOrbit`` を返します。
- ``Tube.section_cut(section)`` は ``IslandChain`` を返します。
- ``TubeChain.section_cut(section)`` は得られた island をマージします。

トロイダルオブジェクトは、最適化された ``section_cut`` メソッドをすでに持っています。
それらを直接使うか、``TopologyWorkflow.section_cut(...)`` を呼び、オブジェクト自身に
実装を dispatch させてください。

Factory
-------

``DynamicalSystemFactory`` は、``callable-flow``、``callable-map``、``hamiltonian``、
``nbody``、``geometric-brownian-motion`` などの安定した文字列キーから、すぐに使える
システムを構築します。

``PoincareMapFactory`` は実行可能な return-map 実装を選びます。既定の
``backend="auto"`` は、cyna field-cache 引数が渡されない限り、現在は可搬な
``GeneralPoincareMap`` を選択します。

``GeometryFactory`` は builder レイヤーを通じて topology 幾何を構築します。設定駆動の
例や、安定した構成キーを必要とする下流パッケージで有用です。

互換性ルール
------------

- ``pyna.topo.Tube``、``Cycle``、``IslandChain`` が core クラスを指すよう変更しない。
  汎用ルートには ``CoreTube``、``CoreCycle``、``CoreIslandChain`` を使う。
- トロイダル専用境界では duck-typed な偽 section を使わない。一級の ``Section``
  オブジェクトを使う。
- registry は可変状態として扱う。テストや下流パッケージで隔離が必要な場合は、ローカルな
  ``Registry`` インスタンスを使う。
