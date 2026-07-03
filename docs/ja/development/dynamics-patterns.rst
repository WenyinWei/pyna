力学系ワークフローと拡張ヘルパー
================================

pyna は数学的幾何と構成方針を分離します。

核心階層は小さく保たれます。

- 連続時間幾何: ``Trajectory``、``Cycle``、``Tube``、``TubeChain``。
- 離散時間幾何: ``Orbit``、``PeriodicOrbit``、``Island``、``IslandChain``。
- トロイダルクラスは、``pyna.topo.Tube``、``pyna.topo.Cycle``、
  ``pyna.topo.IslandChain`` の下で既定の公開トポロジー特殊化のままです。

ヘルパーレイヤーは、この階層の周囲に 1 つのユーザー向けワークフローファサードと明示的な
拡張点を追加します。

ワークフローファサード
----------------------

``TopologyWorkflow`` はチュートリアルと解析スクリプトの最初の入口として推奨されます。
ユーザーが実際にたどる経路へ、低レベルヘルパーを合成します。

1. フロー/写像を構築するか受け取る。
2. ``Trajectory`` を積分するか、``Orbit`` を反復する。
3. 閉じたサンプルを明示的に ``Cycle`` または ``PeriodicOrbit`` へ持ち上げる。
4. ``Cycle``/``Tube``/``TubeChain`` オブジェクトを断面で切る。

このファサードは意図的に薄く作られています。新しい数学を導入せず、notebook コードを
読みやすく保ちながら、各持ち上げを明示的にします。

実例チュートリアル
------------------

コンパクトなワークフローの概観は :doc:`/ja/mini-cases` から始めてください。同じ持ち上げの
考え方を実際のトロイダル計算へ適用する完全な可視化チュートリアルには
:doc:`/notebooks/i18n/ja/tutorials/RMP_resonance_analysis` を使います。サンプルされた Poincare
交差、明示的な X/O 固定点幾何、座標格子の重ね描き、局所多様体枝が示されています。

短いコピー&ペースト用レシピには :doc:`/ja/mini-cases` を使います。このページは
クイックスタートと完全な API リファレンスの間をつなぐものです。

プロトコル
----------

``pyna.topo.protocols`` は ``FlowLike``、``MapLike``、``SectionLike``、``TubeLike`` などの
構造的契約を定義します。すべての基底クラスを直接継承せずに pyna と相互運用したい
新しい領域パッケージを追加するときに使います。

アダプター
----------

``pyna.topo.adapters`` はユーザーデータを安定した中心オブジェクトへ変換します。

- 配列またはソルバー出力から ``Trajectory`` と ``Orbit`` へ。
- 点または固定点に似たオブジェクトから ``SectionPoint`` へ。
- 要求されたとき、検証済みサンプルから ``PeriodicOrbit`` または ``Cycle`` へ。

アダプターは表現を正規化します。数学的主張を隠すべきではありません。たとえば、開いた
サンプル軌道は、呼び出し側が明示的に ``Cycle`` を求め、閉合チェックを受け入れる
か渡さない限り、``Trajectory`` のままです。

ビルダー
--------

``GeometryBuilder``、``IslandChainBuilder``、``TubeChainBuilder`` は構成方針を捕捉します。
ワークフローが複数の低レベル材料からトポロジーを組み立てる場合、ビルダーを優先して
ください。検証、メタデータ、逆参照を集中できます。

ブリッジ
--------

``CoreSectionCutBridge`` は中心オブジェクトの既定の連続から離散へのブリッジです。

- ``Cycle.section_cut(section)`` は ``PeriodicOrbit`` を返します。
- ``Tube.section_cut(section)`` は ``IslandChain`` を返します。
- ``TubeChain.section_cut(section)`` は得られた島をマージします。

トロイダルオブジェクトは、最適化された ``section_cut`` メソッドをすでに持っています。
それらを直接使うか、``TopologyWorkflow.section_cut(...)`` を呼び、オブジェクト自身に
実装をディスパッチさせてください。

Factory
-------

``DynamicalSystemFactory`` は、``callable-flow``、``callable-map``、``hamiltonian``、
``nbody``、``geometric-brownian-motion`` などの安定した文字列キーから、すぐに使える
システムを構築します。

``PoincareMapFactory`` は実行可能な戻り写像実装を選びます。既定の
``backend="auto"`` は、cyna のフィールドキャッシュ引数が渡されない限り、現在は可搬な
``GeneralPoincareMap`` を選択します。

``GeometryFactory`` はビルダーレイヤーを通じてトポロジー幾何を構築します。設定駆動の
例や、安定した構成キーを必要とする下流パッケージで有用です。

互換性ルール
------------

- ``pyna.topo.Tube``、``Cycle``、``IslandChain`` が中心クラスを指すよう変更しない。
  汎用ルートには ``CoreTube``、``CoreCycle``、``CoreIslandChain`` を使う。
- トロイダル専用境界ではダックタイピングされた偽の断面を使わない。一級の ``Section``
  オブジェクトを使う。
- レジストリは可変状態として扱う。テストや下流パッケージで隔離が必要な場合は、ローカルな
  ``Registry`` インスタンスを使う。
