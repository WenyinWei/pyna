力学系ワークフローと拡張ヘルパー
================================

``pyna.topo`` は、中心となるトポロジーオブジェクトの周囲にワークフローヘルパーを公開します。
主なユーザー向け入口は ``TopologyWorkflow`` です。低レベルのプロトコル、アダプター、
ビルダー、ブリッジ、ファクトリーモジュールは、安定した拡張点を必要とする下流ライブラリ向けに
引き続き利用できます。

ワークフローファサード
----------------------

``TopologyWorkflow`` は notebook と日常的なスクリプト向けに設計されています。システム
構築、積分/反復、明示的な持ち上げ、断面切断を、新しい数学的オブジェクト型を増やさずに
組み合わせます。

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

プロトコル
----------

構造的プロトコルは外部システムの拡張契約を記述します。第三者のオブジェクトは、必要な
属性とメソッドを実装すれば参加できます。pyna クラスをサブクラス化する必要はありません。

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

アダプター
----------

アダプターは配列、ソルバー出力、既存の pyna オブジェクトを中心的な幾何表現へ正規化します。
サンプルデータを不変オブジェクトへ暗黙に持ち上げることはありません。

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

ビルダー
--------

ビルダーは明示的な持ち上げ規則を符号化します。たとえば軌道は、閉じたサンプルを
要求できるビルダーまたはアダプター呼び出しを通してのみ ``Cycle`` へ持ち上げられます。

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

ブリッジ
--------

ブリッジは連続時間と離散時間のオブジェクト族を接続します。
``Cycle -> PeriodicOrbit``、``Tube/TubeChain -> IslandChain`` です。

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

ファクトリーとレジストリ
------------------------

ファクトリーはシステム、幾何、Poincare 写像の安定した構築入口を提供します。レジストリは明示的で
重複に強いため、テストや下流ライブラリは自分の拡張を隔離できます。

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
