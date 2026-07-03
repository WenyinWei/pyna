Torus Deformation 理論
=====================

``pyna.toroidal.torus_deformation`` には、制御された摂動に対して不変 torus と共鳴構造が
どのように応答するかを調べるための解析的 torus-deformation ツールが含まれます。

概念上の役割
------------

幾何階層では次のように扱います。

- 不変 torus は ``InvariantTorus`` です。
- 共鳴楕円 cycle は ``Tube`` の core です。
- 双曲 cycle は tube を境界づけ、安定/不安定多様体を生成します。
- tube を Poincare 断面で切ると ``IslandChain`` オブジェクトが得られます。

したがって torus-deformation 計算は、トポロジー制御へ直接つながります。どのスペクトル
摂動が共鳴構造を動かし、分裂させ、修復し、または抑制するかを予測します。

公開 API
--------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

関連モジュール
--------------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
