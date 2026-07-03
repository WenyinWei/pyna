チュートリアルと例
==================

公開 notebook は workflow ごとにまとめられています。ドキュメントビルドでは
``notebooks/`` を Sphinx ソースツリーへコピーするため、以下のパスはリポジトリ構成を
反映しています。

推奨学習経路
------------

:doc:`/ja/quickstart` から始め、汎用幾何 workflow、確率モデル、トロイダル
Monodromy/RMP の例へ進んでください。

1. :doc:`/ja/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/ja/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/ja/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/ja/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/ja/tutorials/RMP_island_validation_solovev`

一般力学系
----------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

汎用幾何 workflow と解析的ステラレータ fixed-point workflow は、独立したテキストのみの
notebook として公開される代わりに、現在は RMP 共鳴 tutorial に統合されています。
その tutorial は同じ昇格の流れを示します: sampled crossings -> fixed-point
geometry -> X/O classification -> manifold and coordinate-grid overlays。

確率微分方程式
--------------

SDE tutorial はローカルで事前実行されています。分布推定では数万から数十万の Monte
Carlo path を使うことが多いためです。GitHub Pages は重いサンプリング cell に CI 時間を
費やす代わりに、保存済み出力を描画します。

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/ja/tutorials/sde_monte_carlo_distribution

磁気座標と平衡
--------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ja/tutorials/magnetic_coordinates_comparison

RMP、磁気島、Poincaré解析
--------------------------

磁気 topology を調べるときは共鳴解析 notebook から始めてください。現在は
無発散 RMP テンプレート、重要な ``m=1`` branch、``cyna`` 固定点検証、多成分の反変
``B^r`` の磁気スペクトル atlas、任意の Poincaré断面と磁気島 overlay を持つモジュール型
``q``/``m/n`` 共鳴図、混合 RMP/nRMP スペクトル、すべての非共鳴 mode からの総 nRMP
応答、磁力線速度変調、摂動次数チェックを扱います。

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ja/tutorials/RMP_resonance_analysis
   /notebooks/i18n/ja/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/ja/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` は共鳴解析 workflow の実行/cache 版として
リポジトリに残されていますが、公開ドキュメントは上の説明版へリンクします。

Monodromy と多様体
------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ja/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/ja/tutorials/monodromy_xcycle_analytic

古典および一般力学系
--------------------

リポジトリには ``notebooks/examples`` の下にも軽量 notebook があります。
``Lorenz_attractor.ipynb``、``resonance_1_1_map.ipynb``、
``Mobiusian_saddle_cycle.ipynb``、``Xcycle_construction.ipynb``、
``FPT_DX_to_DP_sympy.ipynb`` です。これらは、いくつかが節タイトルのない
scratch-style notebook であるため、実行済みドキュメントページではなくソース例として
保持されています。

静的チュートリアル図
--------------------

いくつかの長い workflow は、``notebooks/tutorials`` の下に静的図と生成済み出力として
表現されています。q-profile 診断、PEST/Boozer/Hamada/equal-arc 座標、磁気島抑制
scan、位相制御、Poincaré多様体、Solov'ev single-null 例を扱います。
