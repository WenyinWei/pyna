SDEのモンテカルロ分布
======================

この tutorial は、pyna での実用的な SDE workflow を示します。

1. ``BrownianMotion``、``GeometricBrownianMotion``、または ``ItoSDE`` で Ito モデルを
   定義する。
2. 再現可能なサンプル path を ``Trajectory`` として生成する。
3. 分布推定のため、ベクトル化した Monte Carlo ensemble を実行する。
4. 利用可能な場合は、経験平均、分散、分位点を解析式と比較する。

モデル境界と単一 path 幾何には pyna の SDE クラスを使ってください。大規模 ensemble には、
pyna が専用の ensemble 幾何クラスを持つまで、ベクトル化 NumPy 配列を使います。これにより、
数学的 object model と統計 estimator を混同せずに済みます。単一実現はサンプル trajectory
であり、実現の集合は統計推定器です。

.. note::

   下の実行可能 notebook は保存済み出力付きで commit されており、``nbsphinx`` 実行は
   無効化されています。数値パラメータを変更するときはローカルで再実行してください。
   docs workflow は GitHub Pages で保存済み出力を描画します。

実行可能な notebook:

- :doc:`/notebooks/i18n/ja/tutorials/sde_monte_carlo_distribution`

コピー&ペースト用パターン
--------------------------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution is a pyna Trajectory

   n_paths = 200_000
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=n_paths)
   log_terminal = (
       np.log(100.0)
       + gbm.expected_log_growth()[0] * 1.0
       + gbm.sigma[0] * np.sqrt(1.0) * z
   )
   terminal = np.exp(log_terminal)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

拡張メモ
--------

- ``ItoSDE.diffusion_matrix`` は scalar、vector、matrix diffusion を受け付けます。
- ``ItoSDE.euler_maruyama`` は外部から与えた ``dW`` increment を受け付けるため、
  common-random-number 実験と回帰テストを決定論的にできます。
- 幾何的主張が意味を持つ場合にのみ、1 本のサンプル path を topology オブジェクトへ
  持ち上げてください。Monte Carlo サンプルは分布を推定しますが、自動的に不変集合に
  なるわけではありません。
