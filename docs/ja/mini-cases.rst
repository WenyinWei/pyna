ミニケース
==========

このページはクイックスタートと完全な API リファレンスの間にある短い経路です。扱う
システムの種類がすでに分かっており、最小の動く pyna パターンが欲しいときに使います。

どの入口を使うか
----------------

.. list-table::
   :header-rows: 1

   * - 手元にあるもの
     - ここから始める
     - 通常得られる幾何
   * - ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` または ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory``、必要に応じて ``Cycle``
   * - Hamiltonian ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` または ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - 有限次元写像 ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit``、必要に応じて ``PeriodicOrbit``
   * - トロイダル磁場
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``、``Tube``、``IslandChain``、多様体
   * - 確率過程の教育モデル
     - ``BrownianMotion`` または ``GeometricBrownianMotion``
     - サンプル ``Trajectory`` と統計量

ケース 1: ODE サンプルから閉じた Cycle へ
-----------------------------------------

``Trajectory`` はサンプルデータを意味します。``Cycle`` は、そのサンプルが閉じている
というより強い主張をしていることを意味します。

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow

   wf = TopologyWorkflow(closure_tol=2e-2)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(wf.closing_error(traj))
   cycle = wf.closed_cycle(traj)
   print(cycle.period_value, cycle.ambient_dim)

本番 workflow では閉合許容値を明示的に保ってください。数値上の仮定をレビューしやすく
なります。

ケース 2: 写像反復から周期 orbit へ
-----------------------------------

写像はまず ``Orbit`` オブジェクトを生成します。``PeriodicOrbit`` へ持ち上げるのは、
既知または数値的に検証済みの閉じたサンプルだけにしてください。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap
   from pyna.topo import TopologyWorkflow

   flip = CallableMap(lambda x: np.array([-x[0], -x[1]]), dim=2)
   wf = TopologyWorkflow(closure_tol=1e-12)

   orbit = wf.orbit(flip, [1.0, 0.0], n_iter=2)
   periodic = wf.periodic_orbit(
       orbit.states[:-1],
       map_obj=flip,
       coordinate_names=("x", "y"),
   )
   print(periodic.period, periodic.points[0].state)

写像が別パッケージ由来の場合は、``CallableMap`` で包むか、``__call__(x)`` と
``phase_space`` 属性を実装してください。

ケース 3: 解析 stellarator の O/X 点
------------------------------------

磁場閉じ込めの作業では、磁力線 flow は Poincare 断面で切られます。実行可能な
チュートリアル :doc:`/notebooks/i18n/ja/tutorials/RMP_resonance_analysis` には、完全な
可視化計算が含まれています。

1. 公開されている解析 stellarator モデルを構築する。
2. 無発散の ``m=1`` および ``m>1`` RMP テンプレートを検証する。
3. 非摂動および摂動後の Poincare 断面を追跡する。
4. 解析的な共鳴 X/O 位相を ``cyna`` Newton 固定点と比較する。
5. 反変 ``B^r`` pcolormesh atlas、任意の Poincare 投影付き ``q``/``m/n``
   共鳴図、インタラクティブな Plotly 3-D bar、半径方向の固定 ``n``/固定 ``m`` 図、
   共鳴曲線、切り替え可能な島幅 marker により、多成分 RMP スペクトルを解析する。
6. すべての非共鳴スペクトル行から総 nRMP 応答を計算する。
7. contribution table は ranking と収束の診断にのみ使う。
8. nRMP 磁束面変形と磁力線速度変調を可視化する。
9. 局所安定枝と PEST 形式の座標格子を重ねる。

固定点描画、断面幾何、RMP/nRMP 診断、チュートリアル描画の変更をテストするときは、
この notebook を使ってください。公開前にローカルで走らせられるほど小さく、それでも
下流解析スクリプトが使う公開 helper API を十分に実行します。

ケース 4: カスタムシステム登録
------------------------------

factory は任意です。下流プロジェクトが設定駆動である場合に重要になります。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableFlow
   from pyna.topo.factories import DynamicalSystemFactory

   def make_damped_oscillator(gamma=0.1):
       return CallableFlow(
           lambda x, t: np.array([x[1], -x[0] - gamma*x[1]]),
           dim=2,
           coordinate_names=("q", "p"),
           label="damped oscillator",
       )

   DynamicalSystemFactory.register(
       "damped-oscillator",
       lambda gamma=0.1: make_damped_oscillator(gamma),
       overwrite=True,
   )
   flow = DynamicalSystemFactory.create("damped-oscillator", gamma=0.05)

グローバル登録によってテスト順序に依存してしまう場合は、テストでローカルな
``Registry`` インスタンスを使ってください。

ケース 5: SDE 分布推定
----------------------

単一の SDE path は pyna trajectory です。Monte Carlo ensemble は統計推定器です。
pyna が専用の ensemble オブジェクトを持つまでは、配列のまま扱ってください。

.. code-block:: python

   import numpy as np
   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1)
   print(path.final)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=100_000)
   terminal = 100.0 * np.exp(gbm.expected_log_growth()[0] + gbm.sigma[0] * z)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

Brownian、Ornstein-Uhlenbeck、geometric Brownian motion の分布を扱う完全な実行済み
ケースは :doc:`/ja/tutorials/sde-monte-carlo` を参照してください。

ケース 6: どこをカスタマイズするか
----------------------------------

.. list-table::
   :header-rows: 1

   * - 目的
     - 拡張するもの
     - 注意点
   * - 新しい物理モデル
     - ``CallableFlow``、``HamiltonianSystem``、または ``ContinuousFlow`` のサブクラス
     - 積分メソッドから pyna 幾何を返す
   * - 新しい写像族
     - ``CallableMap`` または ``DiscreteMap`` のサブクラス
     - 安定した座標名を公開する
   * - 新しい断面
     - ``pyna.topo.section.Section`` 形式のオブジェクト
     - crossing/project の意味論を明確に実装する
   * - 新しいデータ形式
     - ``pyna.topo.adapters``
     - データを正規化し、周期性を暗黙に主張しない
   * - 新しい組み立て方針
     - ``pyna.topo.builders``
     - 検証と metadata を集中させる
   * - 新しいバックエンド選択
     - factory または workflow facade
     - 生のバックエンド配列を pyna オブジェクトの背後に保つ

経験則: 数学的オブジェクトには dataclass、入力正規化には adapter、検証には builder、
安定した文字列キーが必要な場合にのみ factory を使います。

Notebook チェックリスト
-----------------------

ドキュメント公開前:

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

保存済み出力を持つ重い notebook は、ローカルで実行して更新後の ``.ipynb`` ファイルを
commit してください。

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

GitHub Pages と同じ notebook セットを使う場合は、Sphinx build をローカルで実行します。

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
