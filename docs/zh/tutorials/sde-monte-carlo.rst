SDE Monte Carlo 分布估计
========================

这个案例展示 pyna 里随机微分方程的实际使用路径：

1. 用 ``BrownianMotion``、``GeometricBrownianMotion`` 或 ``ItoSDE`` 定义模型；
2. 生成一条可复现样本路径，它是 ``Trajectory``；
3. 对大量样本使用向量化 NumPy 做 Monte Carlo 分布估计；
4. 在有解析公式时，对比经验均值、方差和分位数。

使用 pyna 的 SDE 类来表达模型边界和单路径几何；大量 ensemble 在 pyna 拥有
专门的 ensemble 几何类之前，继续使用向量化 NumPy 数组。这样数学对象模型
更诚实：单条随机实现是采样轨道，大量实现是统计估计器。

.. note::

   下面的 notebook 已在本地执行并保存输出，``nbsphinx`` 不会在 GitHub
   workflow 里重新执行重采样单元。修改 Monte Carlo 参数后，请先本地重跑
   notebook，再提交更新后的输出。

可执行 notebook：

- :doc:`/notebooks/tutorials/sde_monte_carlo_distribution`

可复制模式
----------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution 是 pyna Trajectory

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

扩展要点
--------

- ``ItoSDE.diffusion_matrix`` 支持标量、向量或矩阵扩散项；
- ``ItoSDE.euler_maruyama`` 可以传入固定的 ``dW``，便于回归测试和共同随机数实验；
- 只有当几何声称有意义时，才把一条样本路径提升到拓扑对象中；Monte Carlo
  样本用于估计分布，不会自动成为不变集。
