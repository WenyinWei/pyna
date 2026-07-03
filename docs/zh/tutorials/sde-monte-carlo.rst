SDE 蒙特卡洛分布
====================

本教程展示 pyna 中实用的 SDE 工作流：

1. 使用 ``BrownianMotion``、``GeometricBrownianMotion`` 或 ``ItoSDE`` 定义 Itô 模型；
2. 生成可复现的样本路径，并表示为 ``Trajectory``；
3. 运行向量化蒙特卡洛样本集合，用于分布估计；
4. 在可用时，将经验均值、方差和分位数与解析公式比较。

使用 pyna 的 SDE 类来界定模型边界并表示单路径几何。大规模样本集合请使用向量化
NumPy 数组，直到 pyna 增加专用样本集合几何类。这样能让数学对象模型保持清晰：
单次实现是一条样本轨迹，样本云才是统计估计器。

.. note::

   下面的可执行 notebook 已提交保存输出，并禁用了 ``nbsphinx`` 执行。修改数值参数时
   请在本地重新运行；文档工作流会在 GitHub Pages 上渲染这些保存输出。

可执行 notebook：

- :doc:`/notebooks/i18n/zh/tutorials/sde_monte_carlo_distribution`

可复用模式
----------

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

扩展说明
--------

- ``ItoSDE.diffusion_matrix`` 接受标量、向量或矩阵扩散项。
- ``ItoSDE.euler_maruyama`` 接受外部提供的 ``dW`` 增量，因此共同随机数
  实验和回归测试可以保持确定性。
- 只有当几何声明有意义时，才把单条样本路径提升为拓扑对象。蒙特卡洛样本估计
  分布；它们不会自动成为不变集。
