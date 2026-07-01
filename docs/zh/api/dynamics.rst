通用动力系统 (``pyna.dynamics``)
=================================

``pyna.dynamics`` 是非环形场线问题的主要入口，覆盖：

- ``CallableFlow``：任意有限维 ODE；
- ``HamiltonianSystem`` / ``SeparableHamiltonianSystem``：正则 Hamiltonian 系统；
- ``NBodySystem``：引力或静电相互作用；
- ``CallableMap``：有限维离散映射、Jacobian、Lyapunov 指数；
- ``ItoSDE``、``BrownianMotion``、``GeometricBrownianMotion``：Ito SDE。

这些对象输出同一套拓扑几何对象：连续时间得到 ``Trajectory``，离散映射得到
``Orbit``，闭合结构再显式提升为 ``Cycle`` 或 ``PeriodicOrbit``。

.. code-block:: python

   from pyna.dynamics import BrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=123)
   print(path.final)

英文详细 API：

- :doc:`../../en/api/dynamics`
- :doc:`../../en/api/generated/pyna/dynamics/index`
