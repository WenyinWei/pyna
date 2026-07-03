일반 동역학 (``pyna.dynamics``)
===============================

``pyna.dynamics`` 는 넓은 의미의 동역학계 계층입니다. 의도적으로 작게 유지되어
있으며 ``pyna.topo`` 와 함께 쓰기 쉽습니다.

- 표본 trajectory를 갖는 callable ODE flow
- canonical Hamiltonian system 및 separable Hamiltonian
- 쌍대 gravitational/electrostatic N-body system
- Jacobian, fixed-point residual, Lyapunov spectrum 추정을 포함한 유한 차원 map
- Ito SDE, Brownian motion, geometric Brownian motion

클래스들은 상태 우선 관례를 사용합니다. flow에는 ``rhs(x, t)``, map에는
``step(x)`` 를 씁니다.

기하 통합
---------

이 모듈은 토로이달 위상 계층에서 쓰는 것과 같은 geometry class를 반환합니다.

- ``TimeSeriesSolution`` 은 ``pyna.topo.core.Trajectory`` 입니다.
- ``CallableMap.orbit_geometry`` 는 ``pyna.topo.core.Orbit`` 을 반환합니다.
- ``CallableMap.periodic_orbit`` 는 ``pyna.topo.core.PeriodicOrbit`` 을 반환합니다.
- ``pyna.topo.CoreTube`` 와 ``pyna.topo.CoreIslandChain`` 은 일반 유한 차원
  root입니다. ``pyna.topo.Tube`` 는 이전 호환성을 위한 토로이달 specialization으로
  남아 있습니다.

이 덕분에 해밀토니안 계, N-body flow, map, SDE 표본 경로가 자기장 선 위상과
같은 ``Cycle``/``Tube``/``IslandChain`` 어휘를 공유할 수 있습니다.

교육용 notebook 또는 extension이 많은 workflow는 ``TopologyWorkflow`` 와
저수준 adapter, builder, bridge, factory helper를 설명하는
:doc:`dynamics-patterns` 를 보세요.

연속 흐름
---------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

해밀토니안 계
-------------

``H(q, p, t)`` 또는 그 gradient를 제공할 수 있으면 ``HamiltonianSystem`` 을
사용하세요. ``H(q, p) = T(p) + V(q)`` 형태와 velocity-Verlet step에는
``SeparableHamiltonianSystem`` 을 사용합니다.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import SeparableHamiltonianSystem

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   x1 = oscillator.step_velocity_verlet(np.array([1.0, 0.0]), dt=0.01)

.. automodule:: pyna.dynamics
   :no-index:
   :members: HamiltonianSystem, SeparableHamiltonianSystem
   :show-inheritance:

N-body 계
---------

``NBodySystem`` 은 flatten된 state vector를
``[positions.ravel(), velocities.ravel()]`` 로 저장하고, structured array를
pack/unpack하는 helper를 제공합니다. Newtonian gravity와 electrostatic
Coulomb interaction을 지원합니다.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import NBodySystem

   system = NBodySystem([1.0, 1.0], spatial_dim=2, interaction="gravity")
   y0 = system.pack_state(
       positions=np.array([[-1.0, 0.0], [1.0, 0.0]]),
       velocities=np.zeros((2, 2)),
   )
   dy = system.vector_field(y0)

.. automodule:: pyna.dynamics
   :no-index:
   :members: NBodySystem
   :show-inheritance:

맵과 국소 다양체
----------------

``CallableMap`` 은 임의의 유한 차원 map을 다룹니다. ``fixed_point_eigenspaces``
는 고정점의 stable, unstable, center eigenspace를 분류하며 local manifold
구성으로 이어지는 유용한 bridge입니다.

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

확률 미분방정식
--------------

SDE 계층은 Ito 형식 ``dX = a(X,t) dt + B(X,t) dW`` 를 사용하며, 재현 가능한
연구와 교육 예제를 위해 deterministic Euler-Maruyama 구현을 제공합니다.
분포 추정 workflow는 :doc:`/ko/tutorials/sde-monte-carlo` 를 보세요.

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

관련 위상 계층
--------------

topology package는 추상 수학 계층과 푸앵카레 장치를 보관합니다.

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
