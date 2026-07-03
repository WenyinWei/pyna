Общая динамика (``pyna.dynamics``)
==================================

``pyna.dynamics`` - это широкий слой динамических систем. Он намеренно мал и
интероперабелен с ``pyna.topo``:

- вызываемые ODE-потоки с выборочными траекториями
- канонические гамильтоновы системы и separable Hamiltonians
- попарные гравитационные/электростатические системы N-тел
- конечномерные отображения с якобианами, невязками неподвижных точек и
  оценками спектра Ляпунова
- SDE Ито, Brownian motion и geometric Brownian motion

Классы используют соглашение state-first: ``rhs(x, t)`` для потоков и
``step(x)`` для отображений.

Интеграция с геометрией
-----------------------

Модуль возвращает те же классы геометрии, что используются тороидальной
топологией:

- ``TimeSeriesSolution`` является ``pyna.topo.core.Trajectory``.
- ``CallableMap.orbit_geometry`` возвращает ``pyna.topo.core.Orbit``.
- ``CallableMap.periodic_orbit`` возвращает ``pyna.topo.core.PeriodicOrbit``.
- ``pyna.topo.CoreTube`` и ``pyna.topo.CoreIslandChain`` являются общими
  конечномерными корнями; ``pyna.topo.Tube`` остается тороидальной
  специализацией для обратной совместимости.

Это позволяет гамильтоновым системам, потокам N-тел, отображениям и выборочным
путям SDE разделять тот же словарь ``Cycle``/``Tube``/``IslandChain``, что и
топология магнитных силовых линий.

Для учебных notebooks или workflows с большим количеством расширений см.
:doc:`dynamics-patterns` о ``TopologyWorkflow`` и низкоуровневых helpers
adapter, builder, bridge и factory.

Непрерывные потоки
------------------

.. automodule:: pyna.dynamics
   :no-index:
   :members: TimeSeriesSolution, CallableFlow, finite_difference_jacobian
   :show-inheritance:

Гамильтоновы системы
--------------------

Используйте ``HamiltonianSystem``, когда можете предоставить ``H(q, p, t)`` или
его градиент. Используйте ``SeparableHamiltonianSystem`` для
``H(q, p) = T(p) + V(q)`` и шага velocity-Verlet.

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

Системы N-тел
-------------

``NBodySystem`` хранит развернутые векторы состояния как
``[positions.ravel(), velocities.ravel()]`` и предоставляет helpers для упаковки
и распаковки структурированных массивов. Он поддерживает ньютоновскую
гравитацию и электростатические кулоновские взаимодействия.

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

Отображения и локальные многообразия
------------------------------------

``CallableMap`` обрабатывает произвольные конечномерные отображения.
``fixed_point_eigenspaces`` классифицирует устойчивые, неустойчивые и
центральные собственные подпространства неподвижной точки и является полезным
bridge к построению локальных многообразий.

.. automodule:: pyna.dynamics
   :no-index:
   :members: CallableMap, fixed_point_eigenspaces
   :show-inheritance:

Стохастические дифференциальные уравнения
-----------------------------------------

Слой SDE использует форму Ито ``dX = a(X,t) dt + B(X,t) dW`` и детерминированную
реализацию Euler-Maruyama для воспроизводимых исследований и учебных примеров.
Для workflows оценки распределений см. :doc:`/ru/tutorials/sde-monte-carlo`.

.. code-block:: python

   from pyna.dynamics import GeometricBrownianMotion

   stock = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   print(stock.expected_log_growth())

.. automodule:: pyna.dynamics
   :no-index:
   :members: ItoSDE, BrownianMotion, GeometricBrownianMotion
   :show-inheritance:

Связанный топологический слой
-----------------------------

Пакет topology хранит абстрактную математическую иерархию и механику Пуанкаре:

.. automodule:: pyna.topo.dynamics
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.topo.classical_maps
   :no-index:
   :members:
   :show-inheritance:
