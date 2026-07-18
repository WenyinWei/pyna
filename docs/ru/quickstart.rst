Быстрый старт
=============

Эта страница проходит через три ключевые возможности **pyna** - трассировку
силовых линий, отображения Пуанкаре и топологию островов - на простом
аналитическом равновесии токамака, не требующем внешних файлов данных.

.. note::

   Все примеры используют **аналитическое равновесие Solov'ev** (Cerfon &
   Freidberg 2010), масштабированное к параметрам типа EAST (R₀ ≈ 1.86 m,
   B₀ = 5.3 T). Это хороший универсальный тестовый стенд: точное решение
   Grad-Shafranov, компоненты поля в замкнутой форме, настраиваемая форма.

----

1. Построить аналитическое равновесие
-------------------------------------

Начните с импорта равновесия и просмотра его базовых параметров:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"κ  = {eq.kappa:.2f}  δ = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

Возвращаемый объект ``eq`` предоставляет ``eq.BR_BZ(R, Z)``, ``eq.Bphi(R)``,
``eq.psi(R, Z)`` (нормированный поток) и ``eq.q_profile(psi)``.

----

2. Трассировать силовые линии и накопить пересечения Пуанкаре
------------------------------------------------------------

Сечение Пуанкаре записывает координаты (R, Z) каждый раз, когда силовая линия
пересекает выбранное тороидальное сечение (здесь φ = 0). После многих
тороидальных оборотов вложенные магнитные поверхности выглядят как замкнутые
кривые; магнитный остров проявляется как цепочка дискретных точек сечения.

.. code-block:: python

   from pyna.flt import get_backend
   from pyna.topo.poincare import PoincareToroidalSection, poincare_from_fieldlines

   # The accumulator section detects crossings between sampled 3-D points.
   section = PoincareToroidalSection(0.0)

   # --- unit tangent in cylindrical coordinates: dR/dl, dZ/dl, dφ/dl ---
   def field_rhs(rzphi):
       R, Z, _phi = rzphi
       BR, BZ = eq.BR_BZ(R, Z)
       Bphi   = eq.Bphi(R)
       Bnorm  = np.sqrt(BR**2 + BZ**2 + Bphi**2)
       return [BR / Bnorm, BZ / Bnorm, Bphi / (R * Bnorm)]

   # --- seed 8 field lines radially outward from the axis ---
   R_starts = np.linspace(Rmaxis + 0.05, Rmaxis + 0.45, 8)
   Z_starts = np.zeros(8)

   # --- integrate about 80 toroidal turns per line ---
   n_turns = 80
   flt = get_backend('cpu', field_func=field_rhs, dt=0.08)
   pacc = poincare_from_fieldlines(
       field_func=field_rhs,
       start_pts=np.column_stack([R_starts, Z_starts, np.zeros_like(R_starts)]),
       sections=[section],
       t_max=n_turns * 2 * np.pi * Rmaxis,
       backend=flt,
   )
   poincare_pts = pacc.crossing_array(0)[:, :2]

   # --- plot ---
   fig, ax = plt.subplots(figsize=(6, 6))
   ax.scatter(poincare_pts[:, 0], poincare_pts[:, 1], s=0.8, color='steelblue')
   ax.set_xlabel('R (m)')
   ax.set_ylabel('Z (m)')
   ax.set_aspect('equal')
   ax.set_title('Poincaré map — Solov\'ev equilibrium')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/quickstart_poincare.png
   :align: center
   :width: 80%
   :alt: Отображение Пуанкаре аналитического равновесия Solov'ev с вложенными магнитными поверхностями

   **Рисунок 1.** Отображение Пуанкаре аналитического равновесия Solov'ev
   (параметры типа EAST, 250 тороидальных проходов на силовую линию). Каждый
   цвет соответствует одной силовой линии; вложенные замкнутые кривые являются
   магнитными поверхностями. Красный крест отмечает магнитную ось; черная
   кривая - последняя замкнутая магнитная поверхность (LCFS, ψ = 1).

Каждое концентрическое кольцо соответствует одной силовой линии, наматывающейся
на магнитную поверхность. Рациональная поверхность q = m/n - это место, где
резонансное возмущение (например, RMP-катушка) может открыть магнитный остров.

----

3. Найти рациональную поверхность и измерить остров
---------------------------------------------------

После добавления малого резонансного возмущения на поверхности q = 2/1
открывается магнитный остров. pyna может найти поверхность и измерить
полуширину острова одним вызовом:

.. code-block:: python

   from pyna.topo.toroidal_island import locate_rational_surface, island_halfwidth

   # Build q(S) from PEST mesh
   from pyna.toroidal.coords import build_PEST_mesh

   nR, nZ = 100, 100
   R_grid = np.linspace(0.3*eq.R0, 1.5*eq.R0, nR)
   Z_grid = np.linspace(-eq.a*eq.kappa*1.3, eq.a*eq.kappa*1.3, nZ)
   Rg, Zg  = np.meshgrid(R_grid, Z_grid, indexing='ij')

   BR, BZ   = eq.BR_BZ(Rg, Zg)
   Bphi     = eq.Bphi(Rg)
   psi_norm = eq.psi(Rg, Zg)

   S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
       R_grid, Z_grid, BR, BZ, Bphi, psi_norm,
       Rmaxis, Zmaxis, ns=40, ntheta=181
   )
   S_values = S[1:]
   q_values = q_iS[1:]
   print(f"q range: {q_values[0]:.2f} → {q_values[-1]:.2f}")

   # Locate q = 2/1 surface
   res = locate_rational_surface(S_values, q_values, m=2, n=1)
   print(f"q=2/1 surface at S = {res[0]:.4f}  (ψ_norm = {res[0]**2:.4f})")

Возвращаемое значение ``S_res`` (S = √ψ_norm) точно указывает положение
резонансного слоя. Передайте его в ``island_halfwidth`` вместе с возмущенным
отображением Пуанкаре, чтобы получить ширину острова в метрах.

----

4. Общая конечномерная динамика
-------------------------------

pyna не ограничена тороидальными силовыми линиями. Та же объектная модель
топологии доступна для гамильтоновых систем, потоков N-тел, отображений и
выборочных траекторий SDE.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import (
       SeparableHamiltonianSystem,
       CallableMap,
       GeometricBrownianMotion,
   )

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   traj = oscillator.trajectory([1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(traj.final)  # TimeSeriesSolution is a pyna.topo.core.Trajectory

   linear_map = CallableMap(lambda x: np.array([2*x[0], 0.5*x[1]]), dim=2)
   orbit = linear_map.orbit_geometry([1.0, 1.0], n_iter=5)
   print(orbit.period_guess)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
   print(gbm.expected_log_growth())

Используйте объекты :mod:`pyna.topo.core`, такие как ``Cycle``,
``PeriodicOrbit``, ``Tube`` и ``IslandChain``, когда траектория или орбита
отображения повышена из выборочных данных до геометрического/топологического
объекта.

----

5. Построение на основе workflows
---------------------------------

Для крупных проектов и учебных notebooks используйте ``TopologyWorkflow``,
чтобы последовательность анализа оставалась явной и не распадалась на
разрозненные ad-hoc конструкторы в коде.

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow
   from pyna.topo.section import HyperplaneSection

   wf = TopologyWorkflow(closure_tol=1e-3)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
   pmap = wf.poincare_map(flow, section, dt=0.02)

   closed_traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   cycle = wf.closed_cycle(closed_traj)

Низкоуровневые adapters, builders, bridges и factories остаются доступными для
авторов библиотек, но большинству notebooks следует начинать с workflow facade.

----

6. Дальнейшие шаги
------------------

- **Учебные материалы** - проработанные примеры с графиками:
  :doc:`/ru/mini-cases`,
  :doc:`/ru/tutorials/sde-monte-carlo`,
  :doc:`/notebooks/i18n/ru/tutorials/RMP_resonance_analysis`,
  :doc:`/notebooks/i18n/ru/tutorials/magnetic_coordinates_comparison`,
  :doc:`/notebooks/i18n/ru/tutorials/RMP_island_validation_solovev`

- **Справочник API** - полные docstrings:
  :doc:`/ru/api/index`

- **Ускорение CUDA** - установите ``cupy-cuda12x`` и передайте
  ``backend=get_backend('cuda')`` в трассировщик для ускорения до 100x на
  сканированиях ширины островов.
