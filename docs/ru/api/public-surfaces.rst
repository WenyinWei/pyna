Карта публичного API
====================

Эта страница является коротким маршрутом к стабильным интерфейсам pyna.
Сгенерированная AutoAPI остается полной справкой для отладки; ниже перечислены
входы, которые стоит использовать в ноутбуках, исследовательских скриптах и
внешних пакетах.

Геометрический словарь
----------------------

Начинайте здесь, когда важен объект в фазовом пространстве, а не конкретный
решатель, который его создал.

.. list-table::
   :header-rows: 1

   * - Задача
     - Публичные входы
   * - Сэмплированное непрерывное движение
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - Динамика дискретных отображений
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - Геометрия тороидального сечения
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - Явное продвижение объектов и адаптеры
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

Общие динамические системы
--------------------------

Используйте :mod:`pyna.dynamics` для нетороидальных моделей, которые должны
возвращать те же геометрические объекты.

.. list-table::
   :header-rows: 1

   * - Семейство моделей
     - Публичные входы
   * - ODE-потоки
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Гамильтоновы системы
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N-тельные системы
     - :class:`pyna.dynamics.NBodySystem`
   * - Дискретные отображения
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - SDE
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

Тороидальные и RMP workflow
---------------------------

Для магнитных координат, трассировки линий поля, спектрального анализа и
графических overlay используйте следующие модули.

.. list-table::
   :header-rows: 1

   * - Потребность
     - Публичные входы
   * - Равновесия и координаты
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - Трассировка линий и cache-aware workflow
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - Контравариантный радиальный спектр возмущения
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - Диагностика RMP/nRMP
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - Фигуры магнитного спектра
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Overlay Poincare, X/O и островов
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

Когда переходить к AutoAPI
--------------------------

Используйте :doc:`/en/api/generated/pyna/index`, если нужны сигнатуры
конструкторов, унаследованные члены, редкая диагностика или детали реализации.
Новые tutorial и пользовательские примеры по возможности должны оставаться на
публичных входах выше.
