Мини-кейсы
==========

Эта страница - короткий путь между быстрым стартом и полным справочником API.
Используйте ее, когда вы уже знаете тип своей системы и хотите минимальный
рабочий шаблон pyna.

Какой входной пункт выбрать?
----------------------------

.. list-table::
   :header-rows: 1

   * - Что у вас есть
     - С чего начать
     - Какую геометрию обычно получают
   * - ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` или ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory``, затем возможно ``Cycle``
   * - Гамильтониан ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` или ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - Конечномерное отображение ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit``, затем возможно ``PeriodicOrbit``
   * - Тороидальное магнитное поле
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``, ``Tube``, ``IslandChain``, многообразия
   * - Стохастическая учебная модель
     - ``BrownianMotion`` или ``GeometricBrownianMotion``
     - выборочная ``Trajectory`` плюс статистика

Кейс 1: выборка ODE в замкнутый цикл
------------------------------------

``Trajectory`` означает выборочные данные. ``Cycle`` означает более сильное
утверждение: выборка является замкнутой.

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

В production workflow держите допуск замыкания явным. Это делает численные
предположения проверяемыми.

Кейс 2: итерации отображения в периодическую орбиту
---------------------------------------------------

Отображения сначала создают объекты ``Orbit``. Повышайте до ``PeriodicOrbit``
только известные или численно проверенные замкнутые выборки.

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

Если ваше отображение приходит из другого пакета, оберните его в
``CallableMap`` или реализуйте ``__call__(x)`` вместе с атрибутом
``phase_space``.

Кейс 3: аналитические O/X-точки stellarator
-------------------------------------------

Для задач магнитного удержания поток силовых линий пересекается сечением
Пуанкаре. Исполняемый учебник
:doc:`/notebooks/i18n/ru/tutorials/RMP_resonance_analysis` теперь содержит полный
визуальный расчет:

1. построить публичную аналитическую модель stellarator;
2. проверить бездивергентные RMP-шаблоны ``m=1`` и ``m>1``;
3. трассировать невозмущенные и возмущенные сечения Пуанкаре;
4. сравнить аналитические резонансные X/O-фазы с Newton fixed points из
   ``cyna``;
5. анализировать многокомпонентные RMP-спектры с pcolormesh-атласами
   контравариантного ``B^r``, картами резонансов ``q``/``m/n`` с
   необязательными проекциями Пуанкаре, интерактивными 3-D столбцами Plotly,
   радиальными картами fixed-``n``/fixed-``m``, кривыми резонансов и
   переключаемыми маркерами ширины островов;
6. вычислить полный nRMP-ответ от всех нерезонансных строк спектра;
7. использовать таблицы вкладов только как диагностику ранжирования и
   сходимости;
8. визуализировать nRMP-деформацию магнитных поверхностей и модуляцию скорости
   силовых линий;
9. наложить локальные устойчивые ветви и координатную сетку в стиле PEST.

Используйте этот notebook при тестировании изменений в построении неподвижных
точек, геометрии сечений, диагностике RMP/nRMP или рендеринге учебников. Он
достаточно мал для локального запуска перед публикацией документации, но все же
проверяет публичные вспомогательные API, используемые downstream-скриптами
анализа.

Кейс 4: регистрация пользовательской системы
--------------------------------------------

Factories необязательны. Они важны, когда downstream-проект управляется
конфигурацией.

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

Используйте локальные экземпляры ``Registry`` в тестах, если глобальная
регистрация сделала бы порядок тестов значимым.

Кейс 5: оценка распределения SDE
--------------------------------

Одиночные SDE-пути являются траекториями pyna. Ансамбли Монте-Карло являются
статистическими оценивателями; храните их как массивы, пока в pyna не появится
выделенный объект ансамбля.

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

Полный выполненный пример с распределениями Brownian, Ornstein-Uhlenbeck и
geometric Brownian motion см. в :doc:`/ru/tutorials/sde-monte-carlo`.

Кейс 6: где расширять
---------------------

.. list-table::
   :header-rows: 1

   * - Цель
     - Что расширять
     - Что учитывать
   * - Новая физическая модель
     - ``CallableFlow``, ``HamiltonianSystem`` или подкласс ``ContinuousFlow``
     - возвращать геометрию pyna из методов интегрирования
   * - Новое семейство отображений
     - ``CallableMap`` или подкласс ``DiscreteMap``
     - предоставлять стабильные имена координат
   * - Новое сечение
     - объект в стиле ``pyna.topo.section.Section``
     - явно реализовать семантику crossing/project
   * - Новый формат данных
     - ``pyna.topo.adapters``
     - нормализовать данные; не заявлять периодичность молча
   * - Новая политика сборки
     - ``pyna.topo.builders``
     - централизовать validation и metadata
   * - Новый выбор backend
     - factories или workflow facade
     - держать сырые backend arrays за объектами pyna

Практическое правило: используйте dataclasses для математических объектов,
adapters для нормализации входа, builders для validation и factories только
тогда, когда пользователям нужны стабильные строковые ключи.

Checklist для notebooks
-----------------------

Перед публикацией документации:

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

Для тяжелых notebooks с сохраненными выводами запускайте их локально и коммитьте
обновленный файл ``.ipynb``:

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

Для того же набора notebooks, который использует GitHub Pages, соберите Sphinx
локально:

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
