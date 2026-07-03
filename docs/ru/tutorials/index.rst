Учебные материалы и примеры
===========================

Публичные notebooks сгруппированы по workflows. Сборка документации копирует
``notebooks/`` в дерево исходников Sphinx, поэтому пути ниже повторяют структуру
репозитория.

Рекомендуемая траектория обучения
---------------------------------

Начните с :doc:`/ru/quickstart`, затем пройдите workflow общей геометрии,
стохастические модели, а после этого тороидальные примеры монодромии/RMP:

1. :doc:`/ru/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

Общие динамические системы
--------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

Workflow общей геометрии и workflow неподвижных точек аналитического stellarator
теперь включены в учебник по RMP-резонансу, а не публикуются как отдельные
текстовые notebooks. Этот учебник показывает ту же цепочку повышения:
sampled crossings -> fixed-point geometry -> X/O classification -> manifold and
coordinate-grid overlays.

Стохастические дифференциальные уравнения
-----------------------------------------

Учебник SDE предварительно выполняется локально, потому что оценка распределений
часто использует десятки или сотни тысяч путей Монте-Карло. GitHub Pages
рендерит сохраненные выводы вместо траты CI-времени на тяжелые ячейки
sampling.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

Магнитные координаты и равновесия
---------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMP, острова и анализ Пуанкаре
------------------------------

При изучении магнитной топологии начните с notebook анализа резонансов. Теперь
он покрывает бездивергентные RMP-шаблоны, важную ветвь ``m=1``, проверку
неподвижных точек ``cyna``, многокомпонентные атласы магнитного спектра
контравариантного ``B^r``, модульные карты резонансов ``q``/``m/n`` с
необязательными проекциями Пуанкаре и островными наложениями, смешанные спектры
RMP/nRMP, полный nRMP-ответ от всех нерезонансных мод, модуляцию скорости
силовых линий и проверки порядка возмущений.

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` хранится в репозитории как вариант execution/cache
workflow анализа резонансов, но публичная документация ссылается на
пояснительную версию выше.

Монодромия и многообразия
-------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

Классические и общие динамические системы
-----------------------------------------

Репозиторий также содержит легкие notebooks в ``notebooks/examples``:
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` и
``FPT_DX_to_DP_sympy.ipynb``. Они сохраняются как исходные примеры, а не как
выполненные страницы документации, потому что несколько из них являются
черновыми notebooks без заголовков разделов.

Статические иллюстрации учебников
---------------------------------

Некоторые более длинные workflows представлены в репозитории как статические
рисунки и сгенерированные выводы в ``notebooks/tutorials``. Они покрывают
диагностику q-profile, координаты PEST/Boozer/Hamada/equal-arc, сканирования
подавления островов, phase control, многообразия Пуанкаре и примеры Solov'ev с
single-null.
