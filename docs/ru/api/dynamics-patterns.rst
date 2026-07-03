Workflows динамики и helpers для расширения
===========================================

``pyna.topo`` предоставляет workflow helpers вокруг основных топологических
объектов. Главная пользовательская точка входа - ``TopologyWorkflow``;
низкоуровневые модули Protocol, Adapter, Builder, Bridge и Factory остаются
доступными для downstream-библиотек, которым нужны стабильные точки расширения.

Workflow facade
---------------

``TopologyWorkflow`` спроектирован для notebooks и повседневных скриптов. Он
объединяет построение системы, интегрирование/итерацию, явное повышение и
сечения без добавления нового математического типа объекта.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Протоколы
---------

Структурные протоколы описывают контракты расширения для внешних систем.
Сторонние объекты могут участвовать, реализуя требуемые атрибуты и методы;
наследование от классов pyna необязательно.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Адаптеры
--------

Адаптеры нормализуют массивы, выводы решателей и существующие объекты pyna в
основные геометрические представления. Они не повышают выборочные данные до
инвариантных объектов молча.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builders
--------

Builders кодируют явные правила повышения. Например, траектория может быть
повышена до ``Cycle`` только через вызов builder или adapter, который может
требовать замкнутые выборки.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridges
-------

Bridges соединяют семейства объектов непрерывного и дискретного времени:
``Cycle -> PeriodicOrbit`` и ``Tube/TubeChain -> IslandChain``.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factories и registries
----------------------

Factories предоставляют стабильные точки входа для построения систем, геометрии
и отображений Пуанкаре. Registries явные и защищенные от дубликатов, чтобы
тесты и downstream-библиотеки могли изолировать собственные расширения.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
