Слой ускорения cyna
===================

``cyna`` - это C++-слой ускорения, поставляемый с pyna. Он используется там, где
Python hot loops неприемлемы: трассировка силовых линий, пакетные вычисления
Пуанкаре, сканирование неподвижных точек, длины соединения/удары о стенку, поля
катушек и ядра функциональной теории возмущений.

Контракт сборки
---------------

``pyna._cyna`` ожидает скомпилированный бинарный файл ``_cyna_ext`` в пакете.
Установки из исходников собирают его через xmake; wheels PyPI включают его. См.
:doc:`../installation` для настройки платформы и флагов CUDA.

Канонический порядок cylindric field-cache:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Используйте :func:`pyna._cyna.prepare_field_cache`, чтобы преобразовать
``pyna.fields.VectorFieldCylind`` или legacy dict в C-contiguous arrays.

Высокоуровневые и низкоуровневые API
------------------------------------

Для прикладного кода предпочитайте высокоуровневые wrappers:

- ``pyna.flt`` и ``pyna.toroidal.flt`` для трассировки
- ``pyna.topo`` для отображений Пуанкаре, циклов, островов, многообразий и FPT
  response
- ``pyna.toroidal.coils`` для построения полей катушек

Используйте ``pyna._cyna`` напрямую только на границах bridges, для диагностики
или при написании нового высокоуровневого wrapper.

Справка по Python wrapper
-------------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

Утилитарные helpers
-------------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
