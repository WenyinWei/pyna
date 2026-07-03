Трассировка силовых линий (``pyna.flt``)
========================================

Пакет ``pyna.flt`` предоставляет процедуры интегрирования силовых линий,
построенные поверх абстрактной иерархии :mod:`pyna.system`.

**Backends:**

- **CPU/serial** -- чистый Python-интегратор RK4
- **CPU/parallel** -- многопроцессные или многопоточные варианты
- **CUDA** -- необязательный GPU backend через CuPy (ускорение до 118x)
- **OpenCL** -- экспериментально

.. contents:: Подмодули
   :depth: 2
   :local:

----

Основной трассировщик
---------------------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Иерархия динамических систем
----------------------------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
