场线追踪 (``pyna.flt``)
=======================

``pyna.flt`` 提供磁场线追踪的公共入口。典型流程是定义 ``dR/dphi``、
``dZ/dphi`` 的右端，选择 CPU 或 cyna 后端，然后在 Poincare 截面上累积
交点。

.. code-block:: python

   from pyna.flt import FieldLineTracer, get_backend

   tracer = FieldLineTracer(field_rhs, backend=get_backend("cpu"))

生产计算建议把 field function、起点、截面和步长记录进 metadata，便于之后
把交点提升为 ``PeriodicOrbit``、``IslandChain`` 或流形对象时审查来源。

英文详细 API：

- :doc:`../../en/api/flt`
