场线追踪（``pyna.flt``）
========================

``pyna.flt`` 包提供建立在抽象 :mod:`pyna.system` 层次之上的场线积分例程。

**后端：**

- **CPU/serial** -- 纯 Python RK4 积分器
- **CPU/parallel** -- 多进程或多线程变体
- **CUDA** -- 通过 CuPy 提供的可选 GPU 后端（最高 118× 加速）
- **OpenCL** -- 实验性

.. contents:: 子模块
   :depth: 2
   :local:

----

核心 Tracer
-----------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

动力系统层次
------------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
