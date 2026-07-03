cyna 加速层
===========

``cyna`` 是随 pyna 一起发布的 C++ 加速层。它用于 Python 热循环不可接受的场景：
场线追踪、Poincare 批处理、固定点扫描、连接长度/壁面命中、线圈场，以及函数扰动
理论核。

构建约定
--------

``pyna._cyna`` 期望包内存在已编译的 ``_cyna_ext`` 二进制文件。源码安装通过 xmake
构建它；PyPI wheel 已包含它。平台配置和 CUDA flag 见 :doc:`../installation`。

规范的柱坐标场缓存顺序为：

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

使用 :func:`pyna._cyna.prepare_field_cache` 可以把
``pyna.fields.VectorFieldCylind`` 或旧式 dict 转换为 C-contiguous 数组。

高层与低层 API
--------------

应用代码优先使用高层 wrapper：

- ``pyna.flt`` 和 ``pyna.toroidal.flt`` 用于追踪
- ``pyna.topo`` 用于 Poincare 映射、cycle、磁岛、流形和 FPT 响应
- ``pyna.toroidal.coils`` 用于构造线圈场

只有在 bridge 边界、诊断场景，或编写新的高层 wrapper 时，才直接使用
``pyna._cyna``。

Python 包装层参考
-----------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

工具辅助函数
------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
