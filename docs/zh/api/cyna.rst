cyna C++ 加速层
===============

``cyna`` 是 pyna 的 C++ 加速层，通过 xmake 构建。它负责场线追踪、插值、
固定点扫描、壁面命中和扰动响应等瓶颈；高层科学语义仍应由 Python 端
``pyna`` 对象表达。

本地验证：

.. code-block:: python

   from pyna._cyna import is_available, get_version

   print(is_available(), get_version())

如果不可用，先检查 xmake、编译器和当前 Python ABI 是否匹配。

英文详细 API：

- :doc:`../../en/api/cyna`
