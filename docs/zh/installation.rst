.. _zh-installation:

安装指南
========

支持环境
--------

``pyna-chaos`` 支持 CPython 3.9 到 3.13，覆盖 Linux、macOS 和 Windows。
核心依赖包括 NumPy、SciPy、Matplotlib、SymPy、h5py、joblib、Prefect 和
Pydantic。CUDA 加速是可选能力。

从 PyPI 安装
------------

优先使用官方 wheel：

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

wheel 内应包含必需的 ``cyna`` C++ 扩展。若 ``pyna._cyna`` 无法导入，应
按安装/构建问题处理，而不是当作正常的可选后端缺失。

验证安装：

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

从源码安装
----------

源码安装会通过 ``setup.py`` 调用 xmake 编译 ``cyna``：

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

源码构建需要：

- C++17 编译器：GCC 9+、Clang 10+、Apple Clang 或 MSVC 2019+
- xmake 2.8+
- pybind11 头文件，通常由 pip 安装

构建脚本会在常见平台上尝试自动安装 xmake 和最小编译工具链。若 CI 或
服务器环境不允许自动安装，请预装这些工具并设置 ``CYNA_SKIP_TOOL_INSTALL=1``。

cyna C++ 加速层
---------------

``cyna`` 是 pyna 的 C++ 加速层，覆盖场线追踪、Poincare 映射、固定点批量
扫描、线圈场、壁/连接长度扫描和 FPT 内核。Python/C++ 边界的柱坐标磁场
分量顺序固定为：

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

手动底层构建：

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

xmake 的 ``after_build`` 钩子会把 ``_cyna_ext.so`` 或 ``_cyna_ext.pyd``
复制到 ``pyna/_cyna``。日常代码应优先使用 ``pyna.flt``、``pyna.toroidal.flt``、
``pyna.topo`` 和 ``pyna._cyna`` 的高层接口。

CUDA
----

PyPI wheel 默认是 CPU-only。本地源码构建在检测到 ``nvcc`` 时会尝试编译独立
CUDA 后端，除非设置 ``CYNA_WITH_CUDA=0``。

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # 强制 CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # 要求 CUDA 后端构建成功

主扩展 ``_cyna_ext`` 不直接链接 CUDA；只有调用 CUDA 能力时才加载独立后端。

开发与文档
----------

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

本地构建文档：

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

常见问题
--------

``ImportError: pyna._cyna requires the compiled cyna extension``
   安装 PyPI wheel，或在有 xmake 与 C++17 编译器的环境中重新源码构建。

``xmake: command not found``
   手动安装 xmake 后重新执行 ``python -m pip install -e .``。

``pybind11 headers not found``
   在同一个 Python 环境里执行 ``python -m pip install pybind11``。

CUDA 构建失败但 CPU 后端可接受
   设置 ``CYNA_WITH_CUDA=0`` 后重新构建。
