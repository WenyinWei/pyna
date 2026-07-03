安装
====

支持的 Python 版本
------------------

``pyna-chaos`` 支持 Linux、macOS 和 Windows 上的 CPython 3.9 到
3.13。核心 Python 依赖包括 NumPy、SciPy、Matplotlib、SymPy、h5py、
joblib 和 Plotly。Prefect 编排和 CUDA 加速是可选功能。

从 PyPI 安装
------------

如果你的平台已有发布的 wheel，优先使用它：

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

wheel 包含必需的 ``cyna`` C++ 扩展。缺少 ``pyna._cyna`` 扩展应视为安装
问题，而不是正常的可选后端状态。

验证安装：

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

核心包不会安装 Prefect 编排。需要 Prefect 支持的工作流时，请安装 workflow
extra：

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

工作流轨迹/轨道缓存会保存为由 pyna 管理的带版本 payload。Prefect 只用于编排；
它不是持久缓存文件格式。

从源码安装
----------

editable/source 安装会通过 ``setup.py`` 调用 xmake 构建 ``cyna``：

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

源码构建需要：

- C++17 编译器：GCC 9+、Clang 10+、Apple Clang 或 MSVC 2019+
- xmake 2.8+
- pybind11 头文件，通常由 pip 安装

构建脚本会尝试在常见平台上 bootstrap xmake 和最小编译工具链。在受限的 CI
镜像中，请预先安装这些工具，并设置 ``CYNA_SKIP_TOOL_INSTALL=1``，这样缺少工具时
可以快速失败。

cyna C++ 加速
-------------

``cyna`` 是场线追踪、Poincare 映射、固定点扫描、线圈场、壁面/连接长度扫描以及
函数扰动理论核所使用的 C++ 层。Python/C++ 边界处的规范分量顺序为：

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

手动低层构建：

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

xmake 的 ``after_build`` hook 会把 ``_cyna_ext.so`` 或 ``_cyna_ext.pyd`` 复制到
``pyna/_cyna``。应用代码应从 ``pyna.flt``、``pyna.toroidal.flt``、
``pyna.topo`` 和 ``pyna._cyna`` 导入高层 wrapper，而不是直接导入原始扩展。

CUDA
----

已发布的 wheel 仅支持 CPU。本地源码构建在发现 ``nvcc`` 时会自动启用独立 CUDA
后端，除非设置了 ``CYNA_WITH_CUDA=0``。

常用模式：

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

主 ``_cyna_ext`` 模块不链接 CUDA。只有在调用支持 CUDA 的线圈场函数时，CUDA 代码
才会被加载。

开发安装
--------

用于测试、notebook 和文档：

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

本地构建文档：

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

故障排查
--------

``ImportError: pyna._cyna requires the compiled cyna extension``
   从 PyPI 安装平台 wheel，或使用 xmake 和 C++17 编译器从源码重新构建。

``xmake: command not found``
   手动安装 xmake，然后重新运行 ``python -m pip install -e .``。

``pybind11 headers not found``
   在用于构建 pyna 的同一环境中运行 ``python -m pip install pybind11``。

CUDA build fails but CPU is acceptable
   使用 ``CYNA_WITH_CUDA=0`` 重新构建。
