.. _zh-installation:

安装指南
========

系统要求
--------

- Python 3.10+
- NumPy, SciPy, matplotlib
- 可选: CuPy (CUDA加速), joblib, FEniCSx

从 PyPI 安装
------------

.. code-block:: bash

   pip install pyna-chaos

从源码安装
----------

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   pip install -e .

C++ 扩展 (cyna)
---------------

.. code-block:: bash

   cd cyna
   xmake build cyna_python

验证安装
--------

.. code-block:: python

   import pyna
   print(pyna.__version__)
