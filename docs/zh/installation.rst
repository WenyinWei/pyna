安装
====

依赖环境
--------
- Python 3.10+
- NumPy、SciPy、matplotlib
- 可选：CuPy（CUDA 加速）、joblib（计算缓存）、FEniCSx（等离子体响应求解）

从 PyPI 安装
-----------
.. code-block:: bash

   pip install pyna-chaos

从源码安装
----------
.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   pip install -e .

C++ 加速层（cyna）
------------------
.. code-block:: bash

   cd cyna
   xmake build cyna_python
   # 将生成的 _cyna_ext.so 复制到 pyna/_cyna/

验证安装
--------
.. code-block:: python

   import pyna
   print(pyna.__version__)
   
   from pyna._cyna import is_available
   print(f"cyna C++ 后端可用: {is_available()}")
