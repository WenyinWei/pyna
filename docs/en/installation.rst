Installation
============

Requirements
------------
- Python 3.10+
- NumPy, SciPy, matplotlib
- Optional: CuPy (CUDA acceleration), joblib (caching), FEniCSx (plasma response)

From PyPI
---------
.. code-block:: bash

   pip install pyna-chaos

From source
-----------
.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   pip install -e .

C++ acceleration (cyna)
-----------------------
.. code-block:: bash

   cd cyna
   xmake build cyna_python
   # Then copy _cyna_ext.so to pyna/_cyna/

Verify installation
-------------------
.. code-block:: python

   import pyna
   print(pyna.__version__)
   
   # Check CUDA availability
   from pyna.flt import get_backend
   print(get_backend('cuda'))  # 'cuda' or falls back to 'cpu'
   
   # Check cyna C++ backend
   from pyna._cyna import is_available
   print(f"cyna C++ backend: {is_available()}")
