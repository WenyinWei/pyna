Installation
============

Supported Python versions
-------------------------

``pyna-chaos`` supports CPython 3.9 through 3.13 on Linux, macOS, and Windows.
The core Python dependencies are NumPy, SciPy, Matplotlib, SymPy, h5py, joblib,
and Plotly.  Prefect orchestration and CUDA acceleration are optional.

From PyPI
---------

Use the published wheel whenever one is available for your platform:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

The wheel includes the required ``cyna`` C++ extension.  A missing
``pyna._cyna`` extension should be treated as an installation problem, not as a
normal optional-backend state.

Verify the install:

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

Prefect orchestration is not installed by the core package.  Install the
workflow extra when you need Prefect-backed workflows:

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

Workflow trajectory/orbit caches are stored as pyna-managed versioned payloads.
Prefect is used for orchestration; it is not the durable cache file format.

From source
-----------

Editable/source installs build ``cyna`` with xmake through ``setup.py``:

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

Source builds need:

- a C++17 compiler: GCC 9+, Clang 10+, Apple Clang, or MSVC 2019+
- xmake 2.8+
- pybind11 headers, normally installed by pip

The build script tries to bootstrap xmake and a minimal compiler toolchain on
common platforms.  In locked-down CI images, preinstall them and set
``CYNA_SKIP_TOOL_INSTALL=1`` to fail fast when a tool is missing.

cyna C++ Acceleration
---------------------

``cyna`` is the C++ layer used by field-line tracing, Poincare maps, fixed
point scans, coil fields, wall/connection-length scans, and functional
perturbation theory kernels.  The canonical component order at the Python/C++
boundary is:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Manual low-level build:

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

The xmake ``after_build`` hook copies ``_cyna_ext.so`` or ``_cyna_ext.pyd`` into
``pyna/_cyna``.  Application code should import the high-level wrappers from
``pyna.flt``, ``pyna.toroidal.flt``, ``pyna.topo`` and ``pyna._cyna`` rather
than importing the raw extension directly.

CUDA
----

Published wheels are CPU-only.  Local source builds auto-enable the separate
CUDA backend when ``nvcc`` is available unless ``CYNA_WITH_CUDA=0`` is set.

Useful modes:

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

The main ``_cyna_ext`` module does not link against CUDA.  CUDA code is loaded
only when a CUDA-capable coil-field call is made.

Development Install
-------------------

For tests, notebooks, and documentation:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

Build the documentation locally:

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

Troubleshooting
---------------

``ImportError: pyna._cyna requires the compiled cyna extension``
   Install a platform wheel from PyPI or rebuild from source with xmake and a
   C++17 compiler.

``xmake: command not found``
   Install xmake manually, then rerun ``python -m pip install -e .``.

``pybind11 headers not found``
   Run ``python -m pip install pybind11`` in the same environment used to build
   pyna.

CUDA build fails but CPU is acceptable
   Rebuild with ``CYNA_WITH_CUDA=0``.
