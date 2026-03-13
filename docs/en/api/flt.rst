Field-line Tracing (``pyna.flt``)
==================================

The ``pyna.flt`` package provides field-line integration routines built on
top of the abstract :mod:`pyna.system` hierarchy.

**Backends:**

- **CPU/serial** -- pure-Python RK4 integrator
- **CPU/parallel** -- multi-process or multi-threaded variants
- **CUDA** -- optional GPU backend via CuPy (up to 118× speedup)
- **OpenCL** -- experimental

.. contents:: Submodules
   :depth: 2
   :local:

----

Core Tracer
-----------

.. automodule:: pyna.flt
   :members:
   :undoc-members:
   :show-inheritance:

----

Dynamical System Hierarchy
--------------------------

.. automodule:: pyna.system
   :members:
   :undoc-members:
   :show-inheritance:
