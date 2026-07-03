Feldlinienverfolgung (``pyna.flt``)
===================================

Das Paket ``pyna.flt`` stellt Routinen zur Feldlinienintegration bereit, die
auf der abstrakten Hierarchie :mod:`pyna.system` aufbauen.

**Backends:**

- **CPU/serial** -- reiner Python-RK4-Integrator
- **CPU/parallel** -- Multi-Prozess- oder Multi-Thread-Varianten
- **CUDA** -- optionales GPU-Backend über CuPy (bis zu 118× Beschleunigung)
- **OpenCL** -- experimentell

.. contents:: Submodules
   :depth: 2
   :local:

----

Kern-Tracer
-----------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Hierarchie dynamischer Systeme
------------------------------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
