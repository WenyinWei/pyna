Tracage de lignes de champ (``pyna.flt``)
=========================================

Le paquet ``pyna.flt`` fournit des routines d'integration de lignes de champ
construites au-dessus de la hierarchie abstraite :mod:`pyna.system`.

**Backends :**

- **CPU/serial** -- intégrateur RK4 pur Python
- **CPU/parallel** -- variantes multi-processus ou multi-thread
- **CUDA** -- backend GPU facultatif via CuPy (jusqu'a 118x d'accélération)
- **OpenCL** -- experimental

.. contents:: Sous-modules
   :depth: 2
   :local:

----

Traceur principal
-----------------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

Hierarchie des systèmes dynamiques
----------------------------------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
