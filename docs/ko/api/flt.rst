자력선 추적 (``pyna.flt``)
===========================

``pyna.flt`` package는 추상 :mod:`pyna.system` 계층 위에 구축된 자력선 적분
routine을 제공합니다.

**백엔드:**

- **CPU/serial** -- pure-Python RK4 integrator
- **CPU/parallel** -- multi-process 또는 multi-threaded variant
- **CUDA** -- CuPy를 통한 선택적 GPU backend(최대 118× speedup)
- **OpenCL** -- experimental

.. contents:: Submodules
   :depth: 2
   :local:

----

핵심 추적기
-----------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

동역학계 계층
-------------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
