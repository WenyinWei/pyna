cyna 가속 계층
==============

``cyna`` 는 pyna와 함께 제공되는 C++ 가속 계층입니다. Python hot loop를
허용하기 어려운 곳, 즉 자력선 추적, 푸앵카레 batch, 고정점 scan,
connection-length/wall hit, coil field, 함수적 섭동 이론 kernel에 사용됩니다.

빌드 계약
---------

``pyna._cyna`` 는 package 안에 컴파일된 ``_cyna_ext`` binary가 있다고
가정합니다. Source install은 xmake를 통해 이를 빌드하고, PyPI wheel은 이를
포함합니다. 플랫폼 설정과 CUDA flag는 :doc:`../installation` 을 보세요.

표준 cylindrical field-cache 순서는 다음과 같습니다.

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

``pyna.fields.VectorFieldCylind`` 또는 legacy dict를 C-contiguous array로
변환하려면 :func:`pyna._cyna.prepare_field_cache` 를 사용하세요.

고수준 API와 저수준 API
-----------------------

애플리케이션 코드에서는 고수준 wrapper를 선호하세요.

- tracing에는 ``pyna.flt`` 및 ``pyna.toroidal.flt``
- 푸앵카레 맵, cycle, island, manifold, FPT response에는 ``pyna.topo``
- coil field 구성에는 ``pyna.toroidal.coils``

``pyna._cyna`` 는 bridge boundary, diagnostics, 새 고수준 wrapper 작성 시에만
직접 사용하세요.

Python Wrapper 참조
-------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

유틸리티 Helper
---------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
