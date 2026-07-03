토러스 변형 이론
================

``pyna.toroidal.torus_deformation`` 은 invariant torus와 resonant structure가
제어된 perturbation에 어떻게 응답하는지 연구하는 데 쓰는 해석적 torus-deformation
도구를 포함합니다.

개념적 역할
-----------

geometry hierarchy에서:

- invariant torus는 ``InvariantTorus`` 입니다.
- resonant elliptic cycle은 ``Tube`` 의 core입니다.
- hyperbolic cycle은 tube의 경계를 정하고 stable/unstable manifold를 생성합니다.
- tube를 푸앵카레 section으로 자르면 ``IslandChain`` object가 만들어집니다.

따라서 torus-deformation 계산은 topology control로 직접 이어집니다. 어떤 spectral
perturbation이 resonant structure를 이동, 분리, 치유 또는 억제하는지 예측하기
때문입니다.

공개 API
--------

.. automodule:: pyna.toroidal.torus_deformation
   :no-index:
   :members:
   :show-inheritance:

관련 모듈
---------

.. automodule:: pyna.toroidal.perturbation_spectrum
   :no-index:
   :members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.island_control
   :no-index:
   :members:
   :show-inheritance:
