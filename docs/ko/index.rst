pyna - Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-LGPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** 는 **동역학계 분석** 과 **자기 구속 핵융합 물리** 를 위한
Python 라이브러리입니다. 자력선 추적, 푸앵카레 맵, 해밀토니안 계,
N-body 상호작용, 유한 차원 맵, Ito SDE를 다루며, 표본 데이터를
위상공간 기하로 승격할 때 공통으로 쓰는 위상수학 어휘도 제공합니다.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 빠른 시작
      :link: quickstart
      :link-type: doc

      설치, cyna 확인, 첫 토로이달 예제와 일반 동역학 예제를 실행합니다.

   .. grid-item-card:: 미니 사례
      :link: mini-cases
      :link-type: doc

      ODE, 해밀토니안 계, 맵, SDE, 위상 객체 승격을 위한 짧은 레시피입니다.

   .. grid-item-card:: 튜토리얼
      :link: tutorials/index
      :link-type: doc

      Monte Carlo SDE 분포 추정을 포함한 실행 완료 notebook과 서술형 안내입니다.

   .. grid-item-card:: API 레퍼런스
      :link: api/index
      :link-type: doc

      직접 작성한 모듈 안내와 생성된 소스 레퍼런스를 함께 제공합니다.

.. toctree::
   :maxdepth: 2
   :caption: 문서

   installation
   quickstart
   mini-cases
   tutorials/index
   api/index
   theory/index
   development/index
