동역학 워크플로와 확장 헬퍼
=============================

``pyna.topo`` 는 핵심 위상 객체 주변의 워크플로 헬퍼를 노출합니다.
주요 사용자 진입점은 ``TopologyWorkflow`` 입니다. 저수준 프로토콜, 어댑터,
빌더, 브리지, 팩토리 모듈도 하위 라이브러리가 안정적인 확장 지점을
필요로 할 때 사용할 수 있도록 남아 있습니다.

워크플로 파사드
---------------

``TopologyWorkflow`` 는 노트북과 일상적인 스크립트를 위해 설계되었습니다.
새로운 수학적 객체 유형을 추가하지 않고도 시스템 구성, 적분/반복,
명시적 승격, 단면 절단을 결합합니다.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocol
--------

구조적 프로토콜은 외부 시스템을 위한 확장 계약을 설명합니다.
서드파티 객체는 필요한 속성과 메서드를 구현하면 참여할 수 있으며,
pyna 클래스를 하위 클래스화하는 것은 선택 사항입니다.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adapter
-------

어댑터는 배열, 솔버 출력, 기존 pyna 객체를 핵심 기하 표현으로
정규화합니다. 표본 데이터를 불변 객체로 조용히 승격하지 않습니다.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builder
-------

빌더는 명시적인 승격 규칙을 인코딩합니다. 예를 들어 궤적은
닫힌 표본을 요구할 수 있는 빌더 또는 어댑터 호출을 통해서만 ``Cycle`` 로
승격될 수 있습니다.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridge
------

브리지는 연속 시간 객체 계열과 이산 시간 객체 계열을 연결합니다.
``Cycle -> PeriodicOrbit`` 및 ``Tube/TubeChain -> IslandChain`` 입니다.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factory와 Registry
------------------

팩토리는 시스템, 기하, Poincare map을 위한 안정적인 구성 진입점을
제공합니다. 레지스트리는 명시적이고 중복에 안전하므로 테스트와 하위 라이브러리가
자신의 확장을 격리할 수 있습니다.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
