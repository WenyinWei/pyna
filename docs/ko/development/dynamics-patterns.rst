동역학 워크플로와 확장 헬퍼
=============================

pyna는 수학적 기하와 구성 정책을 분리합니다.

핵심 계층 구조는 간결하게 유지됩니다.

- 연속 시간 기하: ``Trajectory``, ``Cycle``, ``Tube``,
  ``TubeChain``;
- 이산 시간 기하: ``Orbit``, ``PeriodicOrbit``, ``Island``,
  ``IslandChain``;
- 토로이달 클래스는 ``pyna.topo.Tube``, ``pyna.topo.Cycle``,
  ``pyna.topo.IslandChain`` 아래의 기본 공개 위상 특수화로
  남아 있습니다.

헬퍼 계층은 이 계층 구조 주변에 하나의 사용자용 워크플로 파사드와
명시적인 확장 지점을 추가합니다.

워크플로 파사드
---------------

``TopologyWorkflow`` 는 튜토리얼과 분석 스크립트에서 가장 먼저 사용할 것을
권장하는 진입점입니다. 사용자가 실제로 따르는 경로에 맞춰 저수준 헬퍼를
조합합니다.

1. 흐름/사상을 만들거나 받는다.
2. ``Trajectory`` 를 적분하거나 ``Orbit`` 을 반복한다.
3. 닫힌 표본을 ``Cycle`` 또는 ``PeriodicOrbit`` 으로 명시적으로 승격한다.
4. ``Cycle``/``Tube``/``TubeChain`` 객체를 단면으로 자른다.

파사드는 의도적으로 얇습니다. 새로운 수학을 도입하지 않으며, 노트북 코드를
읽기 쉽게 유지하면서도 각 승격을 명시적으로 만듭니다.

실습 튜토리얼
-------------

간결한 워크플로 개요는 :doc:`/ko/mini-cases` 에서 시작하세요. 같은 승격
아이디어를 실제 토로이달 계산에 적용하는 완전한 시각 튜토리얼은
:doc:`/notebooks/i18n/ko/tutorials/RMP_resonance_analysis` 를 사용하세요. 이 튜토리얼은
표본 푸앵카레 교차점, 명시적 X/O 고정점 기하, 좌표 격자 오버레이,
국소 다양체 가지를 보여 줍니다.

짧은 복사-붙여넣기 레시피는 :doc:`/ko/mini-cases` 를 사용하세요. 그 페이지는
quickstart와 전체 API 참조 사이의 연결부로 의도되었습니다.

Protocol
--------

``pyna.topo.protocols`` 는 ``FlowLike``, ``MapLike``, ``SectionLike``,
``TubeLike`` 같은 구조적 계약을 정의합니다. 모든 기본 클래스를 직접
상속하지 않고도 pyna와 상호 운용되어야 하는 새 도메인 패키지를 추가할 때
사용하세요.

Adapter
-------

``pyna.topo.adapters`` 는 사용자 데이터를 안정적인 핵심 객체로 변환합니다.

- 배열 또는 솔버 출력을 ``Trajectory`` 및 ``Orbit`` 으로 변환
- 점 또는 고정점 유사 객체를 ``SectionPoint`` 로 변환
- 요청 시 검증된 표본을 ``PeriodicOrbit`` 또는 ``Cycle`` 로 변환

어댑터는 표현을 정규화합니다. 수학적 주장을 숨겨서는 안 됩니다.
예를 들어 열린 표본 궤적은 호출자가 명시적으로 ``Cycle`` 을 요청하고
폐합성 검사를 받아들이거나 통과시키지 않는 한 ``Trajectory`` 로 남습니다.

Builder
-------

``GeometryBuilder``, ``IslandChainBuilder``, ``TubeChainBuilder`` 는 구성
정책을 포착합니다. 워크플로가 여러 저수준 재료에서 위상을 조립할 때는
검증, 메타데이터, 역방향 링크를 중앙화하므로 빌더를 선호하세요.

Bridge
------

``CoreSectionCutBridge`` 는 핵심 객체를 위한 기본 연속-이산 브리지입니다.

- ``Cycle.section_cut(section)`` 은 ``PeriodicOrbit`` 을 반환합니다.
- ``Tube.section_cut(section)`` 은 ``IslandChain`` 을 반환합니다.
- ``TubeChain.section_cut(section)`` 은 결과 자기섬을 병합합니다.

토로이달 객체는 이미 최적화된 ``section_cut`` 메서드를 소유합니다. 이를 직접
사용하거나 ``TopologyWorkflow.section_cut(...)`` 을 호출해 객체가 자신의
구현으로 디스패치하게 하세요.

Factory
-------

``DynamicalSystemFactory`` 는 ``callable-flow``, ``callable-map``, ``hamiltonian``,
``nbody``, ``geometric-brownian-motion`` 같은 안정적인 문자열 키에서 바로
사용할 수 있는 시스템을 만듭니다.

``PoincareMapFactory`` 는 실행 가능한 반환 사상 구현을 선택합니다.
기본 ``backend="auto"`` 는 cyna 필드 캐시 인수가 제공되지 않는 한 이식 가능한
``GeneralPoincareMap`` 을 현재 선택합니다.

``GeometryFactory`` 는 빌더 계층을 통해 위상 기하를 만듭니다.
설정 기반 예제와 안정적인 구성 키가 필요한 하위 패키지에
유용합니다.

호환성 규칙
-----------

- ``pyna.topo.Tube``, ``Cycle``, ``IslandChain`` 이 핵심 클래스를 가리키도록
  바꾸지 마세요. 일반 루트에는 ``CoreTube``, ``CoreCycle``,
  ``CoreIslandChain`` 을 사용하세요.
- 토로이달 전용 경계에서 덕 타이핑된 가짜 단면을 사용하지 마세요.
  일급 ``Section`` 객체를 사용하세요.
- 레지스트리는 가변 상태로 다루세요. 격리가 중요하면 테스트와 하위
  패키지에서 로컬 ``Registry`` 인스턴스를 사용하세요.
