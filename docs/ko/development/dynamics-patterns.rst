동역학 워크플로와 확장 Helper
=============================

pyna는 수학적 geometry와 construction policy를 분리합니다.

Core hierarchy는 compact하게 유지됩니다.

- continuous-time geometry: ``Trajectory``, ``Cycle``, ``Tube``,
  ``TubeChain``;
- discrete-time geometry: ``Orbit``, ``PeriodicOrbit``, ``Island``,
  ``IslandChain``;
- toroidal class는 ``pyna.topo.Tube``, ``pyna.topo.Cycle``,
  ``pyna.topo.IslandChain`` 아래의 기본 public topology specialization으로
  남아 있습니다.

Helper layer는 이 hierarchy 주변에 하나의 사용자-facing workflow facade와
명시적인 extension point를 추가합니다.

워크플로 Facade
---------------

``TopologyWorkflow`` 는 tutorial과 analysis script에서 가장 먼저 사용할 것을
권장하는 진입점입니다. 사용자가 실제로 따르는 경로에 맞춰 저수준 helper를
조합합니다.

1. flow/map을 만들거나 받는다.
2. ``Trajectory`` 를 적분하거나 ``Orbit`` 을 iterate한다.
3. 닫힌 표본을 ``Cycle`` 또는 ``PeriodicOrbit`` 으로 명시적으로 승격한다.
4. ``Cycle``/``Tube``/``TubeChain`` object를 section으로 자른다.

Facade는 의도적으로 얇습니다. 새로운 수학을 도입하지 않으며, notebook 코드를
읽기 쉽게 유지하면서도 각 promotion을 명시적으로 만듭니다.

실습 Tutorial
-------------

간결한 workflow 개요는 :doc:`/ko/mini-cases` 에서 시작하세요. 같은 promotion
아이디어를 실제 toroidal calculation에 적용하는 완전한 시각 튜토리얼은
:doc:`/notebooks/i18n/ko/tutorials/RMP_resonance_analysis` 를 사용하세요. 이 튜토리얼은
표본 푸앵카레 교차점, 명시적 X/O fixed-point geometry, coordinate-grid overlay,
local manifold branch를 보여 줍니다.

짧은 copy-paste recipe는 :doc:`/ko/mini-cases` 를 사용하세요. 그 페이지는
quickstart와 전체 API reference 사이의 bridge로 의도되었습니다.

Protocol
--------

``pyna.topo.protocols`` 는 ``FlowLike``, ``MapLike``, ``SectionLike``,
``TubeLike`` 같은 structural contract를 정의합니다. 모든 base class를 직접
상속하지 않고도 pyna와 상호 운용되어야 하는 새 domain package를 추가할 때
사용하세요.

Adapter
-------

``pyna.topo.adapters`` 는 사용자 데이터를 안정적인 core object로 변환합니다.

- array 또는 solver output을 ``Trajectory`` 및 ``Orbit`` 으로 변환
- point 또는 fixed-point-like object를 ``SectionPoint`` 로 변환
- 요청 시 검증된 표본을 ``PeriodicOrbit`` 또는 ``Cycle`` 로 변환

Adapter는 representation을 정규화합니다. 수학적 주장을 숨겨서는 안 됩니다.
예를 들어 열린 표본 trajectory는 caller가 명시적으로 ``Cycle`` 을 요청하고
closure check를 받아들이거나 통과시키지 않는 한 ``Trajectory`` 로 남습니다.

Builder
-------

``GeometryBuilder``, ``IslandChainBuilder``, ``TubeChainBuilder`` 는 construction
policy를 포착합니다. workflow가 여러 저수준 재료에서 topology를 조립할 때는
validation, metadata, back-link를 중앙화하므로 builder를 선호하세요.

Bridge
------

``CoreSectionCutBridge`` 는 core object를 위한 기본 continuous-to-discrete bridge입니다.

- ``Cycle.section_cut(section)`` 은 ``PeriodicOrbit`` 을 반환합니다.
- ``Tube.section_cut(section)`` 은 ``IslandChain`` 을 반환합니다.
- ``TubeChain.section_cut(section)`` 은 결과 island를 병합합니다.

toroidal object는 이미 optimized ``section_cut`` method를 소유합니다. 이를 직접
사용하거나 ``TopologyWorkflow.section_cut(...)`` 을 호출해 object가 자신의
implementation으로 dispatch하게 하세요.

Factory
-------

``DynamicalSystemFactory`` 는 ``callable-flow``, ``callable-map``, ``hamiltonian``,
``nbody``, ``geometric-brownian-motion`` 같은 안정적인 문자열 key에서 바로
사용할 수 있는 system을 만듭니다.

``PoincareMapFactory`` 는 실행 가능한 return-map implementation을 선택합니다.
기본 ``backend="auto"`` 는 cyna field-cache argument가 제공되지 않는 한 portable
``GeneralPoincareMap`` 을 현재 선택합니다.

``GeometryFactory`` 는 builder layer를 통해 topology geometry를 만듭니다.
config-driven example과 안정적인 construction key가 필요한 downstream package에
유용합니다.

호환성 규칙
-----------

- ``pyna.topo.Tube``, ``Cycle``, ``IslandChain`` 이 core class를 가리키도록
  바꾸지 마세요. generic root에는 ``CoreTube``, ``CoreCycle``,
  ``CoreIslandChain`` 을 사용하세요.
- toroidal-only boundary에서 duck-typed fake section을 사용하지 마세요.
  first class ``Section`` object를 사용하세요.
- registry는 mutable state로 다루세요. isolation이 중요하면 test와 downstream
  package에서 local ``Registry`` instance를 사용하세요.
