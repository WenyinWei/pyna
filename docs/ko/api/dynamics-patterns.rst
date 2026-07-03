동역학 워크플로와 확장 Helper
=============================

``pyna.topo`` 는 core topology object 주변의 workflow helper를 노출합니다.
주요 사용자 진입점은 ``TopologyWorkflow`` 입니다. 저수준 Protocol, Adapter,
Builder, Bridge, Factory 모듈도 downstream library가 안정적인 extension point를
필요로 할 때 사용할 수 있도록 남아 있습니다.

워크플로 Facade
---------------

``TopologyWorkflow`` 는 notebook과 일상적인 script를 위해 설계되었습니다.
새로운 수학적 객체 type을 추가하지 않고도 system construction, integration/iteration,
명시적 promotion, section cut을 결합합니다.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocol
--------

Structural protocol은 외부 system을 위한 extension contract를 설명합니다.
Third-party object는 필요한 attribute와 method를 구현하면 참여할 수 있으며,
pyna class를 subclass하는 것은 선택 사항입니다.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adapter
-------

Adapter는 array, solver output, 기존 pyna object를 core geometry representation으로
정규화합니다. 표본 데이터를 invariant object로 조용히 승격하지 않습니다.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builder
-------

Builder는 명시적인 promotion rule을 encode합니다. 예를 들어 trajectory는
닫힌 표본을 요구할 수 있는 builder 또는 adapter 호출을 통해서만 ``Cycle`` 로
승격될 수 있습니다.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridge
------

Bridge는 continuous-time object family와 discrete-time object family를 연결합니다.
``Cycle -> PeriodicOrbit`` 및 ``Tube/TubeChain -> IslandChain`` 입니다.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factory와 Registry
------------------

Factory는 system, geometry, Poincare map을 위한 안정적인 construction entry point를
제공합니다. Registry는 명시적이고 duplicate-safe하므로 test와 downstream library가
자신의 extension을 격리할 수 있습니다.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
