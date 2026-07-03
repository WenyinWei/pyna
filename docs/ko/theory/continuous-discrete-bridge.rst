연속 및 이산 기하
=================

pyna는 continuous-time 동역학계와 discrete-time 동역학계에 서로 다른 object
family를 사용합니다.

Continuous-time 쪽:

- ``Trajectory`` 는 표본화된 finite-time geometry입니다.
- ``Cycle`` 은 flow의 periodic orbit입니다.
- ``Tube`` 는 elliptic cycle 주변의 resonance zone이며, hyperbolic cycle로
  경계가 정해질 수 있습니다.
- ``TubeChain`` 은 하나의 resonance에 속한 tube를 묶습니다.

Discrete-time 쪽:

- ``Orbit`` 은 표본화된 map iteration geometry입니다.
- ``PeriodicOrbit`` 은 map의 닫힌 orbit입니다.
- ``Island`` 는 section 위의 하나의 reduced resonance island입니다.
- ``IslandChain`` 은 section 수준의 island chain입니다.

두 쪽 사이의 bridge는 section cut입니다. ``Cycle`` 을 푸앵카레 section으로
자르면 return map의 ``PeriodicOrbit`` 이 만들어집니다. ``Tube`` 를 자르면
``IslandChain`` 이 만들어집니다. ``TubeChain`` 을 자르면 그 tube들에서 나온
island chain이 병합됩니다.

이 분리는 의도적입니다. 수치 trajectory는 invariance를 증명하지 않아도 유용한
geometry일 수 있습니다. 따라서 builder와 adapter는 promotion을 명시적으로
만듭니다. 사용자는 표본 trajectory가 ``Cycle`` 이 되기 전이나 map sample이
``PeriodicOrbit`` 이 되기 전에 closure check를 요구할 수 있습니다.

같은 어휘는 일반 유한 차원 system과 toroidal magnetic-field-line specialization이
공유합니다. Generic root는 ``pyna.topo.CoreTube`` 와 관련 이름으로 사용할 수
있고, toroidal default는 ``pyna.topo.Tube``, ``pyna.topo.Cycle``,
``pyna.topo.IslandChain`` 으로 계속 사용할 수 있습니다.

관련 항목
---------

- :doc:`/ko/mini-cases`
- :doc:`/notebooks/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/tutorials/island_jacobian_analysis`
