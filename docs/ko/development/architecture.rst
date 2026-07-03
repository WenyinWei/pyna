아키텍처
========

pyna는 두 가지 생각을 중심으로 구성되어 있습니다.

1. 동역학계는 유한 차원 위상공간의 evolution rule을 정의한다.
2. topology module은 그 위상공간 안에 사는 geometry object를 설명한다.

이 분리는 같은 object hierarchy가 토로이달 자기장 선 구조, 해밀토니안 공명대,
고전 map, N-body orbit, 확률적 sample path를 표현할 수 있게 합니다.

계층 0: 동역학
--------------

``pyna.topo.dynamics`` 는 추상 수학 계층을 제공합니다.

- ``PhaseSpace``
- ``ContinuousFlow``
- ``HamiltonianFlow``
- ``DiscreteMap``
- ``PoincareMap`` 및 ``GeneralPoincareMap``

``pyna.dynamics`` 는 바로 사용할 수 있는 유한 차원 system을 추가합니다.

- ``CallableFlow`` 및 ``CallableMap``
- ``HamiltonianSystem`` 및 ``SeparableHamiltonianSystem``
- ``NBodySystem``
- ``ItoSDE``, ``BrownianMotion`` 및 ``GeometricBrownianMotion``

이 class들은 표본 output에 topology core를 사용합니다. deterministic flow
trajectory는 ``pyna.topo.core.Trajectory`` 이고, discrete iterate cloud는
``pyna.topo.core.Orbit`` 입니다.

계층 1: 기하
------------

``pyna.topo.core`` 는 domain-agnostic geometry hierarchy입니다.

.. list-table::
   :header-rows: 1

   * - Class
     - 의미
     - 시간 type
   * - ``Trajectory``
     - 위상공간 안의 유한 표본 곡선
     - continuous
   * - ``Cycle``
     - continuous flow의 periodic orbit
     - continuous
   * - ``Tube``
     - elliptic cycle 주변의 resonance zone
     - continuous
   * - ``TubeChain``
     - 하나의 resonance를 공유하는 tube family
     - continuous
   * - ``Orbit``
     - map의 유한 표본 iterate
     - discrete
   * - ``PeriodicOrbit``
     - map의 유한 periodic orbit
     - discrete
   * - ``Island``
     - section 위의 하나의 reduced resonance island
     - discrete
   * - ``IslandChain``
     - section 위의 periodic island chain
     - discrete

핵심 bridge는 ``section_cut`` 입니다.

.. code-block:: text

   Cycle       --section_cut--> PeriodicOrbit
   Tube        --section_cut--> IslandChain
   TubeChain   --section_cut--> IslandChain

이는 continuous magnetic island tube가 푸앵카레 단면에서 discrete island chain으로
관측되는 toroidal workflow와 대응됩니다.

계층 2: 토로이달 특수화
-----------------------

``pyna.topo.toroidal`` 은 generic core를 subclass합니다.

.. code-block:: text

   core.SectionPoint   -> toroidal.FixedPoint
   core.PeriodicOrbit  -> toroidal.PeriodicOrbit
   core.Cycle          -> toroidal.Cycle
   core.Island         -> toroidal.Island
   core.IslandChain    -> toroidal.IslandChain
   core.Tube           -> toroidal.Tube
   core.TubeChain      -> toroidal.TubeChain

toroidal layer는 다음을 추가합니다.

- ``R``, ``Z`` 및 ``phi`` coordinates
- winding number ``(m, n)``
- ``DPm`` 및 monodromy classification
- cyna-accelerated section cut 및 tracing
- section-view correspondence 및 reconstruction helper

계층 3: 워크플로와 확장 Helper
------------------------------

``pyna.topo.protocols``, ``adapters``, ``builders``, ``bridges`` 및
``factories`` 는 software-engineering extension layer를 제공합니다. Notebook
사용자를 위한 주 진입점은 ``TopologyWorkflow`` 입니다. 이러한 helper는
construction policy와 backend selection을 수학적 dataclass 밖에 둡니다.
외부 system은 protocol을 따라 호환되고, adapter로 데이터를 정규화하며,
builder를 통해 object를 승격하고, bridge로 continuous geometry를 자르며,
factory로 runtime implementation을 선택할 수 있습니다.

계층 4: 가속
------------

``cyna`` 는 고수준 pyna API 뒤의 병목을 구현합니다. 고수준 과학 object semantics를
소유해서는 안 되며, tracing, interpolation, fixed-point scan, wall hit,
perturbation response를 위한 빠른 kernel을 제공합니다.

설계 규칙
---------

- 새 유한 차원 geometry에는 generic ``pyna.topo.core`` class를 선호하세요.
- toroidal-specific field는 ``pyna.topo.toroidal`` subclass에만 추가하세요.
- 표본 finite trajectory는 geometry이지만 자동으로 invariant set이 되지는 않습니다.
- periodic structure가 model의 일부이거나 수치적으로 검증된 경우에만 object를
  ``Cycle``/``PeriodicOrbit`` 으로 승격하세요.
- cyna는 bridge boundary에 두세요. application-level API는 raw C++ array가 아니라
  pyna object를 반환해야 합니다.
