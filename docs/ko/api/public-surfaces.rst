공개 API 지도
=============

이 페이지는 pyna의 안정적인 공개 인터페이스로 들어가는 짧은 경로입니다.
전체 디버그 레퍼런스는 AutoAPI에 남겨 두고, 여기서는 notebook, 연구 스크립트,
하위 패키지가 우선 사용할 입구를 정리합니다.

기하 객체 어휘
--------------

어떤 solver가 만들었는지보다 위상공간의 객체 자체가 중요할 때 여기서 시작하세요.

.. list-table::
   :header-rows: 1

   * - 작업
     - 공개 입구
   * - 연속시간 샘플 궤적
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - 이산시간 map 동역학
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - 토로이달 단면 기하
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - 명시적 promotion과 adapter
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

일반 동역학
-----------

토로이달 모델이 아니어도 같은 기하 객체를 반환하려면 :mod:`pyna.dynamics` 를
사용하세요.

.. list-table::
   :header-rows: 1

   * - 모델군
     - 공개 입구
   * - ODE flow
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Hamiltonian system
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N-body system
     - :class:`pyna.dynamics.NBodySystem`
   * - 이산 map
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - SDE
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

토로이달 및 RMP 워크플로
------------------------

자기좌표, 자기력선 추적, 자기 스펙트럼 분석, 시각화 overlay에는 다음 모듈을
사용합니다.

.. list-table::
   :header-rows: 1

   * - 필요
     - 공개 입구
   * - 평형과 좌표
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - 자기력선 추적과 cache-aware workflow
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - 반변 radial perturbation spectrum
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - RMP/nRMP 진단
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - 자기 스펙트럼 그림
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Poincare, X/O 점, 자기섬 overlay
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

AutoAPI가 필요한 경우
---------------------

생성자 signature, 상속 member, 드문 진단, 구현 세부가 필요하면
:doc:`/en/api/generated/pyna/index` 를 보세요. 새 tutorial과 사용자용 예제는 가능한 한
위의 공개 입구만 사용하도록 작성합니다.
