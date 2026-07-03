미니 사례
=========

이 페이지는 빠른 시작과 전체 API 레퍼런스 사이의 짧은 경로입니다. 이미 다루는
시스템의 종류를 알고 있고, 동작하는 가장 작은 pyna 패턴이 필요할 때 사용하세요.

어느 진입점을 쓸까?
-------------------

.. list-table::
   :header-rows: 1

   * - 가지고 있는 것
     - 시작점
     - 보통 얻는 기하
   * - ODE ``dx/dt = f(x,t)``
     - ``CallableFlow`` 또는 ``TopologyWorkflow.system("callable-flow", ...)``
     - ``Trajectory`` 이후 필요하면 ``Cycle``
   * - 해밀토니안 ``H(q,p,t)``
     - ``SeparableHamiltonianSystem`` 또는 ``HamiltonianSystem``
     - ``Trajectory`` / ``Cycle``
   * - 유한 차원 맵 ``x -> F(x)``
     - ``CallableMap``
     - ``Orbit`` 이후 필요하면 ``PeriodicOrbit``
   * - 토로이달 자기장
     - ``pyna.flt`` / ``pyna.topo`` / ``pyna.toroidal``
     - ``Cycle``, ``Tube``, ``IslandChain``, manifolds
   * - 확률적 교육 모델
     - ``BrownianMotion`` 또는 ``GeometricBrownianMotion``
     - 표본 ``Trajectory`` 와 통계량

사례 1: ODE 표본에서 닫힌 Cycle로
----------------------------------

``Trajectory`` 는 표본 데이터라는 뜻입니다. ``Cycle`` 은 그 표본이 닫혀 있다는
더 강한 주장을 하고 있다는 뜻입니다.

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow

   wf = TopologyWorkflow(closure_tol=2e-2)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(wf.closing_error(traj))
   cycle = wf.closed_cycle(traj)
   print(cycle.period_value, cycle.ambient_dim)

운영용 workflow에서는 닫힘 허용오차를 명시적으로 유지하세요. 그래야 수치적
가정을 검토할 수 있습니다.

사례 2: 맵 반복에서 주기 Orbit으로
-----------------------------------

맵은 먼저 ``Orbit`` 객체를 만듭니다. 닫힌 표본임이 알려져 있거나 수치적으로
검증된 경우에만 ``PeriodicOrbit`` 으로 승격하세요.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableMap
   from pyna.topo import TopologyWorkflow

   flip = CallableMap(lambda x: np.array([-x[0], -x[1]]), dim=2)
   wf = TopologyWorkflow(closure_tol=1e-12)

   orbit = wf.orbit(flip, [1.0, 0.0], n_iter=2)
   periodic = wf.periodic_orbit(
       orbit.states[:-1],
       map_obj=flip,
       coordinate_names=("x", "y"),
   )
   print(periodic.period, periodic.points[0].state)

맵이 다른 패키지에서 온다면 ``CallableMap`` 으로 감싸거나 ``__call__(x)`` 와
``phase_space`` 속성을 구현하세요.

사례 3: 해석적 Stellarator O/X 점
----------------------------------

자기 구속 작업에서는 자력선 흐름을 푸앵카레 단면으로 자릅니다. 실행 가능한
튜토리얼 :doc:`/notebooks/tutorials/RMP_resonance_analysis` 는 이제 전체
시각적 계산을 담고 있습니다.

1. 공개 해석적 stellarator 모델을 만든다.
2. divergence-free ``m=1`` 및 ``m>1`` RMP template를 검증한다.
3. 무섭동 및 섭동 푸앵카레 단면을 추적한다.
4. 해석적 공명 X/O 위상을 ``cyna`` Newton 고정점과 비교한다.
5. contravariant ``B^r`` pcolormesh atlas, 선택적 푸앵카레 투영을 포함한
   ``q``/``m/n`` resonance map, interactive Plotly 3-D bar, radial
   fixed-``n``/fixed-``m`` map, resonance curve, 켜고 끌 수 있는
   island-width marker로 다성분 RMP spectrum을 분석한다.
6. 모든 비공명 spectrum row에서 총 nRMP 응답을 계산한다.
7. contribution table은 순위와 수렴 진단용으로만 사용한다.
8. nRMP flux-surface deformation과 field-line speed modulation을 시각화한다.
9. 국소 안정 branch와 PEST 스타일 coordinate grid를 겹쳐 그린다.

고정점 plotting, section geometry, RMP/nRMP diagnostics, tutorial rendering을
변경할 때 이 notebook을 사용하세요. 문서를 게시하기 전에 로컬에서 실행할 만큼
작지만, downstream 분석 스크립트가 사용하는 공개 helper API는 충분히
검사합니다.

사례 4: 사용자 정의 System 등록
--------------------------------

Factory는 선택 사항입니다. downstream 프로젝트가 configuration-driven일 때
중요해집니다.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import CallableFlow
   from pyna.topo.factories import DynamicalSystemFactory

   def make_damped_oscillator(gamma=0.1):
       return CallableFlow(
           lambda x, t: np.array([x[1], -x[0] - gamma*x[1]]),
           dim=2,
           coordinate_names=("q", "p"),
           label="damped oscillator",
       )

   DynamicalSystemFactory.register(
       "damped-oscillator",
       lambda gamma=0.1: make_damped_oscillator(gamma),
       overwrite=True,
   )
   flow = DynamicalSystemFactory.create("damped-oscillator", gamma=0.05)

전역 등록 때문에 테스트 순서에 의존성이 생긴다면 테스트에서는 로컬 ``Registry``
인스턴스를 사용하세요.

사례 5: SDE 분포 추정
---------------------

단일 SDE 경로는 pyna trajectory입니다. Monte Carlo ensemble은 통계적
추정기입니다. pyna에 전용 ensemble 객체가 생기기 전까지는 배열로 유지하세요.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import BrownianMotion, GeometricBrownianMotion

   bm = BrownianMotion(dim=1, diffusion=1.0)
   path = bm.euler_maruyama([0.0], (0.0, 1.0), dt=0.01, rng=1)
   print(path.final)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=100_000)
   terminal = 100.0 * np.exp(gbm.expected_log_growth()[0] + gbm.sigma[0] * z)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

Brownian, Ornstein-Uhlenbeck, geometric Brownian motion 분포를 포함한 전체
실행 사례는 :doc:`/ko/tutorials/sde-monte-carlo` 를 보세요.

사례 6: 어디를 사용자 정의할까
------------------------------

.. list-table::
   :header-rows: 1

   * - 목표
     - 확장 지점
     - 유의할 점
   * - 새 물리 모델
     - ``CallableFlow``, ``HamiltonianSystem`` 또는 ``ContinuousFlow`` subclass
     - integration method에서 pyna geometry를 반환
   * - 새 map family
     - ``CallableMap`` 또는 ``DiscreteMap`` subclass
     - 안정적인 coordinate name 노출
   * - 새 section
     - ``pyna.topo.section.Section`` 스타일 객체
     - crossing/project 의미를 명확히 구현
   * - 새 data format
     - ``pyna.topo.adapters``
     - 데이터를 정규화하고, periodicity를 조용히 주장하지 않기
   * - 새 assembly policy
     - ``pyna.topo.builders``
     - validation과 metadata를 중앙화
   * - 새 backend selection
     - factories 또는 workflow facade
     - raw backend array는 pyna object 뒤에 숨기기

경험칙은 다음과 같습니다. 수학적 객체에는 dataclass를, 입력 정규화에는
adapter를, 검증에는 builder를 사용하고, 안정적인 문자열 key가 필요한
사용자에게만 factory를 사용하세요.

Notebook 체크리스트
-------------------

문서를 게시하기 전:

.. code-block:: bash

   .venv/bin/python -m pytest --nbmake \
     notebooks/tutorials/RMP_resonance_analysis.ipynb \
     notebooks/tutorials/island_jacobian_analysis.ipynb

저장된 출력이 있는 무거운 notebook은 로컬에서 실행하고 갱신된 ``.ipynb`` 파일을
commit하세요.

.. code-block:: bash

   .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
     notebooks/tutorials/sde_monte_carlo_distribution.ipynb

GitHub Pages에서 쓰는 같은 notebook 집합으로 Sphinx 빌드를 로컬 실행하려면:

.. code-block:: bash

   rm -rf docs/notebooks docs/_build
   cp -r notebooks docs/notebooks
   make -C docs html SPHINXBUILD=../.venv/bin/sphinx-build
