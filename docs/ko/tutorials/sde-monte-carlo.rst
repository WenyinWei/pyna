SDE 몬테카를로 분포
====================

이 tutorial은 pyna에서의 실용적인 SDE workflow를 보여 줍니다.

1. ``BrownianMotion``, ``GeometricBrownianMotion`` 또는 ``ItoSDE`` 로
   Itô 모델을 정의한다.
2. 재현 가능한 표본 경로를 ``Trajectory`` 로 생성한다.
3. 벡터화된 몬테카를로 ensemble을 실행해 분포 추정치를 얻는다.
4. 가능하면 경험적 평균, 분산, 분위수를 해석식과 비교한다.

pyna의 SDE class는 모델의 책임 범위와 단일 경로 기하에 사용하세요. pyna에
전용 ensemble geometry class가 생기기 전까지는 큰 ensemble에 벡터화된 NumPy
array를 사용하세요. 이렇게 하면 수학적 object model을 정직하게 유지할 수 있습니다.
단일 realization은 표본 trajectory이고, realization cloud는 통계 추정량입니다.

.. note::

   아래의 executable notebook은 저장된 output과 함께 commit되어 있으며
   ``nbsphinx`` execution이 비활성화되어 있습니다. 수치 파라미터를 바꿀 때는
   로컬에서 다시 실행하세요. docs workflow는 GitHub Pages에서 저장된 output을
   렌더링합니다.

실행 가능한 notebook:

- :doc:`/notebooks/i18n/ko/tutorials/sde_monte_carlo_distribution`

복사해 쓸 수 있는 패턴
---------------------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution is a pyna Trajectory

   n_paths = 200_000
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=n_paths)
   log_terminal = (
       np.log(100.0)
       + gbm.expected_log_growth()[0] * 1.0
       + gbm.sigma[0] * np.sqrt(1.0) * z
   )
   terminal = np.exp(log_terminal)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

확장 참고
---------

- ``ItoSDE.diffusion_matrix`` 는 scalar, vector, matrix diffusion을 받습니다.
- ``ItoSDE.euler_maruyama`` 는 외부에서 제공한 ``dW`` increment를 받으므로
  common-random-number experiment와 regression test를 deterministic하게 만들 수
  있습니다.
- 기하학적 주장이 의미 있을 때만 하나의 표본 경로를 위상 객체로 승격하세요.
  몬테카를로 표본은 분포를 추정할 뿐이며, 자동으로 invariant set이 되지
  않습니다.
