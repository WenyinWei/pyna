튜토리얼과 예제
===============

공개 노트북은 워크플로별로 묶어 두었습니다. 문서 빌드는 ``notebooks/`` 를
Sphinx 소스 트리로 복사하므로, 아래 경로는 저장소 배치를 반영합니다.

권장 학습 경로
--------------

:doc:`/ko/quickstart` 에서 시작한 뒤 일반 기하 워크플로, 확률 모델,
그리고 토로이달 Monodromy/RMP 예제로 진행하세요.

1. :doc:`/ko/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/ko/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/ko/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/ko/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/ko/tutorials/RMP_island_validation_solovev`

일반 동역학계
-------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

일반 기하 워크플로와 해석적 스텔러레이터 고정점 워크플로는 이제
독립적인 텍스트 전용 노트북으로 게시되지 않고 RMP 공명 튜토리얼 안에
통합되어 있습니다. 해당 튜토리얼은 같은 승격 과정을 보여 줍니다:
표본 교차점 -> 고정점 기하 -> X/O 분류 -> 다양체 및 좌표 격자 오버레이.

확률 미분방정식
--------------

SDE 튜토리얼은 로컬에서 미리 실행되어 있습니다. 분포 추정은 수만 또는 수십만 개의
몬테카를로 경로를 사용하는 일이 많기 때문입니다. GitHub Pages는 무거운 표본추출
셀에 CI 시간을 쓰는 대신 저장된 출력을 렌더링합니다.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/ko/tutorials/sde_monte_carlo_distribution

자기 좌표와 평형
----------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ko/tutorials/magnetic_coordinates_comparison

RMP, 자기섬, 푸앵카레 분석
--------------------------

자기 위상을 연구할 때는 공명 분석 노트북에서 시작하세요. 현재 이
노트북은 발산이 없는 RMP 템플릿, 중요한 ``m=1`` 가지, ``cyna``
고정점 검증, 다성분 반변 ``B^r`` 자기 스펙트럼 아틀라스,
선택적 Poincaré 및 자기섬 오버레이를 포함하는 모듈식 ``q``/``m/n`` 공명 지도,
혼합 RMP/nRMP 스펙트럼, 모든 비공명 모드에서 얻은 전체 nRMP 응답,
자기력선 속도 변조, 섭동 차수 점검을 다룹니다.

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ko/tutorials/RMP_resonance_analysis
   /notebooks/i18n/ko/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/ko/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` 는 공명 분석 워크플로의 실행/캐시
변형으로 저장소에 유지되지만, 공개 문서는 위의 설명형 버전으로 연결됩니다.

모노드로미와 다양체
------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/ko/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/ko/tutorials/monodromy_xcycle_analytic

고전 및 일반 동역학계
---------------------

저장소에는 ``notebooks/examples`` 아래에도 가벼운 노트북이 있습니다:
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` 및
``FPT_DX_to_DP_sympy.ipynb`` 입니다. 이들은 일부가 섹션 제목이 없는
초안형 노트북이므로 실행된 문서 페이지가 아니라 소스 예제로
유지됩니다.

정적 튜토리얼 그림
------------------

몇몇 긴 워크플로는 저장소의 ``notebooks/tutorials`` 아래에 정적 그림과
생성된 출력으로 표현되어 있습니다. q 프로파일 진단,
PEST/Boozer/Hamada/equal-arc 좌표, 자기섬 억제 스캔, 위상 제어,
Poincaré 다양체, Solov'ev single-null 예제를 다룹니다.
