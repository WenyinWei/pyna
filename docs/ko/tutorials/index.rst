튜토리얼과 예제
===============

공개 notebook은 workflow별로 묶여 있습니다. 문서 빌드는 ``notebooks/`` 를
Sphinx source tree로 복사하므로, 아래 경로는 repository layout을 반영합니다.

권장 학습 경로
--------------

:doc:`/ko/quickstart` 에서 시작한 뒤 generic geometry workflow, stochastic model,
그리고 toroidal monodromy/RMP 예제로 진행하세요.

1. :doc:`/ko/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

일반 동역학계
-------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

generic geometry workflow와 analytic stellarator fixed-point workflow는 이제
독립적인 text-only notebook으로 게시되지 않고 RMP resonance tutorial 안에
통합되어 있습니다. 해당 tutorial은 같은 promotion chain을 보여 줍니다:
sampled crossings -> fixed-point geometry -> X/O classification -> manifold
and coordinate-grid overlays.

확률 미분방정식
--------------

SDE tutorial은 로컬에서 미리 실행되어 있습니다. 분포 추정은 수만 또는 수십만 개의
Monte Carlo path를 사용하는 일이 많기 때문입니다. GitHub Pages는 무거운 sampling
cell에 CI 시간을 쓰는 대신 저장된 output을 render합니다.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

자기 좌표와 평형
----------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMP, 자기섬, 푸앵카레 분석
--------------------------

자기 topology를 연구할 때는 resonance analysis notebook에서 시작하세요. 현재 이
notebook은 divergence-free RMP template, 중요한 ``m=1`` branch, ``cyna``
fixed-point validation, 다성분 contravariant ``B^r`` magnetic-spectrum atlas,
선택적 Poincare 및 island overlay를 포함하는 modular ``q``/``m/n`` resonance map,
mixed RMP/nRMP spectrum, 모든 비공명 mode에서 얻은 total nRMP response,
field-line speed modulation, perturbation-order check를 다룹니다.

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` 는 resonance analysis workflow의 execution/cache
variant로 repository에 유지되지만, 공개 문서는 위의 설명형 version으로 연결됩니다.

모노드로미와 다양체
------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

고전 및 일반 동역학계
---------------------

repository에는 ``notebooks/examples`` 아래에도 가벼운 notebook이 있습니다:
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` 및
``FPT_DX_to_DP_sympy.ipynb`` 입니다. 이들은 일부가 section title이 없는
scratch-style notebook이므로 실행된 문서 페이지가 아니라 source example로
유지됩니다.

정적 Tutorial 그림
------------------

몇몇 긴 workflow는 repository의 ``notebooks/tutorials`` 아래에 static figure와
generated output으로 표현되어 있습니다. q-profile diagnostics,
PEST/Boozer/Hamada/equal-arc coordinates, island suppression scan, phase control,
Poincare manifold, Solov'ev single-null example을 다룹니다.
