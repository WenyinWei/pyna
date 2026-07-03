설치
====

지원 Python 버전
----------------

``pyna-chaos`` 는 Linux, macOS, Windows에서 CPython 3.9부터 3.13까지
지원합니다. 핵심 Python 의존성은 NumPy, SciPy, Matplotlib, SymPy,
h5py, joblib, Plotly입니다. Prefect 오케스트레이션과 CUDA 가속은
선택 사항입니다.

PyPI에서 설치
-------------

사용 중인 플랫폼에 배포 wheel이 있으면 그 wheel을 사용하는 것이 좋습니다.

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

wheel에는 필요한 ``cyna`` C++ 확장이 포함되어 있습니다. ``pyna._cyna``
확장이 없으면 정상적인 선택형 백엔드 상태가 아니라 설치 문제로 보아야 합니다.

설치를 확인합니다.

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

Prefect 오케스트레이션은 핵심 패키지에 포함되어 설치되지 않습니다.
Prefect 기반 워크플로가 필요할 때 workflow extra를 설치하세요.

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

워크플로 trajectory/orbit 캐시는 pyna가 관리하는 버전 지정 payload로
저장됩니다. Prefect는 오케스트레이션에 쓰이며, 영속 캐시 파일 형식이
아닙니다.

소스에서 설치
-------------

editable/source 설치는 ``setup.py`` 를 통해 xmake로 ``cyna`` 를 빌드합니다.

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

소스 빌드에는 다음이 필요합니다.

- C++17 컴파일러: GCC 9+, Clang 10+, Apple Clang 또는 MSVC 2019+
- xmake 2.8+
- pybind11 헤더. 보통 pip로 설치됩니다.

빌드 스크립트는 일반적인 플랫폼에서 xmake와 최소 컴파일러 toolchain을
부트스트랩하려고 시도합니다. 제한된 CI 이미지에서는 이를 미리 설치하고
``CYNA_SKIP_TOOL_INSTALL=1`` 을 설정해 도구가 없을 때 빠르게 실패하게 하세요.

cyna C++ 가속
-------------

``cyna`` 는 자력선 추적, 푸앵카레 맵, 고정점 스캔, 코일장,
벽/connection-length 스캔, 함수적 섭동 이론 커널에 사용되는 C++ 계층입니다.
Python/C++ 경계에서 표준 성분 순서는 다음과 같습니다.

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

수동 저수준 빌드:

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

xmake ``after_build`` hook은 ``_cyna_ext.so`` 또는 ``_cyna_ext.pyd`` 를
``pyna/_cyna`` 로 복사합니다. 애플리케이션 코드는 원시 확장을 직접
import하기보다 ``pyna.flt``, ``pyna.toroidal.flt``, ``pyna.topo``,
``pyna._cyna`` 에서 고수준 wrapper를 import해야 합니다.

CUDA
----

배포 wheel은 CPU 전용입니다. 로컬 소스 빌드는 ``CYNA_WITH_CUDA=0`` 이
설정되어 있지 않고 ``nvcc`` 를 사용할 수 있으면 별도의 CUDA 백엔드를
자동으로 활성화합니다.

유용한 모드:

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

주 ``_cyna_ext`` 모듈은 CUDA에 링크하지 않습니다. CUDA 코드는 CUDA를
사용할 수 있는 coil-field 호출이 이루어질 때만 로드됩니다.

개발 설치
---------

테스트, notebook, 문서 작업용:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

문서를 로컬에서 빌드합니다.

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

문제 해결
---------

``ImportError: pyna._cyna requires the compiled cyna extension``
   PyPI에서 플랫폼 wheel을 설치하거나 xmake와 C++17 컴파일러로 소스에서
   다시 빌드하세요.

``xmake: command not found``
   xmake를 수동으로 설치한 뒤 ``python -m pip install -e .`` 를 다시 실행하세요.

``pybind11 headers not found``
   pyna를 빌드하는 데 사용하는 같은 환경에서 ``python -m pip install pybind11``
   을 실행하세요.

CUDA 빌드가 실패하지만 CPU만으로 충분한 경우
   ``CYNA_WITH_CUDA=0`` 으로 다시 빌드하세요.
