Установка
=========

Поддерживаемые версии Python
----------------------------

``pyna-chaos`` поддерживает CPython 3.9-3.13 в Linux, macOS и Windows. Основные
зависимости Python: NumPy, SciPy, Matplotlib, SymPy, h5py, joblib и Plotly.
Оркестрация Prefect и ускорение CUDA являются необязательными.

Из PyPI
-------

Используйте опубликованный wheel, когда он доступен для вашей платформы:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

Wheel включает требуемое C++-расширение ``cyna``. Отсутствие расширения
``pyna._cyna`` следует рассматривать как проблему установки, а не как обычное
состояние необязательного backend.

Проверьте установку:

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

Оркестрация Prefect не устанавливается базовым пакетом. Установите extra
workflow, когда нужны рабочие процессы на базе Prefect:

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

Кеши траекторий/орбит workflow хранятся как версионированные payload, которыми
управляет pyna. Prefect используется для оркестрации; он не является долговечным
форматом файлов кеша.

Из исходного кода
-----------------

Редактируемые установки и установки из исходников собирают ``cyna`` через xmake
из ``setup.py``:

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

Для сборки из исходников нужны:

- компилятор C++17: GCC 9+, Clang 10+, Apple Clang или MSVC 2019+
- xmake 2.8+
- заголовки pybind11, обычно устанавливаемые через pip

Скрипт сборки пытается автоматически подготовить xmake и минимальную цепочку
компилятора на распространенных платформах. В заблокированных CI-образах
установите их заранее и задайте ``CYNA_SKIP_TOOL_INSTALL=1``, чтобы быстро
получать ошибку при отсутствии инструмента.

C++-ускорение cyna
------------------

``cyna`` - это C++-слой, используемый трассировкой силовых линий, отображениями
Пуанкаре, сканированием неподвижных точек, полями катушек, сканированием
стенки/длины соединения и ядрами функциональной теории возмущений.
Канонический порядок компонент на границе Python/C++:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Ручная низкоуровневая сборка:

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

Hook xmake ``after_build`` копирует ``_cyna_ext.so`` или ``_cyna_ext.pyd`` в
``pyna/_cyna``. Прикладной код должен импортировать высокоуровневые wrappers из
``pyna.flt``, ``pyna.toroidal.flt``, ``pyna.topo`` и ``pyna._cyna``, а не
импортировать сырое расширение напрямую.

CUDA
----

Опубликованные wheels работают только на CPU. Локальные сборки из исходников
автоматически включают отдельный CUDA backend, когда доступен ``nvcc``, если не
задано ``CYNA_WITH_CUDA=0``.

Полезные режимы:

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

Основной модуль ``_cyna_ext`` не линкуется с CUDA. CUDA-код загружается только
при вызове поля катушки, поддерживающего CUDA.

Установка для разработки
------------------------

Для тестов, notebooks и документации:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

Соберите документацию локально:

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

Диагностика проблем
-------------------

``ImportError: pyna._cyna requires the compiled cyna extension``
   Установите platform wheel из PyPI или пересоберите из исходников с xmake и
   компилятором C++17.

``xmake: command not found``
   Установите xmake вручную, затем снова выполните
   ``python -m pip install -e .``.

``pybind11 headers not found``
   Выполните ``python -m pip install pybind11`` в той же среде, которая
   используется для сборки pyna.

CUDA build fails but CPU is acceptable
   Пересоберите с ``CYNA_WITH_CUDA=0``.
