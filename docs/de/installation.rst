Installation
============

Unterstützte Python-Versionen
-----------------------------

``pyna-chaos`` unterstützt CPython 3.9 bis 3.13 unter Linux, macOS und
Windows.  Die zentralen Python-Abhängigkeiten sind NumPy, SciPy, Matplotlib,
SymPy, h5py, joblib und Plotly.  Prefect-Orchestrierung und CUDA-Beschleunigung
sind optional.

Von PyPI
--------

Verwenden Sie das veröffentlichte Wheel, sofern eines für Ihre Plattform
verfügbar ist:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

Das Wheel enthält die erforderliche C++-Erweiterung ``cyna``.  Eine fehlende
Erweiterung ``pyna._cyna`` sollte als Installationsproblem behandelt werden,
nicht als normaler Zustand eines optionalen Backends.

Installation prüfen:

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

Die Prefect-Orchestrierung wird nicht mit dem Kernpaket installiert.
Installieren Sie das Workflow-Extra, wenn Sie Prefect-gestützte Workflows
benötigen:

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

Workflow-Caches für Trajektorien und Orbits werden als von pyna verwaltete,
versionierte Nutzdaten gespeichert.  Prefect dient der Orchestrierung; es ist
nicht das dauerhafte Cache-Dateiformat.

Aus dem Quellcode
-----------------

Editierbare Installationen und Quellinstallationen bauen ``cyna`` über
``setup.py`` mit xmake:

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

Quell-Builds benötigen:

- einen C++17-Compiler: GCC 9+, Clang 10+, Apple Clang oder MSVC 2019+
- xmake 2.8+
- pybind11-Header, normalerweise über pip installiert

Das Build-Skript versucht, xmake und eine minimale Compiler-Toolchain auf
gängigen Plattformen automatisch bereitzustellen.  In abgeschotteten
CI-Images sollten diese Werkzeuge vorinstalliert werden; setzen Sie
``CYNA_SKIP_TOOL_INSTALL=1``, damit der Build bei fehlenden Werkzeugen früh
fehlschlägt.

cyna-C++-Beschleunigung
-----------------------

``cyna`` ist die C++-Schicht für Feldlinienverfolgung, Poincare-Karten,
Fixpunktscans, Spulenfelder, Wand- und Connection-Length-Scans sowie Kernel der
funktionalen Störungstheorie.  Die kanonische Komponentenreihenfolge an der
Python/C++-Grenze lautet:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Manueller Low-Level-Build:

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

Der xmake-Hook ``after_build`` kopiert ``_cyna_ext.so`` oder ``_cyna_ext.pyd``
nach ``pyna/_cyna``.  Anwendungscode sollte die High-Level-Wrapper aus
``pyna.flt``, ``pyna.toroidal.flt``, ``pyna.topo`` und ``pyna._cyna``
importieren, statt die rohe Erweiterung direkt zu importieren.

CUDA
----

Veröffentlichte Wheels sind nur für die CPU gebaut.  Lokale Quell-Builds
aktivieren das separate CUDA-Backend automatisch, wenn ``nvcc`` verfügbar ist,
sofern nicht ``CYNA_WITH_CUDA=0`` gesetzt ist.

Nützliche Modi:

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

Das Hauptmodul ``_cyna_ext`` linkt nicht gegen CUDA.  CUDA-Code wird erst
geladen, wenn ein CUDA-fähiger Aufruf für Spulenfelder erfolgt.

Entwicklungsinstallation
------------------------

Für Tests, Notebooks und Dokumentation:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

Dokumentation lokal bauen:

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

Fehlersuche
-----------

``ImportError: pyna._cyna requires the compiled cyna extension``
   Installieren Sie ein Plattform-Wheel von PyPI oder bauen Sie aus dem
   Quellcode mit xmake und einem C++17-Compiler neu.

``xmake: command not found``
   Installieren Sie xmake manuell und führen Sie anschließend
   ``python -m pip install -e .`` erneut aus.

``pybind11 headers not found``
   Führen Sie ``python -m pip install pybind11`` in derselben Umgebung aus, in
   der pyna gebaut wird.

CUDA build fails but CPU is acceptable
   Bauen Sie mit ``CYNA_WITH_CUDA=0`` neu.
