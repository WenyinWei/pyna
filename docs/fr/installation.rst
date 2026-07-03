Installation
============

Versions de Python prises en charge
-----------------------------------

``pyna-chaos`` prend en charge CPython 3.9 a 3.13 sous Linux, macOS et Windows.
Les dependances Python principales sont NumPy, SciPy, Matplotlib, SymPy, h5py,
joblib et Plotly. L'orchestration Prefect et l'accélération CUDA sont
facultatives.

Depuis PyPI
-----------

Utilisez la roue publiee lorsqu'elle est disponible pour votre plateforme :

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

La roue inclut l'extension C++ ``cyna`` requise. Une extension ``pyna._cyna``
manquante doit etre traitee comme un probleme d'installation, et non comme un
etat normal de backend facultatif.

Verifiez l'installation :

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

L'orchestration Prefect n'est pas installee par le paquet de base. Installez
l'extra workflow lorsque vous avez besoin de workflows appuyes par Prefect :

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

Les caches de trajectoires/orbites de workflow sont stockes comme charges utiles
versionnees gerees par pyna. Prefect sert à l'orchestration ; il n'est pas le
format durable des fichiers de cache.

Depuis les sources
------------------

Les installations editables/depuis les sources construisent ``cyna`` avec xmake
via ``setup.py`` :

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

Les builds depuis les sources exigent :

- un compilateur C++17 : GCC 9+, Clang 10+, Apple Clang ou MSVC 2019+
- xmake 2.8+
- les en-tetes pybind11, normalement installes par pip

Le script de build tente d'amorcer xmake et une chaine d'outils de compilation
minimale sur les plateformes courantes. Dans des images CI verrouillees,
preinstallez-les et definissez ``CYNA_SKIP_TOOL_INSTALL=1`` pour echouer
rapidement lorsqu'un outil manque.

Acceleration C++ cyna
---------------------

``cyna`` est la couche C++ utilisée par le traçage de lignes de champ, les
cartes de Poincaré, les balayages de points fixes, les champs de bobines, les
balayages mur/longueur de connexion et les noyaux de théorie de perturbation
fonctionnelle. L'ordre canonique des composantes a la frontiere Python/C++ est :

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Build manuel bas niveau :

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

Le hook xmake ``after_build`` copie ``_cyna_ext.so`` ou ``_cyna_ext.pyd`` dans
``pyna/_cyna``. Le code applicatif doit importer les wrappers de haut niveau
depuis ``pyna.flt``, ``pyna.toroidal.flt``, ``pyna.topo`` et ``pyna._cyna``
plutot que d'importer directement l'extension brute.

CUDA
----

Les roues publiees sont uniquement CPU. Les builds locaux depuis les sources
activent automatiquement le backend CUDA sépare lorsque ``nvcc`` est disponible,
sauf si ``CYNA_WITH_CUDA=0`` est defini.

Modes utiles :

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

Le module principal ``_cyna_ext`` n'est pas lie a CUDA. Le code CUDA n'est
charge que lorsqu'un appel de champ de bobine compatible CUDA est effectue.

Installation de développement
-----------------------------

Pour les tests, notebooks et la documentation :

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

Construisez la documentation localement :

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

Depannage
---------

``ImportError: pyna._cyna requires the compiled cyna extension``
   Installez une roue de plateforme depuis PyPI ou reconstruisez depuis les
   sources avec xmake et un compilateur C++17.

``xmake: command not found``
   Installez xmake manuellement, puis relancez ``python -m pip install -e .``.

``pybind11 headers not found``
   Executez ``python -m pip install pybind11`` dans le même environnement que
   celui utilise pour construire pyna.

CUDA build fails but CPU is acceptable
   Reconstruisez avec ``CYNA_WITH_CUDA=0``.
