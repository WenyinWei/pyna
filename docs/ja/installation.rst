インストール
============

対応 Python バージョン
----------------------

``pyna-chaos`` は Linux、macOS、Windows 上の CPython 3.9 から 3.13 までを
サポートします。中核となる Python 依存関係は NumPy、SciPy、Matplotlib、SymPy、
h5py、joblib、Plotly です。Prefect によるオーケストレーションと CUDA 加速は
任意機能です。

PyPI から
---------

対象プラットフォーム向けの wheel が公開されている場合は、それを使ってください。

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pyna-chaos

wheel には必須の ``cyna`` C++ 拡張が含まれます。``pyna._cyna`` 拡張が見つからない
場合は、通常の任意バックエンド状態ではなく、インストール上の問題として扱って
ください。

インストールを確認します。

.. code-block:: python

   import pyna
   from pyna._cyna import get_version, is_available

   print(pyna.__version__)
   print(is_available(), get_version())

Prefect オーケストレーションはコアパッケージには含まれません。Prefect ベースの
workflow が必要な場合は workflow extra をインストールします。

.. code-block:: bash

   python -m pip install "pyna-chaos[workflow]"

workflow の trajectory/orbit キャッシュは、pyna が管理するバージョン付き payload
として保存されます。Prefect はオーケストレーションに使われるもので、永続キャッシュ
ファイル形式ではありません。

ソースから
----------

editable/source インストールでは、``setup.py`` から xmake を通して ``cyna`` を
ビルドします。

.. code-block:: bash

   git clone https://github.com/WenyinWei/pyna.git
   cd pyna
   python -m pip install --upgrade pip
   python -m pip install -e .

ソースビルドには次が必要です。

- C++17 コンパイラ: GCC 9+、Clang 10+、Apple Clang、または MSVC 2019+
- xmake 2.8+
- pybind11 ヘッダー。通常は pip によりインストールされます

ビルドスクリプトは一般的なプラットフォームで xmake と最小限のコンパイラツール
チェーンを bootstrap しようとします。制限の厳しい CI イメージでは事前にそれらを
インストールし、ツールがない場合にすばやく失敗するよう
``CYNA_SKIP_TOOL_INSTALL=1`` を設定してください。

cyna C++ 加速
-------------

``cyna`` は、磁力線追跡、Poincare 写像、固定点スキャン、コイル場、壁/接続長
スキャン、関数摂動理論カーネルで使われる C++ 層です。Python/C++ 境界での正準的な
成分順序は次のとおりです。

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

手動の低レベルビルド:

.. code-block:: bash

   cd cyna
   xmake config --yes --mode=release --require=no --with-cuda=n
   xmake build cyna_python

xmake の ``after_build`` hook は ``_cyna_ext.so`` または ``_cyna_ext.pyd`` を
``pyna/_cyna`` にコピーします。アプリケーションコードでは、生の拡張を直接 import
するのではなく、``pyna.flt``、``pyna.toroidal.flt``、``pyna.topo``、
``pyna._cyna`` から高レベル wrapper を import してください。

CUDA
----

公開 wheel は CPU のみです。ローカルのソースビルドでは、``CYNA_WITH_CUDA=0`` が
設定されていない限り、``nvcc`` が利用可能なときに独立した CUDA バックエンドが
自動で有効になります。

よく使うモード:

.. code-block:: bash

   CYNA_WITH_CUDA=0 python -m pip install -e .  # force CPU-only
   CYNA_WITH_CUDA=1 python -m pip install -e .  # require CUDA backend build

メインの ``_cyna_ext`` モジュールは CUDA にリンクしません。CUDA コードは、CUDA
対応のコイル場呼び出しが行われたときだけ読み込まれます。

開発用インストール
------------------

テスト、notebook、ドキュメント用:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"
   pytest

ドキュメントをローカルでビルドします。

.. code-block:: bash

   cd docs
   cp -r ../notebooks notebooks
   make html

トラブルシューティング
----------------------

``ImportError: pyna._cyna requires the compiled cyna extension``
   PyPI からプラットフォーム wheel をインストールするか、xmake と C++17 コンパイラで
   ソースから再ビルドしてください。

``xmake: command not found``
   xmake を手動でインストールしてから、``python -m pip install -e .`` を再実行して
   ください。

``pybind11 headers not found``
   pyna のビルドに使う同じ環境で ``python -m pip install pybind11`` を実行して
   ください。

CUDA build fails but CPU is acceptable
   ``CYNA_WITH_CUDA=0`` で再ビルドしてください。
