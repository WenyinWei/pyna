cyna 加速層
===========

``cyna`` は pyna に同梱される C++ 加速層です。Python のホットループでは許容できない
場面で使われます。磁力線追跡、Poincare バッチ、固定点スキャン、接続長/壁 hit、
コイル場、関数摂動理論カーネルが対象です。

ビルド契約
----------

``pyna._cyna`` は、パッケージ内にコンパイル済みの ``_cyna_ext`` バイナリがあることを
期待します。ソースインストールでは xmake でビルドし、PyPI wheel には含まれています。
プラットフォーム設定と CUDA flag については :doc:`../installation` を参照して
ください。

正準的な円筒座標場キャッシュの順序は次のとおりです。

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

``pyna.fields.VectorFieldCylind`` または legacy dict を C-contiguous 配列へ変換するには
:func:`pyna._cyna.prepare_field_cache` を使います。

高レベル API と低レベル API
---------------------------

アプリケーションコードでは高レベル wrapper を優先してください。

- 追跡には ``pyna.flt`` と ``pyna.toroidal.flt``
- Poincare 写像、cycle、島、多様体、FPT 応答には ``pyna.topo``
- コイル場構築には ``pyna.toroidal.coils``

``pyna._cyna`` を直接使うのは、bridge 境界、診断、新しい高レベル wrapper を書く場合
だけにしてください。

Python Wrapper リファレンス
---------------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

Utility Helper
--------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
