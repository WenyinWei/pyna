磁力線追跡（``pyna.flt``）
==========================

``pyna.flt`` パッケージは、抽象 :mod:`pyna.system` 階層の上に構築された磁力線積分
ルーチンを提供します。

**バックエンド:**

- **CPU/serial** -- 純 Python RK4 積分器
- **CPU/parallel** -- マルチプロセスまたはマルチスレッド変種
- **CUDA** -- CuPy による任意の GPU バックエンド（最大 118× 高速化）
- **OpenCL** -- 実験的

.. contents:: サブモジュール
   :depth: 2
   :local:

----

Core Tracer
-----------

.. automodule:: pyna.flt
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

力学系階層
----------

.. automodule:: pyna.system
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
