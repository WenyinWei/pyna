pyna - Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-LGPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** は **力学系解析** と **磁場閉じ込め核融合物理** のための Python
ライブラリです。磁力線追跡、Poincare 写像、Hamiltonian 系、N-body 相互作用、
有限次元写像、Ito SDE、そしてサンプルデータを相空間幾何へ持ち上げるときに使う
共通のトポロジー語彙を扱います。

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: すぐに始める
      :link: quickstart
      :link-type: doc

      インストール、cyna の確認、最初のトロイダル例と一般力学系例を実行します。

   .. grid-item-card:: ミニケース
      :link: mini-cases
      :link-type: doc

      ODE、Hamiltonian 系、写像、SDE、トポロジーへの持ち上げの短いレシピです。

   .. grid-item-card:: チュートリアル
      :link: tutorials/index
      :link-type: doc

      Monte Carlo SDE 分布推定を含む、実行済み notebook と解説付きガイドです。

   .. grid-item-card:: API リファレンス
      :link: api/index
      :link-type: doc

      手書きのモジュールガイドと、生成されたソースリファレンスです。

.. toctree::
   :maxdepth: 2
   :caption: ドキュメント

   installation
   quickstart
   mini-cases
   tutorials/index
   api/index
   theory/index
   development/index
