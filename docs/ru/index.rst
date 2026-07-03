pyna - Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-LGPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** - это библиотека Python для **анализа динамических систем** и
**физики магнитного удержания плазмы**. Она охватывает трассировку магнитных
силовых линий, отображения Пуанкаре, гамильтоновы системы, взаимодействия
N-тел, конечномерные отображения, SDE Ито, а также общий топологический словарь,
который используется для перевода дискретно выбранных данных в геометрию
фазового пространства.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Быстрый старт
      :link: quickstart
      :link-type: doc

      Установите пакет, проверьте cyna и запустите первые тороидальные и
      общединамические примеры.

   .. grid-item-card:: Мини-кейсы
      :link: mini-cases
      :link-type: doc

      Короткие рецепты для ODE, гамильтоновых систем, отображений, SDE и
      топологического повышения данных.

   .. grid-item-card:: Учебные материалы
      :link: tutorials/index
      :link-type: doc

      Выполненные notebooks и пояснительные руководства, включая оценку
      распределений SDE методом Монте-Карло.

   .. grid-item-card:: Справочник API
      :link: api/index
      :link-type: doc

      Написанные вручную руководства по модулям и сгенерированная справка по
      исходному коду.

.. toctree::
   :maxdepth: 2
   :caption: Документация

   installation
   quickstart
   mini-cases
   tutorials/index
   api/index
   theory/index
   development/index
