pyna - Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-LGPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** est une bibliothèque Python pour l'**analyse des systèmes dynamiques**
et la **physique de la fusion par confinement magnétique**. Elle couvre le
traçage de lignes de champ, les cartes de Poincaré, les systèmes hamiltoniens,
les interactions à N corps, les applications de dimension finie, les SDE
d'Ito, ainsi que le vocabulaire topologique commun qui permet de promouvoir des
données échantillonnées en géométrie d'espace des phases.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Démarrer rapidement
      :link: quickstart
      :link-type: doc

      Installer, vérifier cyna, puis exécuter les premiers exemples toroïdaux
      et de dynamique générale.

   .. grid-item-card:: Mini-cas
      :link: mini-cases
      :link-type: doc

      Recettes courtes pour les ODE, les systèmes hamiltoniens, les cartes,
      les SDE et la promotion topologique.

   .. grid-item-card:: Tutoriels
      :link: tutorials/index
      :link-type: doc

      Notebooks exécutés et guides narratifs, dont l'estimation de
      distributions de SDE par Monte Carlo.

   .. grid-item-card:: Référence API
      :link: api/index
      :link-type: doc

      Guides de modules rédigés manuellement, plus la référence source générée.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   installation
   quickstart
   mini-cases
   tutorials/index
   api/index
   theory/index
   development/index
