pyna — Python DYNAmics
======================

.. image:: https://img.shields.io/pypi/v/pyna-chaos?color=blue&label=PyPI
   :target: https://pypi.org/project/pyna-chaos/
.. image:: https://img.shields.io/pypi/pyversions/pyna-chaos
.. image:: https://img.shields.io/badge/license-GPL--3.0-green
.. image:: https://github.com/WenyinWei/pyna/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/WenyinWei/pyna/actions

**pyna** is a Python library for **dynamical systems analysis** and
**magnetic confinement fusion (MCF) physics** — from field-line tracing and
Poincaré maps to analytic torus-deformation theory and publication-quality
tokamak visualizations.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔀 Field-line Tracing
      :link: en/api/flt
      :link-type: doc

      RK4 integrator with optional CUDA acceleration (118× speedup).

   .. grid-item-card:: 🌀 Poincaré Maps & Islands
      :link: en/api/topo
      :link-type: doc

      Multi-section maps, X/O-point detection, island width extraction.

   .. grid-item-card:: 🗺️ Manifold Visualization
      :link: en/api/MCF
      :link-type: doc

      Publication-quality stable/unstable manifold plots for tokamaks.

   .. grid-item-card:: 📡 Torus Deformation
      :link: en/api/MCF
      :link-type: doc

      Non-resonant BNF-derived analytic spectral theory (Wei 2025).

   .. grid-item-card:: 🧲 MCF Equilibria
      :link: en/api/MCF
      :link-type: doc

      Solov'ev, Grad-Shafranov, stellarator analytic/numeric solutions.

   .. grid-item-card:: 📐 Magnetic Coordinates
      :link: en/api/MCF
      :link-type: doc

      PEST, Boozer, Hamada, Equal-arc transformations.

.. toctree::
   :maxdepth: 2
   :caption: English Documentation

   en/installation
   en/quickstart
   en/api/index
   en/tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: 中文文档

   zh/installation
   zh/quickstart
   zh/api/index
