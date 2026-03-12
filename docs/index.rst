pyna — Python Dynamical Systems Library
=========================================

.. image:: https://img.shields.io/badge/version-0.1.0-blue
.. image:: https://img.shields.io/badge/python-3.10+-green
.. image:: https://img.shields.io/badge/license-MIT-lightgrey

**pyna** is a Python library for dynamical systems analysis and magnetic confinement fusion (MCF) plasma physics research.

**核心特性 / Core Features**

- 🔀 **Field Line Tracing (FLT)** — RK4 integrator, CUDA acceleration (118× speedup)
- 🌀 **Poincaré Maps & Island Analysis** — X/O point detection, island width extraction
- 🎯 **FPT Topology Control** — Functional Perturbation Theory for magnetic topology control
- 🧲 **MCF Equilibria** — Solov'ev, GS solver, stellarator configurations
- 📐 **Magnetic Coordinates** — PEST, Boozer, Hamada, Equal-arc
- ⚡ **C++ Acceleration (cyna)** — Optional C++ backend for performance-critical ops

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
