Tutorials and Examples
======================

The public notebooks are grouped by workflow.  The documentation build copies
``notebooks/`` into the Sphinx source tree, so paths below mirror the repository
layout.

Recommended Learning Path
-------------------------

Start with :doc:`/en/quickstart`, then work through the generic geometry
workflow before moving to toroidal monodromy and RMP examples:

1. :doc:`/en/mini-cases`
2. :doc:`/notebooks/tutorials/general_dynamics_geometry_patterns`
3. :doc:`/notebooks/tutorials/analytic_stellarator_geometry_workflow`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

General Dynamical Systems
-------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/general_dynamics_geometry_patterns
   /notebooks/tutorials/analytic_stellarator_geometry_workflow

Magnetic Coordinates and Equilibria
-----------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMPs, Islands and Poincare Analysis
-----------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` is kept in the repository as an execution/cache
variant of the resonance analysis workflow, but the public documentation links
to the explanatory version above.

Monodromy and Manifolds
-----------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

Classical and General Dynamical Systems
---------------------------------------

The repository also includes lightweight notebooks under ``notebooks/examples``:
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` and
``FPT_DX_to_DP_sympy.ipynb``.  They are kept as source examples rather than
executed documentation pages because several are scratch-style notebooks without
section titles.

Static Tutorial Figures
-----------------------

Several longer workflows are represented in the repository as static figures
and generated outputs under ``notebooks/tutorials``.  They cover q-profile
diagnostics, PEST/Boozer/Hamada/equal-arc coordinates, island suppression scans,
phase control, Poincare manifolds and Solov'ev single-null examples.
